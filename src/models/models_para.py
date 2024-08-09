import numpy as np
import torch
import torch.nn as nn 

from .blocks import *
from .swin_transformer import SwinTransformerV2_1D
from .loss import *
from ..utils import fetch_feats_by_index, compute_tiou


class SOONet(nn.Module):

    def __init__(self, cfg):
        
        super().__init__()
        nscales = cfg.MODEL.SCALE_NUM #4
        hidden_dim = cfg.MODEL.HIDDEN_DIM
        snippet_length = cfg.MODEL.SNIPPET_LENGTH  #10
        enable_stage2 = cfg.MODEL.ENABLE_STAGE2  #参数为true
        stage2_pool = cfg.MODEL.STAGE2_POOL  #在论文中对应计算内部帧和文本相似度之后的average操作
        stage2_topk = cfg.MODEL.STAGE2_TOPK  #100
        topk = cfg.TEST.TOPK  #100

        self.video_encoder = SwinTransformerV2_1D(
                                patch_size=snippet_length, 
                                in_chans=hidden_dim, 
                                embed_dim=hidden_dim, 
                                depths=[2]*nscales, 
                                num_heads=[8]*nscales,
                                window_size=[64]*nscales,   #参数是[64,64,64,64]
                                mlp_ratio=2., 
                                qkv_bias=True,
                                drop_rate=0., 
                                attn_drop_rate=0., 
                                drop_path_rate=0.1,
                                norm_layer=nn.LayerNorm, 
                                patch_norm=True,
                                use_checkpoint=False, 
                                pretrained_window_sizes=[0]*nscales
                            )
        
        self.q2v_stage1 = Q2VRankerStage1(nscales, hidden_dim)  #anchor rank optimization 4，512
        self.v2q_stage1 = V2QRankerStage1(nscales, hidden_dim)  #query rank optimization
        if enable_stage2:
            self.q2v_stage2 = Q2VRankerStage2(nscales, hidden_dim, snippet_length, stage2_pool)
            self.v2q_stage2 = V2QRankerStage2(nscales, hidden_dim)
        self.regressor = BboxRegressor(hidden_dim, enable_stage2)
        self.rank_loss = ApproxNDCGLoss(cfg)
        self.reg_loss = IOULoss(cfg)

        self.nscales = nscales
        self.enable_stage2 = enable_stage2
        self.stage2_topk = stage2_topk  #在测试阶段才使用了top-k 100
        self.cfg = cfg
        self.topk = topk  # 100
        self.enable_nms = cfg.MODEL.ENABLE_NMS


    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    

    def forward_train(self, 
                      query_feats=None,
                      query_masks=None,
                      video_feats=None,  #在dataloaer中给出了0号维度bsz值为1 
                      start_ts=None,  #bsz,num_all_anchors
                      end_ts=None,
                      scale_boundaries=None,
                      overlaps=None,
                      timestamps=None,
                      anchor_masks=None,
                      **kwargs):

        sent_feat = query_feats #bsz 一个视频下选择了多个query，每一行都是一个query
        #ctx表示context-based? 经过video encoder之后就变成多尺度  输入时bsz, hidden_dim, frames
        #ctx_feats是一个列表，每个元素是(1,anchor_num,hidden_dim) ，anchor_num随着尺度长度的变化而变化,1表示bsz
        ctx_feats = self.video_encoder(video_feats.permute(0, 2, 1)) #是要使用完整的长视频特征，每次dataloader只拿一个视频
        qv_ctx_scores = self.q2v_stage1(ctx_feats, sent_feat)  #得到每个query和所有尺度的分数，pre-ranking阶段得分
        vq_ctx_scores = self.v2q_stage1(ctx_feats, sent_feat)  #论文中在损失部分说了对ApproxNDCG更好的损失优化
        if self.enable_stage2:
            hit_indices = list()
            filtered_ctx_feats = list()
            starts = list()
            ends = list()
            stage2_overlaps = list()
            #re-ranking阶段选出有重叠的片段 （论文中不是说top-m嘛？这个参数在哪里使用了？）
            for i in range(self.nscales): #每一个尺度下进行遍历
                #经测试一个(1,80,64)和(1,160,64)的向量输出的结果窗口尺寸的确是10，20，40，80，和dataloader部分的处理结果一致
                #scale_boundaries扩充了bsz维度，但是每一行数据都是一样的，所以取第一个元素即可
                scale_first = scale_boundaries[0][i] 
                # print(len(scale_boundaries))
                scale_last = scale_boundaries[0][i+1]  #scale_boundaries是累加的
                # overlap的维度比较多，第一个维度是bsz，因此需要全部使用，第二个维度是scale，只需要部分使用
                gt = overlaps[:, scale_first:scale_last] #得到的是一维向量。overlaps是包含不同的query,所以之后的计算也得到的是不同query对应的结果？
                
                #sum(0)表示保持第一个维度要丢失，第一个维度不是bsz嘛？怎么能在这个维度上进行求和？
                #难道表示的是video-centric，只要有一个尺度和bsz中的一条样本有重叠，就会被选中参与后面的计算？
                #因为后面的filtered_ctx_feats也的确是在视频特征上进行的操作和query的bsz无关（guess）
                indices = torch.nonzero(gt.sum(0) > 0, as_tuple=True)[0]
                hit_indices.append(indices)
                #torch.index_select(input, dim, index):这个函数从 input 张量中选取指定维度 dim 上由 index 张量指定的索引的元素
                #ctx列表中每个元素第一个维度为1 ：bsz
                filtered_ctx_feats.append(torch.index_select(ctx_feats[i], 1, indices)) #当前尺度下有overlap的anchor特征，选择操作是否可微？？
                #得到的结果为bsz,selected num anchors，不同的query的start和end是相同的，仅overlap,timestamps不同
                starts.append(torch.index_select(start_ts[:, scale_first:scale_last], 1, indices))
                ends.append(torch.index_select(end_ts[:, scale_first:scale_last], 1, indices))
                
                #所有bsz中的query存在一个与anchor有相交，不代表所有query都有相交，所以会有重复性吧？
                stage2_overlaps.append(torch.index_select(overlaps[:, scale_first:scale_last], 1, indices))
            
            starts = torch.cat(starts, dim=1)  #因为不同尺度下第一个维度都是bsz，所以在第二个维度上进行合并，表示所有尺度下的anchor
            ends = torch.cat(ends, dim=1)
            stage2_overlaps = torch.cat(stage2_overlaps, dim=1)

            #使用的是原本的video特征来进行content-based特征提取，而不是在context-based的基础上，论文中也是如此
            #ctn_feats列表中四个元素，每个元素(1,selected_anchors, snippet_length, D)
            #计算ctn_feats时，还是使用视频本身的特征，没有结合query
            qv_merge_scores, qv_ctn_scores, ctn_feats = self.q2v_stage2(
                video_feats, sent_feat, hit_indices, qv_ctx_scores)  #hit_indices类似于结构图中的top-m选择
            vq_ctn_scores = self.v2q_stage2(ctn_feats, sent_feat) #论文中这个步骤不太明白
            ctx_feats = filtered_ctx_feats  #并没有使用top-m的结果，而是只要有相交部分就会被选中
        else:
            ctn_feats = None
            qv_merge_scores = qv_ctx_scores
            starts = start_ts
            ends = end_ts
            stage2_overlaps = None
        
        #ctn_feats列表中四个元素，每个元素(1,selected_anchors, snippet_length, D)
        #bbox_bias会得到(bsz_query,total_num_anchors,2)的结果
        bbox_bias = self.regressor(ctx_feats, ctn_feats, sent_feat) #结合context-based和content-based的特征进行bbox回归

        qv_ctx_scores = torch.sigmoid(torch.cat(qv_ctx_scores, dim=1))
        qv_ctn_scores = torch.sigmoid(torch.cat(qv_ctn_scores, dim=1))
        vq_ctx_scores = torch.sigmoid(torch.cat(vq_ctx_scores, dim=1))
        vq_ctn_scores = torch.sigmoid(torch.cat(vq_ctn_scores, dim=1))
        final_scores = torch.sigmoid(torch.cat(qv_merge_scores, dim=1))

        loss_dict = self.loss(qv_ctx_scores, qv_ctn_scores, vq_ctx_scores, vq_ctn_scores, bbox_bias,
                               timestamps, overlaps, stage2_overlaps, starts, ends, anchor_masks)

        return loss_dict

    def forward_test(self,
                     query_feats=None,
                     query_masks=None,
                     video_feats=None,
                     start_ts=None,
                     end_ts=None,
                     scale_boundaries=None,
                     **kwargs):

        ori_ctx_feats = self.video_encoder(video_feats.permute(0, 2, 1))
        batch_size = self.cfg.TEST.BATCH_SIZE
        query_num = len(query_feats)
        num_batches = math.ceil(query_num / batch_size)

        merge_scores, merge_bboxes = list(), list()
        for bid in range(num_batches):
            sent_feat = query_feats[bid*int(batch_size):(bid+1)*int(batch_size)]
            qv_ctx_scores = self.q2v_stage1(ori_ctx_feats, sent_feat)
            if self.enable_stage2:
                hit_indices = list()
                starts = list()
                ends = list()
                filtered_ctx_feats = list()
                for i in range(self.nscales):
                    scale_first = scale_boundaries[0][i]
                    scale_last = scale_boundaries[0][i+1]

                    _, indices = torch.sort(qv_ctx_scores[i], dim=1, descending=True)
                    indices = indices[:, :self.stage2_topk]  #每一个层次都选择top-k个，开销也不小吧
                    hit_indices.append(indices)

                    filtered_ctx_feats.append(fetch_feats_by_index(ori_ctx_feats[i].repeat(indices.size(0), 1, 1), indices))
                    starts.append(fetch_feats_by_index(start_ts[bid*int(batch_size):(bid+1)*int(batch_size), scale_first:scale_last], indices))
                    ends.append(fetch_feats_by_index(end_ts[bid*int(batch_size):(bid+1)*int(batch_size), scale_first:scale_last], indices))
                
                starts = torch.cat(starts, dim=1)
                ends = torch.cat(ends, dim=1)

                qv_merge_scores, qv_ctn_scores, ctn_feats = self.q2v_stage2(
                    video_feats, sent_feat, hit_indices, qv_ctx_scores)
                ctx_feats = filtered_ctx_feats 
            else:
                ctx_feats = ori_ctx_feats
                ctn_feats = None
                qv_merge_scores = qv_ctx_scores
                starts = start_ts[bid*int(batch_size):(bid+1)*int(batch_size)]
                ends = end_ts[bid*int(batch_size):(bid+1)*int(batch_size)]
            
            bbox_bias = self.regressor(ctx_feats, ctn_feats, sent_feat)
            final_scores = torch.sigmoid(torch.cat(qv_merge_scores, dim=1))


            pred_scores, pred_bboxes = list(), list()

            final_scores = final_scores.cpu().numpy()
            starts = starts.cpu().numpy()
            ends = ends.cpu().numpy()
            bbox_bias = bbox_bias.cpu().numpy()
            
            rank_ids = np.argsort(final_scores, axis=1)
            rank_ids = rank_ids[:, ::-1]
            query_num = len(rank_ids)
            ori_start = starts[np.arange(query_num)[:, None], rank_ids]
            ori_end = ends[np.arange(query_num)[:, None], rank_ids]
            duration = ori_end - ori_start
            sebias = bbox_bias[np.arange(query_num)[:, None], rank_ids]
            sbias, ebias = sebias[:, :, 0], sebias[:, :, 1]
            pred_start = np.maximum(0, ori_start + sbias * duration)
            pred_end = ori_end + ebias * duration

            pred_scores = final_scores[np.arange(query_num)[:, None], rank_ids]
            pred_bboxes = np.stack([pred_start, pred_end], axis=2)
            if self.enable_nms:
                nms_res = list()
                for i in range(query_num):
                    bbox_nms = self.nms(pred_bboxes[i], thresh=0.3, topk=self.topk)
                    nms_res.append(bbox_nms)
                pred_bboxes = nms_res
            else:
                pred_scores = pred_scores[:, :self.topk].tolist()
                pred_bboxes = pred_bboxes[:, :self.topk, :].tolist()

            merge_scores.extend(pred_scores)
            merge_bboxes.extend(pred_bboxes)

        return merge_scores, merge_bboxes

    
    def loss(self, 
             qv_ctx_scores, 
             qv_ctn_scores, 
             vq_ctx_scores, 
             vq_ctn_scores, 
             bbox_bias,
             timestamps,
             overlaps, 
             stage2_overlaps, 
             starts, 
             ends,
             anchor_masks):
        qv_ctx_loss = self.rank_loss(overlaps, qv_ctx_scores, mask=anchor_masks)
        vq_overlaps, vq_ctx_scores = self.filter_anchor_by_iou(overlaps, vq_ctx_scores)
        vq_ctx_loss = self.rank_loss(vq_overlaps, vq_ctx_scores, mask=torch.ones_like(vq_ctx_scores))

        qv_ctn_loss, vq_ctn_loss, iou_loss = 0.0, 0.0, 0.0
        if self.cfg.MODEL.ENABLE_STAGE2:
            qv_ctn_loss = self.rank_loss(stage2_overlaps, qv_ctn_scores, mask=torch.ones_like(qv_ctn_scores))
            vq_overlaps_s2, vq_ctn_scores = self.filter_anchor_by_iou(stage2_overlaps, vq_ctn_scores)
            vq_ctn_loss = self.rank_loss(vq_overlaps_s2, vq_ctn_scores, mask=torch.ones_like(vq_ctn_scores))

        if self.cfg.LOSS.REGRESS.ENABLE:
            sbias = bbox_bias[:, :, 0]
            ebias = bbox_bias[:, :, 1]
            duration = ends - starts
            pred_start = starts + sbias * duration
            pred_end = ends + ebias * duration

            if self.cfg.MODEL.ENABLE_STAGE2:
                iou_mask = stage2_overlaps > self.cfg.LOSS.REGRESS.IOU_THRESH
            else:
                iou_mask = overlaps > self.cfg.LOSS.REGRESS.IOU_THRESH
            _, iou_loss = self.reg_loss(pred_start, pred_end, timestamps[:, 0:1], timestamps[:, 1:2], iou_mask)

        total_loss = self.cfg.LOSS.Q2V.CTX_WEIGHT * qv_ctx_loss + \
                     self.cfg.LOSS.Q2V.CTN_WEIGHT * qv_ctn_loss + \
                     self.cfg.LOSS.V2Q.CTX_WEIGHT * vq_ctx_loss + \
                     self.cfg.LOSS.V2Q.CTN_WEIGHT * vq_ctn_loss + \
                     self.cfg.LOSS.REGRESS.WEIGHT * iou_loss

        loss_dict = {
            "qv_ctx_loss": qv_ctx_loss,
            "qv_ctn_loss": qv_ctn_loss,
            "vq_ctx_loss": vq_ctx_loss,
            "vq_ctn_loss": vq_ctn_loss,
            "reg_loss": iou_loss,
            "total_loss": total_loss
        }
        return loss_dict


    def filter_anchor_by_iou(self, gt, pred):
        indicator = (torch.sum((gt > self.cfg.LOSS.V2Q.MIN_IOU).float(), dim=0, keepdim=False) > 0).long()
        moment_num = torch.sum(indicator)
        _, index = torch.sort(indicator, descending=True)
        index = index[:moment_num]
        gt = torch.index_select(gt, 1, index).transpose(0, 1)
        pred = torch.index_select(pred, 1, index).transpose(0, 1)
        return gt, pred


    def nms(self, pred, thresh=0.3, topk=5):
        nms_res = list()
        mask = [False] * len(pred)
        for i in range(len(pred)):
            f = pred[i].copy()
            if not mask[i]:
                nms_res.append(f)
                if len(nms_res) >= topk:
                    break
                for j in range(i, len(pred)):
                    tiou = compute_tiou(pred[i], pred[j])
                    if tiou > thresh:
                        mask[j] = True
        del mask
        return nms_res