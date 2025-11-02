# Copyright (c) Facebook, Inc. and its affiliates.
# 这段代码是用于旋转目标检测中的旋转区域建议网络（RRPN）。主要实现了以下功能：
#
# 1. **find_top_rrpn_proposals函数**：
#    - 对于每个特征图，选择预测得分最高的前 pre_nms_topk 个建议框。
#    - 对这些建议框应用非极大值抑制（NMS），裁剪建议框，并移除尺寸较小的框。
#    - 如果 `training` 为 True，则返回所有特征图中得分最高的 post_nms_topk 个建议框；否则，返回每个特征图中得分最高的 post_nms_topk 个建议框。
#
# 2. **RRPN类**：
#    - 继承自RPN类，是旋转区域建议网络的实现。
#    - 包含了从配置文件中加载网络参数的方法 `from_config`。
#    - 实现了标记和采样锚点的方法 `label_and_sample_anchors`，用于为每个锚点标记正负类别和匹配的真实边界框。
#    - 实现了预测建议框的方法 `predict_proposals`，调用了 `find_top_rrpn_proposals` 函数来获取最终的建议框。
#
# 总体来说，这段代码用于生成旋转目标检测任务中的建议框，包括预测、筛选、NMS处理以及尺寸裁剪等操作。
import itertools
import logging
from typing import Dict, List
import torch

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, batched_nms_rotated, cat
from detectron2.structures import Instances, RotatedBoxes, pairwise_iou_rotated
from detectron2.utils.memory import retry_if_cuda_oom

from ..box_regression import Box2BoxTransformRotated
from .build import PROPOSAL_GENERATOR_REGISTRY
from .rpn import RPN

logger = logging.getLogger(__name__)


def find_top_rrpn_proposals(
    proposals,
    pred_objectness_logits,
    image_sizes,
    nms_thresh,
    pre_nms_topk,
    post_nms_topk,
    min_box_size,
    training,
):
    """
    对于每个特征图，选择 `pre_nms_topk` 个最高得分的提议，
    应用 NMS，裁剪提议，并移除小框。如果 `training` 为 True，则返回
    所有特征图中的 `post_nms_topk` 个最高得分提议，
    否则，返回每个特征图的最高 `post_nms_topk` 得分提议。

    参数:
        proposals (list[Tensor]): 一个包含 L 个张量的列表。张量 i 的形状为 (N, Hi*Wi*A, 5)。
            所有特征图上的提议预测。
        pred_objectness_logits (list[Tensor]): 一个包含 L 个张量的列表。张量 i 的形状为 (N, Hi*Wi*A)。
        image_sizes (list[tuple]): 每个图像的大小 (h, w)
        nms_thresh (float): 用于 NMS 的 IoU 阈值
        pre_nms_topk (int): 应用 NMS 之前要保留的前 k 个得分提议的数量。
            当 RRPN 在多个特征图上运行时（如 FPN），此数字适用于每个特征图。
        post_nms_topk (int): 应用 NMS 后要保留的前 k 个得分提议的数量。
            当 RRPN 在多个特征图上运行时（如 FPN），此数字是总数，
            所有特征图的总和。
        min_box_size(float): 提议框在像素中的最小边长（相对于输入图像的绝对单位）。
        training (bool): 如果提议用于训练，则为 True，否则为 False。
            此参数仅存在以支持遗留错误；请查找“NB: Legacy bug ...”
            注释。

    返回:
        proposals (list[Instances]): N 个 Instances 的列表。第 i 个 Instances
            存储图像 i 的 post_nms_topk 物体提议。
    """
    num_images = len(image_sizes)
    device = proposals[0].device

    # 1. 选择每个层级和每个图像的前 k 个锚框
    topk_scores = []  # #lvl 张量，每个形状为 N x topk
    topk_proposals = []
    level_ids = []  # #lvl 张量，每个形状为 (topk,)
    batch_idx = torch.arange(num_images, device=device)
    for level_id, proposals_i, logits_i in zip(
        itertools.count(), proposals, pred_objectness_logits
    ):
        Hi_Wi_A = logits_i.shape[1]
        num_proposals_i = min(pre_nms_topk, Hi_Wi_A)

        # 排序比 topk 更快 (https://github.com/pytorch/pytorch/issues/22812)
        logits_i, idx = logits_i.sort(descending=True, dim=1)
        topk_scores_i = logits_i[batch_idx, :num_proposals_i]
        topk_idx = idx[batch_idx, :num_proposals_i]

        # 每个都是 N x topk
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 5

        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device))

    # 2. 将所有层级连接在一起
    topk_scores = cat(topk_scores, dim=1)
    topk_proposals = cat(topk_proposals, dim=1)
    level_ids = cat(level_ids, dim=0)

    # 3. 对于每个图像，运行每层 NMS，并选择前 k 个结果。
    results = []
    for n, image_size in enumerate(image_sizes):
        boxes = RotatedBoxes(topk_proposals[n])
        scores_per_img = topk_scores[n]
        valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(scores_per_img)
        if not valid_mask.all():
            boxes = boxes[valid_mask]
            scores_per_img = scores_per_img[valid_mask]
        boxes.clip(image_size)

        # 过滤空框
        keep = boxes.nonempty(threshold=min_box_size)
        lvl = level_ids
        if keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = (boxes[keep], scores_per_img[keep], level_ids[keep])

        keep = batched_nms_rotated(boxes.tensor, scores_per_img, lvl, nms_thresh)
        # 在 Detectron1 中，训练和测试时的行为不同。
        # (https://github.com/facebookresearch/Detectron/issues/459)
        # 在训练期间，topk 是在训练批次中的 *所有* 提议上进行的。
        # 在测试期间，它是针对每个图像单独的提议。
        # 因此，训练行为变得依赖于批次，
        # 配置 "POST_NMS_TOPK_TRAIN" 最终依赖于批次大小。
        # 这个错误在 Detectron2 中得到解决，以使行为独立于批次大小。
        keep = keep[:post_nms_topk]

        res = Instances(image_size)
        res.proposal_boxes = boxes[keep]
        res.objectness_logits = scores_per_img[keep]
        results.append(res)
    return results


@PROPOSAL_GENERATOR_REGISTRY.register()
class RRPN(RPN):
    """
    RRPN（旋转区域提议网络）描述见 :paper:`RRPN`。
    """

    @configurable
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.anchor_boundary_thresh >= 0:
            raise NotImplementedError(
                "anchor_boundary_thresh 是未为 RRPN 实现的遗留选项。"
            )

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = super().from_config(cfg, input_shape)
        ret["box2box_transform"] = Box2BoxTransformRotated(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        return ret

    @torch.no_grad()
    def label_and_sample_anchors(self, anchors: List[RotatedBoxes], gt_instances: List[Instances]):
        """
        参数:
            anchors (list[RotatedBoxes]): 每个特征图的锚框。
            gt_instances: 每个图像的真实实例。

        返回:
            list[Tensor]:
                一个 #img 张量的列表。第 i 个元素是一个标签向量，其长度为
                所有特征图中锚框的总数。标签值在 {-1, 0, 1} 中，
                含义：-1 = 忽略；0 = 负类；1 = 正类。
            list[Tensor]:
                第 i 个元素是一个 Nx5 张量，其中 N 是所有特征图中锚框的总数。
                值是每个锚框匹配的 gt 框。
                对于未标记为 1 的锚框，值是未定义的。
        """
        anchors = RotatedBoxes.cat(anchors)

        gt_boxes = [x.gt_boxes for x in gt_instances]
        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        for gt_boxes_i in gt_boxes:
            """
            gt_boxes_i: 第 i 张图像的真实框
            """
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou_rotated)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
            # 匹配是内存消耗大的，可能导致 CPU 张量。但结果很小
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)

            # 每个锚框的标签向量 (-1, 0, 1)
            gt_labels_i = self._subsample_labels(gt_labels_i)

            if len(gt_boxes_i) == 0:
                # 这些值不会被使用，因为锚框标记为背景
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor

            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)
        return gt_labels, matched_gt_boxes

    @torch.no_grad()
    def predict_proposals(self, anchors, pred_objectness_logits, pred_anchor_deltas, image_sizes):
        pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
        return find_top_rrpn_proposals(
            pred_proposals,
            pred_objectness_logits,
            image_sizes,
            self.nms_thresh,
            self.pre_nms_topk[self.training],
            self.post_nms_topk[self.training],
            self.min_box_size,
            self.training,
        )
