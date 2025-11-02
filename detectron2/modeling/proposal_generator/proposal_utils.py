# Copyright (c) Facebook, Inc. and its affiliates.
# 这段代码是用于目标检测中的区域提议网络（RPN）处理的一部分，具体来说，是在选择、筛选和处理检测模型生成的候选区域提议（proposals）。主要功能包括：
#
# 1. **find_top_rpn_proposals函数**：
#    - 为每个特征图选择预测得分最高的前pre_nms_topk个提议。
#    - 应用非极大值抑制（NMS）来消除重叠的提议。
#    - 裁剪提议框，移除尺寸过小的框。
#    - 返回每张图像中得分最高的post_nms_topk个提议。
#
# 2. **add_ground_truth_to_proposals函数** 和 **add_ground_truth_to_proposals_single_image函数**：
#    - 用于将地面真实框（ground truth）添加到候选提议中，以增强模型训练时的稳定性和性能。
#    - 将地面真实框转换为Instances对象，并为其分配与目标相关的logits值。
#    - 最终，将地面真实框与候选提议合并，形成新的Instances对象，用于后续的训练或评估。
#
# 整体来看，这些函数是在处理和优化目标检测模型的候选区域提议过程中的关键步骤，涵盖了提议选择、NMS处理、框裁剪、以及地面真实框的集成等功能。
import logging
import math
from typing import List, Tuple, Union
import torch

from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances
from detectron2.utils.env import TORCH_VERSION

logger = logging.getLogger(__name__)


def _is_tracing():
    if torch.jit.is_scripting():
        # https://github.com/pytorch/pytorch/issues/47379
        return False
    else:
        return TORCH_VERSION >= (1, 7) and torch.jit.is_tracing()


def find_top_rpn_proposals(
    proposals: List[torch.Tensor],
    pred_objectness_logits: List[torch.Tensor],
    image_sizes: List[Tuple[int, int]],
    nms_thresh: float,
    pre_nms_topk: int,
    post_nms_topk: int,
    min_box_size: float,
    training: bool,
):
    """
    对每个特征图，选择 `pre_nms_topk` 个得分最高的建议框，
    应用 NMS，裁剪建议框，移除小框。返回每幅图像中所有特征图的
    `post_nms_topk` 个得分最高的建议框。

    参数：
        proposals (list[Tensor]): 包含 L 个张量的列表。张量 i 的形状为 (N, Hi*Wi*A, 4)。
            所有特征图上的建议框预测。
        pred_objectness_logits (list[Tensor]): 包含 L 个张量的列表。张量 i 的形状为 (N, Hi*Wi*A)。
        image_sizes (list[tuple]): 每幅图像的尺寸 (h, w)
        nms_thresh (float): 用于 NMS 的 IoU 阈值
        pre_nms_topk (int): 应用 NMS 前要保留的前 k 个得分最高的建议框数量。
            当 RPN 在多个特征图上运行时（如在 FPN 中），这个数字是针对每个
            特征图的。
        post_nms_topk (int): 应用 NMS 后要保留的前 k 个得分最高的建议框数量。
            当 RPN 在多个特征图上运行时（如在 FPN 中），这个数字是总数，
            涉及所有特征图。
        min_box_size (float): 最小建议框边长（以像素为单位，绝对值
            相对于输入图像）。
        training (bool): 如果建议框用于训练则为 True，否则为 False。
            这个参数仅用于支持一个遗留错误；查找 "NB: Legacy bug ..."
            注释。

    返回：
        list[Instances]: 包含 N 个 Instances 的列表。第 i 个 Instances
            存储图像 i 的 post_nms_topk 个目标建议框，按其
            目标得分降序排序。
    """
    num_images = len(image_sizes)  # 图像数量
    device = proposals[0].device  # 获取设备信息

    # 1. 为每个级别和每幅图像选择前 k 个锚框
    topk_scores = []  # #lvl 张量，每个形状为 N x topk
    topk_proposals = []
    level_ids = []  # #lvl 张量，每个形状为 (topk,)
    batch_idx = torch.arange(num_images, device=device)  # 批量索引
    for level_id, (proposals_i, logits_i) in enumerate(zip(proposals, pred_objectness_logits)):
        Hi_Wi_A = logits_i.shape[1]  # 获取当前级别的锚框数量
        if isinstance(Hi_Wi_A, torch.Tensor):  # 如果是张量，则使用张量中的值
            num_proposals_i = torch.clamp(Hi_Wi_A, max=pre_nms_topk)  # 限制最大值
        else:
            num_proposals_i = min(Hi_Wi_A, pre_nms_topk)  # 取当前和预设的最小值

        # 排序比 topk 更快: https://github.com/pytorch/pytorch/issues/22812
        logits_i, idx = logits_i.sort(descending=True, dim=1)  # 排序，获取索引
        topk_scores_i = logits_i.narrow(1, 0, num_proposals_i)  # 得分最高的前 k 个
        topk_idx = idx.narrow(1, 0, num_proposals_i)  # 索引前 k 个

        # 每个是 N x topk
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 4

        topk_proposals.append(topk_proposals_i)  # 添加到列表中
        topk_scores.append(topk_scores_i)
        level_ids.append(torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device))

    # 2. 将所有级别连接在一起
    topk_scores = cat(topk_scores, dim=1)  # 连接得分
    topk_proposals = cat(topk_proposals, dim=1)  # 连接建议框
    level_ids = cat(level_ids, dim=0)  # 连接级别索引

    # 3. 对每幅图像执行每级的 NMS，并选择前 k 个结果。
    results: List[Instances] = []
    for n, image_size in enumerate(image_sizes):
        boxes = Boxes(topk_proposals[n])  # 创建 Boxes 对象
        scores_per_img = topk_scores[n]  # 获取当前图像的得分
        lvl = level_ids  # 获取级别索引

        valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(scores_per_img)  # 检查有效性
        if not valid_mask.all():  # 如果存在无效值
            if training:
                raise FloatingPointError(
                    "预测的框或得分包含 Inf/NaN. 训练已经发散."
                )
            boxes = boxes[valid_mask]  # 仅保留有效框
            scores_per_img = scores_per_img[valid_mask]  # 仅保留有效得分
            lvl = lvl[valid_mask]  # 仅保留有效级别
        boxes.clip(image_size)  # 裁剪框到图像尺寸

        # 过滤掉空框
        keep = boxes.nonempty(threshold=min_box_size)  # 保留非空框
        if _is_tracing() or keep.sum().item() != len(boxes):  # 如果正在跟踪或保留框数与总数不一致
            boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], lvl[keep]  # 更新框、得分和级别

        keep = batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)  # 执行批量 NMS
        # 在 Detectron1 中，训练和测试期间的行为不同。
        # （https://github.com/facebookresearch/Detectron/issues/459）
        # 在训练期间，topk 是基于*所有*图像中的提议。
        # 在测试期间，它是针对每幅图像的提议。
        # 因此，训练行为变得依赖于批量，
        # 配置 "POST_NMS_TOPK_TRAIN" 最终依赖于批量大小。
        # 此错误在 Detectron2 中得到解决，使行为独立于批量大小。
        keep = keep[:post_nms_topk]  # 保留已排序的框

        res = Instances(image_size)  # 创建 Instances 对象
        res.proposal_boxes = boxes[keep]  # 设置建议框
        res.objectness_logits = scores_per_img[keep]  # 设置得分
        results.append(res)  # 添加到结果列表
    return results  # 返回最终的建议框结果



def add_ground_truth_to_proposals(
    gt: Union[List[Instances], List[Boxes]], proposals: List[Instances]
) -> List[Instances]:
    """
    对所有图像调用 `add_ground_truth_to_proposals_single_image`。

    参数：
        gt (Union[List[Instances], List[Boxes]]): 包含 N 个元素的列表。元素 i 是一个 Instances
            对象，表示图像 i 的真实标签。
        proposals (list[Instances]): 包含 N 个元素的列表。元素 i 是一个 Instances
            对象，表示图像 i 的建议框。

    返回：
        list[Instances]: 包含 N 个 Instances 的列表。每个实例是图像的建议框，
            包含字段 "proposal_boxes" 和 "objectness_logits"。
    """
    assert gt is not None  # 确保真实标签不为空

    # 检查建议框和真实标签的数量是否一致
    if len(proposals) != len(gt):
        raise ValueError("proposals 和 gt 应该与图像数量相同！")
    if len(proposals) == 0:  # 如果没有建议框，直接返回
        return proposals

    # 对每幅图像调用单独的处理函数，将真实标签添加到建议框中
    return [
        add_ground_truth_to_proposals_single_image(gt_i, proposals_i)
        for gt_i, proposals_i in zip(gt, proposals)  # 按索引配对真实标签和建议框
    ]



def add_ground_truth_to_proposals_single_image(
    gt: Union[Instances, Boxes], proposals: Instances
) -> Instances:
    """
    将 `gt` 信息添加到 `proposals` 中。

    参数：
        与 `add_ground_truth_to_proposals` 相同，但针对每幅图像的 gt 和 proposals。

    返回：
        与 `add_ground_truth_to_proposals` 相同，但仅针对一幅图像。
    """
    if isinstance(gt, Boxes):
        # 将 Boxes 转换为 Instances
        gt = Instances(proposals.image_size, gt_boxes=gt)

    gt_boxes = gt.gt_boxes  # 获取真实框
    device = proposals.objectness_logits.device  # 获取设备信息
    # 为所有真实框分配对应的目标概率的 logit 值
    gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
    gt_logits = gt_logit_value * torch.ones(len(gt_boxes), device=device)  # 创建 logit 向量

    # 连接 gt_boxes 和 proposals，要求它们具有相同的字段
    gt_proposal = Instances(proposals.image_size, **gt.get_fields())  # 创建新的 Instances 对象
    gt_proposal.proposal_boxes = gt_boxes  # 设置真实框
    gt_proposal.objectness_logits = gt_logits  # 设置 logit 值

    # 检查 proposals 中的每个字段在 gt_proposal 中是否存在
    for key in proposals.get_fields().keys():
        assert gt_proposal.has(
            key
        ), "属性 '{}' 在 `proposals` 中不存在于 `gt` 中".format(key)

    # 注意：Instances.cat 仅使用第一个项的字段。后续项中的额外字段将被丢弃。
    new_proposals = Instances.cat([proposals, gt_proposal])  # 合并 proposals 和 gt_proposal

    return new_proposals  # 返回新的提议
