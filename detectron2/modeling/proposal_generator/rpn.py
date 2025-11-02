# Copyright (c) Facebook, Inc. and its affiliates.
# 这段代码是用于构建和训练区域建议网络（RPN）的核心部分，它是Faster R-CNN中的一个重要组成部分。让我来解释一下主要的功能和组成部分：
#
# 1. **Registry 和 build 函数**:
#    - `RPN_HEAD_REGISTRY` 是一个注册表，用于存储和管理不同类型的RPN头部模型。
#    - `build_rpn_head(cfg, input_shape)` 函数根据配置文件和输入形状构建特定类型的RPN头部模型。
#
# 2. **RPN 头部模型** (`StandardRPNHead` 类):
#    - 这个类定义了标准的RPN头部模型，实现了对象性分类和边界框回归。它通过卷积层生成共享的隐藏状态，然后分别用1x1卷积预测每个锚点的对象性logits和边界框回归参数。
#    - 使用ReLU作为激活函数的3x3卷积层生成隐藏表示，然后用1x1卷积层分别预测对象性logits和边界框回归参数。
#
# 3. **RPN 类** (`RPN` 类):
#    - `RPN` 类定义了区域建议网络，负责生成候选区域建议。
#    - 使用 `label_and_sample_anchors` 函数进行锚点标记和采样，生成正负样本标签和对应的真实边界框。
#    - 使用 `losses` 函数计算RPN预测与真实标签之间的损失，包括对象性分类损失和边界框回归损失。
#
# 4. **配置和参数**:
#    - 通过配置文件 (`cfg`) 和输入形状 (`input_shape`) 初始化RPN头部模型和RPN类。
#    - 使用注册表管理不同类型的RPN头部模型，并根据配置选择特定的头部模型。
#
# 这些组件共同工作，构成了一个完整的区域建议网络，用于在目标检测任务中生成候选框。这种结构和逻辑是目标检测模型中常见的，通过卷积神经网络和损失函数来实现区域提议的生成和优化。
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.registry import Registry

from ..anchor_generator import build_anchor_generator
from ..box_regression import Box2BoxTransform, _dense_box_regression_loss
from ..matcher import Matcher
from ..sampling import subsample_labels
from .build import PROPOSAL_GENERATOR_REGISTRY
from .proposal_utils import find_top_rpn_proposals

RPN_HEAD_REGISTRY = Registry("RPN_HEAD")
RPN_HEAD_REGISTRY.__doc__ = """
Registry for RPN heads, which take feature maps and perform
objectness classification and bounding box regression for anchors.

The registered object will be called with `obj(cfg, input_shape)`.
The call should return a `nn.Module` object.
"""


"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    A: number of cell anchors (must be the same for all feature maps)
    Hi, Wi: height and width of the i-th feature map
    B: size of the box parameterization

Naming convention:

    objectness: refers to the binary classification of an anchor as object vs. not object.

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`), or 5d for rotated boxes.

    pred_objectness_logits: predicted objectness scores in [-inf, +inf]; use
        sigmoid(pred_objectness_logits) to estimate P(object).

    gt_labels: ground-truth binary classification labels for objectness

    pred_anchor_deltas: predicted box2box transform deltas

    gt_anchor_deltas: ground-truth box2box transform deltas
"""


def build_rpn_head(cfg, input_shape):
    """
    Build an RPN head defined by `cfg.MODEL.RPN.HEAD_NAME`.
    """
    name = cfg.MODEL.RPN.HEAD_NAME
    return RPN_HEAD_REGISTRY.get(name)(cfg, input_shape)


@RPN_HEAD_REGISTRY.register()
class StandardRPNHead(nn.Module):
    """
    标准的 RPN 分类和回归头，参考论文 :paper:`Faster R-CNN`。
    使用 3x3 卷积产生共享的隐藏状态，从中一个 1x1 卷积预测
    每个锚点的目标概率 logits，另一个 1x1 卷积预测边界框增量
    指定如何将每个锚点变形为一个物体提议。
    """

    @configurable
    def __init__(
        self, *, in_channels: int, num_anchors: int, box_dim: int = 4, conv_dims: List[int] = (-1,)
    ):
        """
        注意：该接口为实验性接口。

        参数：
            in_channels (int): 输入特征通道的数量。当使用多个输入特征时，
                它们必须具有相同的通道数。
            num_anchors (int): 每个空间位置上要预测的锚点数量。
                每个特征图的总锚点数量为 `num_anchors * H * W`。
            box_dim (int): 边界框的维度，也即每个锚点的回归预测数量。
                一个轴对齐的框有 box_dim=4，而一个旋转的框有 box_dim=5。
            conv_dims (list[int]): 整数列表，表示 N 个卷积层的输出通道。
                设置为 -1 将使用与输入通道相同的输出通道数。
        """
        super().__init__()
        cur_channels = in_channels
        # 为了向后兼容，保持旧变量名称和结构。
        # 否则旧的检查点将无法加载。
        if len(conv_dims) == 1:
            out_channels = cur_channels if conv_dims[0] == -1 else conv_dims[0]
            # 用于隐藏表示的 3x3 卷积
            self.conv = self._get_rpn_conv(cur_channels, out_channels)
            cur_channels = out_channels
        else:
            self.conv = nn.Sequential()
            for k, conv_dim in enumerate(conv_dims):
                out_channels = cur_channels if conv_dim == -1 else conv_dim
                if out_channels <= 0:
                    raise ValueError(
                        f"卷积输出通道数应大于 0. 得到 {out_channels}"
                    )
                conv = self._get_rpn_conv(cur_channels, out_channels)
                self.conv.add_module(f"conv{k}", conv)
                cur_channels = out_channels
        # 用于预测目标概率 logits 的 1x1 卷积
        self.objectness_logits = nn.Conv2d(cur_channels, num_anchors, kernel_size=1, stride=1)
        # 用于预测框到框变换增量的 1x1 卷积
        self.anchor_deltas = nn.Conv2d(cur_channels, num_anchors * box_dim, kernel_size=1, stride=1)

        # 保持权重初始化顺序以向后兼容。
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def _get_rpn_conv(self, in_channels, out_channels):
        return Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.ReLU(),
        )

    @classmethod
    def from_config(cls, cfg, input_shape):
        # 标准 RPN 在各层之间共享：
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "每一层的通道数必须相同！"
        in_channels = in_channels[0]

        # RPNHead 应该与锚生成器的输入相同
        # 注意：假设创建锚生成器不会有不必要的副作用。
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_anchors = anchor_generator.num_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_anchors)) == 1
        ), "每个空间位置的锚点数量必须相同"
        return {
            "in_channels": in_channels,
            "num_anchors": num_anchors[0],
            "box_dim": box_dim,
            "conv_dims": cfg.MODEL.RPN.CONV_DIMS,
        }

    def forward(self, features: List[torch.Tensor]):
        """
        参数：
            features (list[Tensor]): 特征图列表

        返回：
            list[Tensor]: 一个包含 L 个元素的列表。
                第 i 个元素是形状为 (N, A, Hi, Wi) 的张量，
                代表所有锚点的预测目标概率 logits。A 是锚点数量。
            list[Tensor]: 一个包含 L 个元素的列表。第 i 个元素是形状
                为 (N, A*box_dim, Hi, Wi) 的张量，代表
                用于将锚点转换为提议的预测 "增量"。
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = self.conv(x)  # 通过卷积层处理特征图
            pred_objectness_logits.append(self.objectness_logits(t))  # 预测目标概率 logits
            pred_anchor_deltas.append(self.anchor_deltas(t))  # 预测框增量
        return pred_objectness_logits, pred_anchor_deltas  # 返回预测结果


@PROPOSAL_GENERATOR_REGISTRY.register()
class RPN(nn.Module):
    """
    区域提议网络，参考论文 :paper:`Faster R-CNN`。
    """

    @configurable
    def __init__(
            self,
            *,
            in_features: List[str],  # 输入特征名称列表
            head: nn.Module,  # 预测每个层的 logits 和回归增量的模块
            anchor_generator: nn.Module,  # 从特征列表创建锚点的模块，通常是 :class:`AnchorGenerator` 的实例
            anchor_matcher: Matcher,  # 通过与真实标签匹配为锚点标注
            box2box_transform: Box2BoxTransform,  # 定义锚点框到实例框的转换
            batch_size_per_image: int,  # 每张图像用于训练的锚点数量
            positive_fraction: float,  # 用于训练的前景锚点的比例
            pre_nms_topk: Tuple[float, float],  # 在 NMS 之前选择的前 k 个提议数量（训练，测试）
            post_nms_topk: Tuple[float, float],  # 在 NMS 之后选择的前 k 个提议数量（训练，测试）
            nms_thresh: float = 0.7,  # 用于去重预测提议的 NMS 阈值
            min_box_size: float = 0.0,  # 移除任何边长小于此阈值的提议框（以输入图像像素为单位）
            anchor_boundary_thresh: float = -1.0,  # 旧版选项
            loss_weight: Union[float, Dict[str, float]] = 1.0,  # 用于损失的权重，可以是单个 float 或字典
            box_reg_loss_type: str = "smooth_l1",  # 使用的损失类型，支持 "smooth_l1", "giou"
            smooth_l1_beta: float = 0.0,  # 平滑 L1 回归损失的 beta 参数，默认使用 L1 损失
    ):
        """
        注意：该接口为实验性接口。
        """
        super().__init__()
        self.in_features = in_features  # 保存输入特征
        self.rpn_head = head  # 保存 RPN 头部
        self.anchor_generator = anchor_generator  # 保存锚点生成器
        self.anchor_matcher = anchor_matcher  # 保存锚点匹配器
        self.box2box_transform = box2box_transform  # 保存框转换对象
        self.batch_size_per_image = batch_size_per_image  # 保存每张图像的批处理大小
        self.positive_fraction = positive_fraction  # 保存正样本的比例

        # 根据训练状态映射到训练/测试设置
        self.pre_nms_topk = {True: pre_nms_topk[0], False: pre_nms_topk[1]}
        self.post_nms_topk = {True: post_nms_topk[0], False: post_nms_topk[1]}

        self.nms_thresh = nms_thresh  # 保存 NMS 阈值
        self.min_box_size = float(min_box_size)  # 保存最小框大小
        self.anchor_boundary_thresh = anchor_boundary_thresh  # 保存锚点边界阈值

        # 如果损失权重是单个浮点数，则将其转换为字典
        if isinstance(loss_weight, float):
            loss_weight = {"loss_rpn_cls": loss_weight, "loss_rpn_loc": loss_weight}
        self.loss_weight = loss_weight  # 保存损失权重

        self.box_reg_loss_type = box_reg_loss_type  # 保存框回归损失类型
        self.smooth_l1_beta = smooth_l1_beta  # 保存平滑 L1 beta 参数

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # 从配置构建 RPN
        in_features = cfg.MODEL.RPN.IN_FEATURES  # 获取输入特征
        ret = {
            "in_features": in_features,
            "min_box_size": cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE,
            "nms_thresh": cfg.MODEL.RPN.NMS_THRESH,
            "batch_size_per_image": cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.RPN.POSITIVE_FRACTION,
            "loss_weight": {
                "loss_rpn_cls": cfg.MODEL.RPN.LOSS_WEIGHT,
                "loss_rpn_loc": cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT * cfg.MODEL.RPN.LOSS_WEIGHT,
            },
            "anchor_boundary_thresh": cfg.MODEL.RPN.BOUNDARY_THRESH,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS),
            "box_reg_loss_type": cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE,
            "smooth_l1_beta": cfg.MODEL.RPN.SMOOTH_L1_BETA,
        }

        # 获取 NMS 前后的 top k 设置
        ret["pre_nms_topk"] = (cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN, cfg.MODEL.RPN.PRE_NMS_TOPK_TEST)
        ret["post_nms_topk"] = (cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.RPN.POST_NMS_TOPK_TEST)

        # 构建锚点生成器
        ret["anchor_generator"] = build_anchor_generator(cfg, [input_shape[f] for f in in_features])
        # 构建锚点匹配器
        ret["anchor_matcher"] = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        # 构建 RPN 头部
        ret["head"] = build_rpn_head(cfg, [input_shape[f] for f in in_features])
        return ret

    def _subsample_labels(self, label):
        """
        随机抽样正负例的子集，并将未被选中的标签向量元素覆盖为忽略值（-1）。

        参数:
            labels (Tensor): 一个包含 -1, 0, 1 的向量。将被就地修改并返回。
        """
        pos_idx, neg_idx = subsample_labels(
            label, self.batch_size_per_image, self.positive_fraction, 0
        )
        # ! 用忽略标签 (-1) 填充，然后设置正负标签
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    @torch.jit.unused
    @torch.no_grad()
    def label_and_sample_anchors(
            self, anchors: List[Boxes], gt_instances: List[Instances]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        参数:
            anchors (list[Boxes]): 每个特征图的锚框。
            gt_instances: 每个图像的真实实例。

        返回:
            list[Tensor]:
                包含每个图像的标签的张量列表。第 i 个元素是一个标签向量，其长度为
                所有特征图上锚框的总数 R = sum(Hi * Wi * A)。
                标签值为 {-1, 0, 1}，含义: -1 = 忽略; 0 = 负类; 1 = 正类。
            list[Tensor]:
                第 i 个元素是一个 Rx4 的张量，包含与每个锚框匹配的真实框。未标记为 1 的锚框值未定义。
        """
        anchors = Boxes.cat(anchors)

        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        for image_size_i, gt_boxes_i in zip(image_sizes, gt_boxes):
            """
            image_size_i: 第 i 张图像的尺寸 (h, w)
            gt_boxes_i: 第 i 张图像的真实框
            """

            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
            # 匹配过程消耗内存，结果可能是 CPU 张量。结果较小
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            if self.anchor_boundary_thresh >= 0:
                # 丢弃超出图像边界的锚框
                # 注意: 这是遗留功能，默认在 Detectron2 中关闭
                anchors_inside_image = anchors.inside_box(image_size_i, self.anchor_boundary_thresh)
                gt_labels_i[~anchors_inside_image] = -1

            # 为每个锚框生成标签向量 (-1, 0, 1)
            gt_labels_i = self._subsample_labels(gt_labels_i)

            if len(gt_boxes_i) == 0:
                # 这些值不会被使用，因为锚框标记为背景
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                # TODO 对于被忽略的框进行浪费的索引计算
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor

            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)
        return gt_labels, matched_gt_boxes

    @torch.jit.unused
    def losses(
            self,
            anchors: List[Boxes],
            pred_objectness_logits: List[torch.Tensor],
            gt_labels: List[torch.Tensor],
            pred_anchor_deltas: List[torch.Tensor],
            gt_boxes: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        返回一组 RPN 预测及其相关真实值的损失。

        参数:
            anchors (list[Boxes or RotatedBoxes]): 每个特征图的锚框，每个形状为 (Hi*Wi*A, B)，其中 B 是框的维度 (4 或 5)。
            pred_objectness_logits (list[Tensor]): 长度为 L 的张量列表。
                第 i 个元素是形状为 (N, Hi*Wi*A) 的张量，表示所有锚框的预测物体性 logits。
            gt_labels (list[Tensor]): :meth:`label_and_sample_anchors` 的输出。
            pred_anchor_deltas (list[Tensor]): 长度为 L 的张量列表。第 i 个元素是形状为
                (N, Hi*Wi*A, 4 或 5) 的张量，表示用于将锚框转换为提议的预测“增量”。
            gt_boxes (list[Tensor]): :meth:`label_and_sample_anchors` 的输出。

        返回:
            dict[loss name -> loss value]: 从损失名称映射到损失值的字典。
                损失名称包括: `loss_rpn_cls` 表示物体性分类损失和
                `loss_rpn_loc` 表示提议定位损失。
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))

        # 记录每张图像中用于训练的正负锚框数量
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)

        localization_loss = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
        )

        valid_mask = gt_labels >= 0
        objectness_loss = F.binary_cross_entropy_with_logits(
            cat(pred_objectness_logits, dim=1)[valid_mask],
            gt_labels[valid_mask].to(torch.float32),
            reduction="sum",
        )
        normalizer = self.batch_size_per_image * num_images
        losses = {
            "loss_rpn_cls": objectness_loss / normalizer,
            # 原始的 Faster R-CNN 论文使用略有不同的归一化方法
            # 但在实践中没有影响
            "loss_rpn_loc": localization_loss / normalizer,
        }
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            gt_instances: Optional[List[Instances]] = None,
    ):
        """
        参数:
            images (ImageList): 输入图像，长度为 `N`
            features (dict[str, Tensor]): 输入数据，映射从特征图名称到张量。轴 0 表示输入数据中的图像数量 `N`;
                轴 1-3 分别为通道、高度和宽度，这可能在特征图之间有所不同 (例如，如果使用特征金字塔)。
            gt_instances (list[Instances], optional): 长度为 `N` 的 `Instances` 列表。
                每个 `Instances` 存储对应图像的真实实例。

        返回:
            proposals: list[Instances]: 包含字段 "proposal_boxes" 和 "objectness_logits"
            loss: dict[Tensor] 或 None
        """
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)  # [63000, 4] 的列表

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)  # [1, 1024, 55, 50] 的列表
        # 转置 Hi*Wi*A 维度到中间：
        pred_objectness_logits = [  # [1, 41250] 的列表
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]  # [1, 60, 55, 50] -> [1, 15, 4, 55, 50]

        if self.training:
            assert gt_instances is not None, "RPN requires gt_instances in training!"
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
            losses = self.losses(
                anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
            )
        else:
            losses = {}
        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )
        return proposals, losses # list of proposals, for each image

    def predict_proposals(
            self,
            anchors: List[Boxes],
            pred_objectness_logits: List[torch.Tensor],
            pred_anchor_deltas: List[torch.Tensor],
            image_sizes: List[Tuple[int, int]],
    ):
        """
        解码所有预测的框回归增量为提议。通过应用 NMS 找到顶级提议
        并移除过小的框。

        返回:
            proposals (list[Instances]): N 个 Instances 的列表。第 i 个 Instances
                存储图像 i 的 post_nms_topk 物体提议，按其
                物体性分数降序排列。
        """
        # 这些提议被视为与 roi heads 联合训练的固定提议。
        # 这种方法忽略了与提议框坐标相关的导数，这些坐标也是网络响应的一部分。
        with torch.no_grad():
            pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
            return find_top_rpn_proposals(
                pred_proposals,
                pred_objectness_logits,
                image_sizes,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_size,
                self.training,
            )

    def _decode_proposals(self, anchors: List[Boxes], pred_anchor_deltas: List[torch.Tensor]):
        """
        通过应用预测的锚框增量将锚框转换为提议。

        返回:
            proposals (list[Tensor]): 一个包含 L 个张量的列表。张量 i 的形状为
                (N, Hi*Wi*A, B)
        """
        N = pred_anchor_deltas[0].shape[0]
        proposals = []
        # 对于每个特征图
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            B = anchors_i.tensor.size(1)
            pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)
            # 扩展锚框至形状 (N*Hi*Wi*A, B)
            anchors_i = anchors_i.tensor.unsqueeze(0).expand(N, -1, -1).reshape(-1, B)
            proposals_i = self.box2box_transform.apply_deltas(pred_anchor_deltas_i, anchors_i)
            # 将特征图提议添加至形状 (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals
