# Copyright (c) Facebook, Inc. and its affiliates.
# 这段代码是用于构建目标建议生成器（proposal generator）的工具函数和注册机制。具体功能如下：
#
# 1. **PROPOSAL_GENERATOR_REGISTRY**：
#    - 使用 `Registry` 类创建了一个注册表 `PROPOSAL_GENERATOR_REGISTRY`，用于注册和管理不同类型的目标建议生成器。
#    - 注册表的文档字符串说明了注册器的作用，即从特征图中生成对象建议（object proposals）。
#
# 2. **build_proposal_generator 函数**：
#    - 这个函数根据配置文件中指定的 `cfg.MODEL.PROPOSAL_GENERATOR.NAME` 构建目标建议生成器。
#    - 如果配置为 `"PrecomputedProposals"`，则返回 `None`，表示不需要使用建议生成器。
#    - 否则，根据注册表 `PROPOSAL_GENERATOR_REGISTRY` 中的注册类型，调用相应的生成器构建函数，并传入配置和输入形状 `input_shape`。
#
# 总体来说，这段代码提供了一个灵活的机制，可以根据配置文件选择不同类型的目标建议生成器，适应不同的模型需求和任务设置。
from detectron2.utils.registry import Registry

PROPOSAL_GENERATOR_REGISTRY = Registry("PROPOSAL_GENERATOR")
PROPOSAL_GENERATOR_REGISTRY.__doc__ = """
Registry for proposal generator, which produces object proposals from feature maps.

The registered object will be called with `obj(cfg, input_shape)`.
The call should return a `nn.Module` object.
"""

from . import rpn, rrpn  # noqa F401 isort:skip


def build_proposal_generator(cfg, input_shape):
    """
    Build a proposal generator from `cfg.MODEL.PROPOSAL_GENERATOR.NAME`.
    The name can be "PrecomputedProposals" to use no proposal generator.
    """
    name = cfg.MODEL.PROPOSAL_GENERATOR.NAME # RPN
    if name == "PrecomputedProposals":
        return None

    return PROPOSAL_GENERATOR_REGISTRY.get(name)(cfg, input_shape)
