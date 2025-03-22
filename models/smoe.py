import torch

from typing import Tuple
from models.smoe_config import SpatialMoEConfig
from models.smoe_routing import SMoERouting, SMoERouting_soe


# class SpatialMoE2d(torch.nn.Module):
#     """
#     Spatial mixture-of-experts layer.

#     This implements the actual MoE. The gating function must be applied
#     separately (to facilitate sharing).

#     """

#     def __init__(self, smoe_config: SpatialMoEConfig) -> None:
#         super().__init__()
#         self.smoe_config = smoe_config
#         if smoe_config.out_planes > smoe_config.num_experts:
#             raise ValueError(f'SpatialMoE: out planes {smoe_config.out_planes}'
#                              f' > num experts {smoe_config.num_experts}')
#         if smoe_config.expert_block is None:
#             smoe_config.expert_block = torch.nn.Conv2d
#         # self.experts = smoe_config.expert_block(
#         #     smoe_config.in_planes, smoe_config.num_experts,
#         #     kernel_size=smoe_config.kernel_size, padding=smoe_config.padding,
#         #     bias=False)

#     def extra_repr(self) -> str:
#         return (f'out_planes={self.smoe_config.out_planes}'
#                 f' num_experts={self.smoe_config.num_experts}')

#     def forward(self, x: torch.Tensor, routing_weights: torch.Tensor, experts) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:     
#         selected_experts, routing_map, routing_indices = SMoERouting.apply(
#             experts, routing_weights, self.smoe_config, self)
#         return selected_experts, routing_map, routing_indices


class GatedSpatialMoE2d(torch.nn.Module):
    """Spatial MoE with internal gating function."""

    def __init__(self, smoe_config: SpatialMoEConfig) -> None:
        super().__init__()
        self.smoe_config = smoe_config
        # self.smoe = SpatialMoE2d(smoe_config)
        self.gate = smoe_config.gate_block(smoe_config)

    def extra_repr(self) -> str:
        return repr(self.smoe_config)

    def forward(self, x: torch.Tensor, experts) -> torch.Tensor:
        self.routing_weights = self.gate(x)
        self.routing_weights = self.routing_weights.unsqueeze(-1).repeat(1, 1, 1, 1, experts.size(-1))
        selected, self.routing_map, self.routing_indices = SMoERouting.apply(
            experts, self.routing_weights, self.smoe_config, self)       
        # x, self.smoe.routing_map, self.smoe.routing_indices = self.smoe(
        #     x, self.smoe.routing_weights, experts)
        selected = selected.sum(2)
        return selected
    
class GatedSpatialMoE2d_s(torch.nn.Module):
    """Spatial MoE with internal gating function."""

    def __init__(self, smoe_config: SpatialMoEConfig) -> None:
        super().__init__()
        self.smoe_config = smoe_config
        # self.smoe = SpatialMoE2d(smoe_config)
        self.gate = smoe_config.gate_block(smoe_config)

    def extra_repr(self) -> str:
        return repr(self.smoe_config)

    def forward(self, x: torch.Tensor, experts) -> torch.Tensor:
        # self.routing_weights = self.gate(x)
        self.routing_weights = self.gate(experts)
        # self.routing_weights = self.routing_weights.unsqueeze(-1).repeat(1, 1, 1, 1, experts.size(-1))
        selected, self.routing_map, self.routing_indices = SMoERouting.apply(
            experts, self.routing_weights, self.smoe_config, self)       
        # x, self.smoe.routing_map, self.smoe.routing_indices = self.smoe(
        #     x, self.smoe.routing_weights, experts)
        # selected = selected.sum(2)
        return selected
    
# class GatedSpatialMoE2d_soe(torch.nn.Module):
#     """Spatial MoE with internal gating function."""

#     def __init__(self, smoe_config: SpatialMoEConfig) -> None:
#         super().__init__()
#         self.smoe_config = smoe_config
#         # self.smoe = SpatialMoE2d(smoe_config)
#         self.gate = smoe_config.gate_block(smoe_config)

#     def extra_repr(self) -> str:
#         return repr(self.smoe_config)

#     def forward(self, x: torch.Tensor, experts) -> torch.Tensor:
#         self.routing_weights = self.gate(x)
#         self.routing_weights = self.routing_weights.unsqueeze(-1).repeat(1, 1, 1, experts.size(-1))
#         selected, self.routing_map, self.routing_indices = SMoERouting_soe.apply(
#             experts, self.routing_weights, self.smoe_config, self)       
#         # x, self.smoe.routing_map, self.smoe.routing_indices = self.smoe(
#         #     x, self.smoe.routing_weights, experts)
#         selected = selected.sum(1)
#         return selected