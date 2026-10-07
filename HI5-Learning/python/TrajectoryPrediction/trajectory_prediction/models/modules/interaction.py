# Copyright 2024, Theodor Westny. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Optional, Any
import torch
from torch import nn


class InteractionNet(nn.Module):
    def __init__(self,
                 num_hidden: int,
                 interaction_aware: bool = True,
                 num_heads: int = 4) -> None:
        super().__init__()

        if interaction_aware:
            self.interaction = nn.MultiheadAttention(num_hidden, num_heads=num_heads)
        else:
            self.interaction = lambda x, *args, **kwargs: (x, None)

    def forward(self, x: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None,
                edge_attrs: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Any]:

        social_mask = torch.full((x.size(0), x.size(0)), float('-inf'), device=x.device)
        if edge_index is not None:
            social_mask[edge_index[0], edge_index[1]] = 0
        social_mask.fill_diagonal_(0)

        # Encode agent interactions
        y, attn = self.interaction(x, x, x, attn_mask=social_mask)  # (N, D), (N, N)

        if attn is not None:
            y = y + x

        return y, attn
