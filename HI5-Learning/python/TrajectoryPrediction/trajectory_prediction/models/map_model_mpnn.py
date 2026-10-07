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

from typing import Any
import torch
from torch import nn
from torch_geometric.data import HeteroData
from trajectory_prediction.models.modules.interaction import InteractionNet
from trajectory_prediction.models.modules.map_encoder import MPNNMapEncoder, GATMapEncoder


class TrajNet(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        num_inputs = config["num_inputs"]
        num_outputs = config["num_outputs"]
        num_layers = config["num_layers"]
        num_hidden = config["num_hidden"]

        self.inter_aware = config["interaction_aware"]
        self.use_map = config["use_map"]
        self.num_modes = config["num_modes"]
        self.num_outputs = num_outputs
        self.ph = config["pred_hrz"]
        self.step_size = config["step_size"]

        # Embedding layer. Map the input space to the hidden space
        self.embed = nn.Linear(num_inputs, num_hidden)

        # GRU is a type of RNN. Used to encode the agents histories
        self.encoder = nn.GRU(num_hidden, num_hidden, num_layers, batch_first=True)

        # InteractionNet implements a MHA mechanism between nodes. Used to model the inter-agent interactions
        self.interaction = InteractionNet(num_hidden, num_heads=4)

        # Map encoder
        map_class = config["map_encoder"]
        if map_class == "mpnn":
            self.map_encoder = MPNNMapEncoder(2, num_hidden)
        elif map_class == "gat":
            self.map_encoder = GATMapEncoder(2, num_hidden)
        else:
            raise ValueError(f"Unknown map encoder class: {map_class}")

        # Combine the map and agent hidden states
        self.combiner = nn.Sequential(
            nn.Linear(num_hidden * 2, num_hidden),
            nn.GELU(),
            nn.Linear(num_hidden, num_hidden),
        )

        # GRU is a type of RNN. Used to decode the agents histories
        self.decoder = nn.GRUCell(self.num_outputs, num_hidden)

        # Output layer. Map the hidden state to the output space
        self.output = nn.Linear(num_hidden, self.num_outputs)

    def freeze_encoder(self):
        self.embed.requires_grad_(False)
        self.encoder.requires_grad_(False)
        self.interaction.requires_grad_(False)

    def forward(self, data: HeteroData, *args) -> tuple[torch.Tensor, Any, None]:
        edge_index = data["agent"]["edge_index"]  # (2, E)
        edge_attrs = data["agent"]["edge_attr"]  # (E, 1)
        x = torch.cat(
            [data["agent"]["inp_pos"], data["agent"]["inp_vel"], data["agent"]["inp_acc"], data["agent"]["inp_yaw"]],
            dim=-1,
        )  # (N, H, 5)

        x0 = x[:, -1, : self.num_outputs]  # (N, 2)

        # Embed the input space
        x = self.embed(x)  # (N, H, D)

        # Encode the agent history
        o, h = self.encoder(x)
        x = h[-1]  # (N, D)

        # If interaction is disabled, self-attention is used
        if not self.inter_aware:
            edge_index = None
            edge_attrs = None

        # Encode agent interactions
        x, attn = self.interaction(x, edge_index, edge_attrs)  # (N, D), (N, N)

        # First hidden state of the decoder is the last hidden state of the encoder
        h = x  # (N, D)

        map_h = self.map_encoder(data)

        h = self.combiner(torch.cat([h, map_h], dim=-1))

        # Decode the agent future (hidden) states
        pred = []
        for i in range(self.ph):
            h = self.decoder(x0, h)

            # Map the hidden state to the output space.
            x0 = self.output(h)  # (N, 2)
            pred.append(x0)

        pred = torch.stack(pred, dim=1)  # (N, T, 2)

        # Reshape the output tensor to match the expected shape
        return pred, attn, None
