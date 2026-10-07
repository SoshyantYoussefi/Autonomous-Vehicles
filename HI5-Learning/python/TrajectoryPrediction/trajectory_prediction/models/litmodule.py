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
import lightning.pytorch as pl

from torch import nn
import torch.distributions as tdist
from torch_geometric.data import HeteroData
from torch_geometric.nn import knn_graph, radius_graph, radius
from torch_geometric.utils import to_undirected, subgraph
from torch_geometric.utils import dropout_edge

from trajectory_prediction.metrics import MinADE, MinFDE, MissRate


class LitModel(pl.LightningModule):
    def __init__(self, model: nn.Module, config: dict, **kwargs) -> None:
        super().__init__()
        self.model = model
        self.dataset = config["dataset"]
        self.max_epochs = config["epochs"]
        self.learning_rate = config["lr"]
        self.r = config["radius"]
        self.agent_neighbors = config.get("agent_neighbors", 16)
        self.map_neighbors = config.get("map_neighbors", 8)
        self.standardize = config.get("standardize", False)

        self.save_hyperparameters(ignore=["model"])

        self.min_ade = MinADE()
        self.min_fde = MinFDE()
        self.mr = MissRate()

    def create_edge_indices(self, data: HeteroData) -> tuple[torch.Tensor, torch.Tensor]:
        batch = data["agent"]["batch"]
        pos = data["agent"]["inp_pos"]
        vel = data["agent"]["inp_vel"]
        mask = data["agent"]["input_mask"]

        # edge_index = knn_graph(x=pos[:, -1], k=8, batch=batch, loop=True)
        edge_index = radius_graph(
            x=pos[:, -1], r=self.r, batch=batch, loop=True, max_num_neighbors=self.agent_neighbors
        )
        edge_index = subgraph(subset=mask[:, -1], edge_index=edge_index)[0]
        edge_index = to_undirected(edge_index)

        edge_index, _ = dropout_edge(edge_index, p=0.5, training=self.training)

        dist_norm = torch.linalg.norm(pos[edge_index[0], -1] - pos[edge_index[1], -1], dim=-1, keepdim=True)
        vel_norm = torch.linalg.norm(vel[edge_index[0], -1] - vel[edge_index[1], -1], dim=-1, keepdim=True)

        edge_attrs = torch.cat([dist_norm, vel_norm], dim=-1)

        return edge_index, edge_attrs

    def post_process(self, data: HeteroData, test: bool = False) -> HeteroData:
        edge_indices, edge_attributes = self.create_edge_indices(data)
        if test:
            data = data.clone()
        data["agent"]["edge_index"] = edge_indices
        data["agent"]["edge_attr"] = edge_attributes

        if self.model.use_map and "map_point" in data.node_types and "position" in data["map_point"]:
            agent_pos = data["agent"]["inp_pos"][:, -1]
            agent_batch = data["agent"]["batch"]
            map_pos = data["map_point"]["position"]
            map_batch = data["map_point"]["batch"]

            edge_index_m2a = radius(
                x=agent_pos,
                y=map_pos,
                r=self.r,
                batch_x=agent_batch,
                batch_y=map_batch,
                max_num_neighbors=self.map_neighbors,
            )

            edge_index_m2a, _ = dropout_edge(edge_index_m2a, p=0.5, training=self.training)

            edge_type_m2a = torch.zeros_like(edge_index_m2a, dtype=torch.long)[-1:].transpose(0, 1)

            # Adjust edge indices for map-to-agent edges
            num_points = data["map_point"]["num_nodes"]
            edge_index_m2a[1] += num_points

            # map_point_type = data['map_point']['type']
            map_edge_index = data["map_point", "to", "map_point"]["edge_index"]
            map_edge_type = data["map_point", "to", "map_point"]["type"]

            map_agent_pos = torch.cat([map_pos, agent_pos], dim=0)
            # map_agent_type = torch.cat([map_point_type, torch.zeros_like(agent_pos[:, -1:], dtype=torch.long)], dim=0)
            map_agent_edge_index = torch.cat([map_edge_index, edge_index_m2a], dim=1)
            map_agent_edge_type = torch.cat([map_edge_type, edge_type_m2a], dim=0)

            data["map_point"]["position"] = map_agent_pos
            # data['map_point']["type"] = map_agent_type
            data["map_point", "to", "map_point"]["edge_index"] = map_agent_edge_index
            data["map_point", "to", "map_point"]["type"] = map_agent_edge_type[:, 0]

        return data

    def forward(self, data: HeteroData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        data = self.post_process(data, test=not self.training)
        trg = data["agent"]["trg_pos"]
        pred, attn, z = self.model(data, self.training)
        return trg, pred, attn, z

    def training_step(self, data: HeteroData, batch_idx: int) -> torch.Tensor:
        trg_vel = data["agent"]["trg_vel"]
        mask = data["agent"]["valid_mask"]

        trg_pos, pred_pos, _, z = self(data)

        pred_vel = torch.gradient(pred_pos, spacing=self.model.step_size, dim=1, edge_order=1)[0]

        pred = torch.cat([pred_pos, pred_vel], dim=-1)
        trg = torch.cat([trg_pos, trg_vel], dim=-1)

        norm = torch.linalg.norm(pred - trg, dim=-1)

        num_valid_steps = mask.sum(dim=-1)  # (N,)
        scored_agents = num_valid_steps > 0
        norm = norm * mask  # (N, T)
        norm = norm[scored_agents]
        num_valid_steps = num_valid_steps[scored_agents]
        ade = norm.sum(dim=-1) / num_valid_steps

        loss = ade.mean()

        if z is not None:
            # Only used for CVAE
            mu, sigma = z
            p = tdist.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
            q = tdist.Normal(mu, sigma)
            kl = tdist.kl_divergence(q, p).mean()

            # KL annealing
            beta = min(1, self.current_epoch / self.max_epochs)

            loss += kl * beta

        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=trg.size(0), prog_bar=True)
        return loss

    def validation_step(self, data: HeteroData, batch_idx: int) -> None:
        mask = data["agent"]["ma_mask"]

        trg, pred, *_ = self(data)

        self.min_ade.update(pred, trg, mask=mask)
        self.min_fde.update(pred, trg, mask=mask)
        self.mr.update(pred, trg, mask=mask)

        metric_dict = {"val_min_ade": self.min_ade, "val_min_fde": self.min_fde, "val_mr": self.mr}

        self.log_dict(metric_dict, on_step=False, on_epoch=True, batch_size=trg.size(0), prog_bar=True)

    def test_step(self, data: HeteroData, batch_idx: int) -> None:
        mask = data["agent"]["ma_mask"]

        trg, pred, *_ = self(data)

        if self.standardize:
            pred = pred * self.pos_sig.unsqueeze(1) + self.pos_mu.unsqueeze(1)
            trg = trg * self.pos_sig + self.pos_mu

        self.min_ade.update(pred, trg, mask=mask)
        self.min_fde.update(pred, trg, mask=mask)
        self.mr.update(pred, trg, mask=mask)

        metric_dict = {"test_min_ade": self.min_ade, "test_min_fde": self.min_fde, "test_mr": self.mr}

        self.log_dict(metric_dict, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)

        return [optimizer], [scheduler]
