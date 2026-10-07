from typing import Optional, Iterable

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import torch
from matplotlib import colormaps
from matplotlib.collections import LineCollection
import seaborn as sns

import numpy as np
from torch import Tensor
from torch_geometric.data import HeteroData

from trajectory_prediction.metrics.utils import filter_prediction

# Get the tab20 colormap
cmap = colormaps["tab20b"]


def plot_edges_from_type(
    map_pos: torch.Tensor,
    edge_index: torch.Tensor,
    ax: plt.Axes,
    color: str = "k",
    lw: float = 1.0,
    style: str = "solid",
) -> None:
    for i in range(edge_index.shape[1]):
        source = edge_index[0, i].item()
        target = edge_index[1, i].item()
        source_pos = np.array(map_pos[source])
        target_pos = np.array(map_pos[target])
        ax.plot([source_pos[0], target_pos[0]], [source_pos[1], target_pos[1]], color=color, zorder=0, lw=lw, ls=style)


def plot_scenario_map(ax: plt.Axes, data: HeteroData, scenario_idx: int = 0):
    # Check if map is available
    if "map_point" in data.node_types:
        if "position" in data["map_point"]:
            batch_mask = data["map_point"]["batch"] == scenario_idx
            map_pos = data["map_point"]["position"][batch_mask]
            # map_type = data['map_point']['type'][batch_mask]

            batch_nodes = torch.where(batch_mask)[0]
            edge_index = data["map_point", "to", "map_point"]["edge_index"]
            edge_type = data["map_point", "to", "map_point"]["type"]
            edge_mask = torch.isin(edge_index[0], batch_nodes) & torch.isin(edge_index[1], batch_nodes)
            edge_index = edge_index[:, edge_mask]
            edge_type = edge_type[edge_mask, 0]

            # Remap node indices to start from 0
            edge_index = edge_index - edge_index.min()

            lane_nodes = edge_type == 1
            border_nodes = (edge_type == 3) | (edge_type == 2)
            stop_lines = edge_type == 4
            virtual = edge_type == 5

            # plot stop lines
            # ax.scatter(map_pos[stop_lines, 0], map_pos[stop_lines, 1], c='r', marker='.', s=10, zorder=0, alpha=0.5)
            plot_edges_from_type(map_pos, edge_index[:, stop_lines], ax, color="grey", lw=1.5)

            # plot border nodes
            # ax.scatter(map_pos[border_nodes, 0], map_pos[border_nodes, 1], c='k', marker='.', s=10, zorder=0, alpha=0.5)
            plot_edges_from_type(map_pos, edge_index[:, border_nodes], ax, color="k", lw=1.0)

            # plot lane nodes
            # ax.scatter(map_pos[lane_nodes, 0], map_pos[lane_nodes, 1], c='grey', marker='.', s=10, zorder=0, alpha=0.5)
            plot_edges_from_type(map_pos, edge_index[:, lane_nodes], ax, color="grey", lw=1.0, style="dashed")

            # plot virtual nodes
            # ax.scatter(map_pos[virtual, 0], map_pos[virtual, 1], c='b', marker='.', s=10, zorder=0, alpha=0.5)
            plot_edges_from_type(map_pos, edge_index[:, virtual], ax, color="grey", lw=0.5, style="dotted")


def get_agent_positions(data: HeteroData, scenario_idx: int, agent_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    scenario_mask = data["agent"]["batch"] == scenario_idx
    inp_pos = data["agent"]["inp_pos"][scenario_mask][agent_idx]
    inp_pos[~data["agent"]["input_mask"][scenario_mask][agent_idx]] = float("nan")
    trg_pos = data["agent"]["trg_pos"][scenario_mask][agent_idx]
    trg_pos[~data["agent"]["valid_mask"][scenario_mask][agent_idx]] = float("nan")

    return inp_pos, trg_pos


def plot_scenario(
    ax: plt.Axes,
    data: HeteroData,
    prediction: Optional[Tensor] = None,
    scenario_idx: int = 0,
    highlight_idx: Optional[int | Iterable[int] | str] = None,
    grey_out: bool = True,
    kde_estimate: bool = False,
    best_of_many: bool = True,
    zoom: bool = False,
):

    ax.plot([], [], c="k", lw=5, label="Input")
    ax.scatter([], [], c="k", s=130, marker="*", label="Agent")
    ax.scatter([], [], c="w", edgecolors="k", s=25, marker="o", label="Ground Truth")

    plot_scenario_map(ax, data, scenario_idx)

    # Get the agents in the batch
    batch_where = data["agent"]["batch"] == scenario_idx

    # Get the input and target positions for the agents in the batch
    inp_pos = data["agent"]["inp_pos"][batch_where]
    trg_pos = data["agent"]["trg_pos"][batch_where]

    input_mask = data["agent"]["input_mask"][batch_where]
    valid_mask = data["agent"]["valid_mask"][batch_where]

    # Get the prediction for the agents in the batch
    if prediction is not None:
        prediction = prediction[batch_where]

        if best_of_many and prediction.dim() == 4:
            prediction, _ = filter_prediction(prediction, trg_pos)

        if not kde_estimate:
            ax.plot([], [], c="k", label="Prediction", marker=".", lw=1.5, markersize=5)

    # Get the number of agents in the batch
    n = inp_pos.shape[0]

    # Check if the highlight index is valid
    if highlight_idx is not None:
        if isinstance(highlight_idx, int):
            assert 0 <= highlight_idx < n, f"The highlight index {highlight_idx} is out of range, must be in [0, {n})"
            highlight_idx = [highlight_idx]

        elif isinstance(highlight_idx, str):
            if highlight_idx == "ma":
                ma_mask = data["agent"]["ma_mask"][batch_where]
                highlight_idx = torch.where(ma_mask[:, 0])[0].tolist()
            elif highlight_idx == "random":
                highlight_idx = np.random.randint(n)
            elif highlight_idx == "first":
                highlight_idx = 0
            elif highlight_idx == "last":
                highlight_idx = n - 1
            else:
                raise ValueError(
                    f"highlight_idx was provided as a string {highlight_idx}, but must be ['random', "
                    f"'first', 'last']"
                )

            if isinstance(highlight_idx, int):
                highlight_idx = [highlight_idx]

        elif isinstance(highlight_idx, Iterable):
            for idx in highlight_idx:
                assert 0 <= idx < n, f"The highlight index {idx} is out of range, must be in [0, {n})"

        else:
            raise ValueError(f"highlight_idx must be an int in [0, {n} or a string ['random', 'first', 'last']")

    x_zoom = []
    y_zoom = []

    # Create a color index for each agent
    ci = np.linspace(1, 0, n)

    # Create a RandomState object with the fixed seed
    rng = np.random.RandomState(21)

    # randomize color order
    rng.shuffle(ci)

    for i in range(n):
        alpha_multiplier = 1.0
        color = cmap(ci[i])
        zorder = 2
        if highlight_idx is not None and i not in highlight_idx:
            alpha_multiplier = 0.2
            if grey_out:
                color = "gray"
            zorder = 0

        # Plot the input position (with a fading line)
        linefade = colors.to_rgb(color) + (0.0,)
        color_with_alpha = colors.to_rgb(color) + (alpha_multiplier,)
        myfade = colors.LinearSegmentedColormap.from_list("my", [linefade, color_with_alpha])

        alphas = np.clip(np.exp(np.linspace(0, 1, inp_pos.shape[1] - 1)) - 0.8, 0, 1)
        masked_pos = inp_pos[i, :, :2] * input_mask[i, :, None]
        # make zero positions nan
        masked_pos[masked_pos == 0] = np.nan
        tmp = masked_pos[:, None, :]
        segments = np.hstack((tmp[:-1], tmp[1:]))
        lc = LineCollection(segments, array=alphas, cmap=myfade, lw=5, zorder=0)
        ax.add_collection(lc)

        # plot text for highlighted agents
        if highlight_idx is not None and i in highlight_idx and len(highlight_idx) > 1:
            x0 = inp_pos[i, -1, 0]
            y0 = inp_pos[i, -1, 1]

            if zoom:
                x_zoom.append(x0)
                y_zoom.append(y0)

            ax.text(x0, y0, f"{i}", fontsize=12, color="black", zorder=10, alpha=1.0, fontweight="bold")

        # Plot position at prediction time
        ax.scatter(
            inp_pos[i, -1, 0], inp_pos[i, -1, 1], alpha=alpha_multiplier, s=135, marker="*", color=color, zorder=zorder
        )

        # Plot the ground truth target position
        masked_trg_pos = trg_pos[i, :, :2] * valid_mask[i, :, None]
        masked_trg_pos[masked_trg_pos == 0] = np.nan

        if highlight_idx is not None and i in highlight_idx and len(highlight_idx) > 1:
            x_max = np.nanmax(masked_trg_pos[:, 0])
            y_max = np.nanmax(masked_trg_pos[:, 1])

            x_min = np.nanmin(masked_trg_pos[:, 0])
            y_min = np.nanmin(masked_trg_pos[:, 1])

            if zoom:
                x_zoom.append(x_max)
                x_zoom.append(x_min)
                y_zoom.append(y_max)
                y_zoom.append(y_min)

        ax.scatter(
            masked_trg_pos[:, 0],
            masked_trg_pos[:, 1],
            alpha=alpha_multiplier,
            marker="o",
            lw=0.5,
            s=25,
            edgecolors="k",
            color=color,
            zorder=3 * zorder,
        )

        # Plot the predicted target position
        if prediction is not None:
            if highlight_idx is not None and i not in highlight_idx:
                continue
            else:
                p = prediction[i]

                if p.dim() == 2:
                    # plot the best (only) prediction
                    ax.plot(
                        p[:, 0],
                        p[:, 1],
                        alpha=alpha_multiplier,
                        marker=".",
                        lw=1.5,
                        markersize=5,
                        color=color,
                        zorder=2,
                    )

                    ax.scatter(p[-1, 0], p[-1, 1], alpha=alpha_multiplier, s=40, marker="X", color=color, zorder=2)

                elif kde_estimate:
                    assert p.dim() == 3, "Prediction must be 4D tensor for KDE estimate"

                    # plot 2d heatmap of the prediction
                    flat_predictions = p.view(-1, 2).cpu().numpy()
                    sns.kdeplot(
                        x=flat_predictions[:, 0],
                        y=flat_predictions[:, 1],
                        cmap=myfade,
                        fill=True,
                        levels=50,
                        thresh=0.05,
                        zorder=1,
                        ax=ax,
                    )

                else:
                    num_modes = p.shape[1]
                    for j in range(num_modes):
                        ax.plot(
                            p[:, j, 0],
                            p[:, j, 1],
                            alpha=alpha_multiplier,
                            marker=".",
                            lw=1.5,
                            markersize=5,
                            color=color,
                            zorder=2,
                        )

                        ax.scatter(
                            p[-1, j, 0], p[-1, j, 1], alpha=alpha_multiplier, s=40, marker="X", color=color, zorder=2
                        )

    if len(x_zoom) > 0:
        x_min = min(x_zoom) - 5
        x_max = max(x_zoom) + 5
        y_min = min(y_zoom) - 5
        y_max = max(y_zoom) + 5
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        bbox = (1.0, 1)
    else:
        bbox = None

    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    sns.despine(ax=ax)

    ax.legend(fontsize=14, loc="lower left", fancybox=True, shadow=False, handlelength=0.8, bbox_to_anchor=bbox)

    # return fig, ax


def plot_heatmap(
    ax: plt.Axes,
    data: HeteroData,
    attn_matrix: Tensor,
    scenario_idx: int = 0,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    num_ticks: int = 20,
):
    batch_where = data["agent"]["batch"] == scenario_idx

    # Get the attention matrix for the agents in the batch
    sub_attn = attn_matrix[batch_where][:, batch_where]
    n = sub_attn.shape[0]

    if end_idx is None:
        end_idx = n

    assert 0 <= start_idx < end_idx, f"start_idx is out of range, must be in [0, {end_idx})"

    # Slice the attention matrix based on the start and end indices
    slicer = slice(start_idx, end_idx)
    sub_attn = sub_attn[slicer, slicer]

    sns.heatmap(sub_attn, cmap="YlOrRd", cbar=True, ax=ax)

    # Calculate the number of ticks
    num_ticks = min(num_ticks, end_idx - start_idx)

    # Calculate tick positions and labels
    indices = np.linspace(start_idx, end_idx - 1, num_ticks).astype(int)
    positions = (indices - start_idx) / (end_idx - start_idx) * sub_attn.shape[0]

    # Set the tick positions and labels
    ax.set_xticks(positions + 0.5)
    ax.set_yticks(positions + 0.5)
    ax.set_xticklabels(indices)
    ax.set_yticklabels(indices)

    ax.set_xlabel("Key (Source) Agent Index")
    ax.set_ylabel("Query (Target) Agent Index")

    # Move x-axis ticks to the top
    ax.xaxis.tick_top()

    # Move x-axis label to the top
    ax.xaxis.set_label_position("top")

    ax.set_aspect("equal")
    ax.set_title("Attention Heatmap")
