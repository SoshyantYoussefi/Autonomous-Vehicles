"""Plotting helpers for state-lattice planning"""

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np


def _new_axes(ax, figure_name, figsize=None):
    if ax is None:
        _, ax = plt.subplots(num=figure_name, clear=True, figsize=figsize)
    return ax


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_pose(ax, pose, color, label, arrow_length=1.0, arrow_width=0.075):
    """Draw a labelled position and its heading."""
    direction = arrow_length * np.array([np.cos(pose[2]), np.sin(pose[2])])
    ax.plot(*pose[:2], "o", color=color, markersize=8, label=label)
    ax.arrow(
        *pose[:2],
        *direction,
        width=arrow_width,
        edgecolor=color,
        facecolor=color,
    )


def _result_groups(results):
    if isinstance(results, Mapping):
        groups = results.items()
    else:
        groups = [(None, results)]

    return [
        (label, [result for result in group if result])
        for label, group in groups
    ]


def plot_motion_primitives(mp, ax=None):
    ax = _new_axes(ax, "Motion primitives")
    mp.plot("b", lw=0.5, ax=ax)
    ax.set(xlabel="x [m]", ylabel="y [m]", title="Motion primitives")
    ax.set_aspect("equal", adjustable="box")
    _despine(ax)
    return ax


def plot_planning_mission(world, start, goal, ax=None):
    """Plot a planning world together with the start and goal poses"""
    ax = _new_axes(ax, "Planning mission")
    world.draw(ax=ax)
    _draw_pose(ax, start, "b", "start")
    _draw_pose(ax, goal, "k", "goal")
    ax.set(
        xlabel="x [m]",
        ylabel="y [m]",
        xlim=(world.xmin, world.xmax),
        ylim=(world.ymin, world.ymax),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    _despine(ax)
    return ax


def plot_lattice_plan(world, mp, start, goal, result, ax=None):
    """Plot one lattice plan, distinguishing forward and reverse motion"""
    if not result:
        return None

    ax = _new_axes(ax, f"Lattice plan - {result['name']}")
    world.draw(ax=ax)
    path, segments = mp.plan_to_path(start, result)

    for direction, first, last in segments:
        forward = direction == 1
        ax.plot(
            path[first:last, 0],
            path[first:last, 1],
            color="tab:blue" if forward else "tab:red",
            lw=2.5,
            label="forward" if forward else "reverse",
        )

    _draw_pose(ax, start, "b", "start")
    _draw_pose(ax, goal, "k", "goal")
    ax.set(
        xlabel="x [m]",
        ylabel="y [m]",
        xlim=(world.xmin, world.xmax),
        ylim=(world.ymin, world.ymax),
        title=f"{result['name']}: length = {result['length']:.2f} m",
    )
    ax.set_aspect("equal", adjustable="box")

    # A direction may occur in several non-adjacent path segments.
    handles, labels = ax.get_legend_handles_labels()
    unique_entries = dict(zip(labels, handles))
    ax.legend(unique_entries.values(), unique_entries.keys())
    _despine(ax)
    return ax


def _plot_result_comparison(results, x_key, y_key, xlabel, ylabel, title, ax):
    ax = _new_axes(ax, title)
    groups = _result_groups(results)
    multiple_groups = len(groups) > 1 or (groups and groups[0][0] is not None)

    for group_label, group in groups:
        if not group:
            continue
        x = [result[x_key] for result in group]
        y = [result[y_key] for result in group]
        ax.scatter(x, y, s=55, label=str(group_label) if multiple_groups else None)
        for result, xi, yi in zip(group, x, y):
            ax.annotate(
                result["name"],
                (xi, yi),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize="small",
            )

    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(alpha=0.25)
    if multiple_groups:
        ax.legend(title="Mission")
    _despine(ax)
    return ax


def plot_plan_lengths_vs_planning_times(results, ax=None):
    """Plot plan length against runtime for each planner"""
    return _plot_result_comparison(
        results,
        x_key="time",
        y_key="length",
        xlabel="Planning time [s]",
        ylabel="Plan length [m]",
        title="Plan length vs. planning time",
        ax=ax,
    )


def plot_planning_time_vs_visited_nodes(results, ax=None):
    """Plot runtime against visited nodes for one or several missions"""
    return _plot_result_comparison(
        results,
        x_key="num_expanded_nodes",
        y_key="time",
        xlabel="Number of visited nodes",
        ylabel="Planning time [s]",
        title="Planning time vs. visited nodes",
        ax=ax,
    )


def plot_visited_nodes(results, ax=None):
    """Plot visited nodes and planning time as paired bars."""
    ax = _new_axes(ax, "Search effort by heuristic", figsize=(8, 5.5))
    ax.figure.set_size_inches(8, 5.5, forward=True)
    groups = _result_groups(results)
    multiple_groups = len(groups) > 1

    labels = []
    visited_nodes = []
    planning_times = []
    for group_label, group in groups:
        for result in group:
            label = result["name"]
            if multiple_groups and group_label is not None:
                label = f"{group_label}\n{label}"
            labels.append(label)
            visited_nodes.append(result["num_expanded_nodes"])
            planning_times.append(result["time"])

    positions = np.arange(len(labels))
    bar_width = 0.38
    time_ax = ax.twinx()
    node_bars = ax.bar(
        positions - bar_width / 2,
        visited_nodes,
        width=bar_width,
        color="tab:blue",
        label="Visited nodes",
    )
    time_bars = time_ax.bar(
        positions + bar_width / 2,
        planning_times,
        width=bar_width,
        color="tab:orange",
        label="Planning time",
    )

    ax.bar_label(node_bars, padding=10, fmt="%d")
    time_ax.bar_label(time_bars, padding=10, fmt="%.4f s")
    ax.set(
        ylabel="Number of visited nodes",
        title="Visited nodes and planning time by heuristic",
        xticks=positions,
        xticklabels=labels,
    )
    time_ax.set_ylabel("Planning time [s]")
    ax.set_title(ax.get_title(), pad=12)
    ax.tick_params(axis="x", labelrotation=15)
    for tick_label in ax.get_xticklabels():
        tick_label.set_horizontalalignment("right")
    ax.figure.subplots_adjust(bottom=0.19, top=0.88)
    ax.grid(axis="y", alpha=0.25)
    ax.legend([node_bars, time_bars], ["Visited nodes", "Planning time"])
    _despine(ax)
    time_ax.spines["top"].set_visible(False)
    return ax, time_ax
