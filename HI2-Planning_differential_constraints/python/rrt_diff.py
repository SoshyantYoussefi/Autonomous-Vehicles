# %% TSFS12 Hand-in Exercise 2: Planning for Vehicles with Differential Motion Constraints --- RRT with Motion Model for a Simple Car

import math

import matplotlib.pyplot as plt
import numpy as np

from misc import Timer
from world import BoxWorld


# %% Define the Planning World

mission_nbr = 3 

world = BoxWorld([[0, 10], [0, 10]])
if mission_nbr == 1:
    world.add_box(0, 1, 2, 4)
    world.add_box(0, 6, 6, 4)
    world.add_box(4, 1, 6, 4)
    world.add_box(7, 7, 3, 3)
    start = np.array([1, 0, np.pi / 4])
    goal = np.array([6.5, 9, np.pi / 2])
elif mission_nbr == 2:
    world.add_box(0, 1, 3, 4)
    world.add_box(0, 7, 10, 3)
    world.add_box(4, 1, 6, 4)
    start = np.array([1, 0, 0])
    goal = np.array([8, 6, np.pi / 2])
elif mission_nbr == 3:
    world.add_box(3, 0, 2, 6)
    world.add_box(6, 4, 2, 6)
    start = np.array([1, 1, 0])
    goal = np.array([9, 9, np.pi / 2])
else:
    raise ValueError("mission_nbr must be 1, 2, or 3")

_, ax = plt.subplots(num=10, clear=True)
world.draw()
ax.set_xlabel("x")
ax.set_ylabel("y")
_ = ax.axis([world.xmin, world.xmax, world.ymin, world.ymax])


# %% Car Simulation


def sim_car(xk, u, step, h=0.01, L=1.5, v=15):
    """Forward-Euler simulation of the kinematic single-track model."""

    t = 0
    N = int(step / h) + 1
    states = np.zeros((3, N))
    states[:, 0] = xk

    k = 0
    while k < N - 1:
        hk = min(h, step - t)
        states[0, k + 1] = states[0, k] + hk * v * math.cos(states[2, k])
        states[1, k + 1] = states[1, k] + hk * v * math.sin(states[2, k])
        states[2, k + 1] = states[2, k] + hk * v * math.tan(u) / L
        t = t + h
        k = k + 1

    return states


heading_weight = 0.0  # metres per radian; 0 means position-only distance


# %% RRT for Kinematic Car Model


def rrt_diff(start, goal, u_c, sim, world, opts):
    """Build an RRT for a forward-only kinematic car."""
    rg = np.random.default_rng(opts.get("seed"))

    def sample_free():
        """Sample a collision-free target pose, with optional goal bias."""
        if rg.uniform(0, 1) < opts["beta"]:
            return np.array(goal)
        else:
            found_random = False
            th = rg.uniform(0, 1) * 2 * np.pi - np.pi
            while not found_random:
                p = rg.uniform(0, 1, 2) * [world.xmax - world.xmin, world.ymax - world.ymin] + [
                    world.xmin,
                    world.ymin,
                ]
                if world.obstacle_free(p[:, None]):
                    found_random = True
        return np.array([p[0], p[1], th])

    def nearest(x):
        """Return the nearest node index."""
        return np.argmin(distance_fcn(nodes, x[:, None]))

    def steer_candidates(x_nearest, x_rand):
        """Return collision-free steering trajectories and their target distances."""
        new_paths = [sim(x_nearest, ui, opts["lambda"]) for ui in u_c]
        new_free = np.where(
            [
                world.obstacle_free(traj_i)
                and all(world.in_bound(traj_i[:2, k]) for k in range(traj_i.shape[1]))
                for traj_i in new_paths
            ]
        )[0]
        valid_new_paths = [new_paths[i] for i in new_free]

        if valid_new_paths:
            dist_to_x_rand = [distance_fcn(xi[:, -1], x_rand) for xi in valid_new_paths]
        else:
            dist_to_x_rand = -1

        return valid_new_paths, dist_to_x_rand

    def distance_fcn(x1, x2):
        """Distance between states, with an optional wrapped heading penalty."""
        d2 = x1 - x2
        dtheta = np.arctan2(np.sin(d2[2]), np.cos(d2[2]))
        return np.sqrt(d2[0] ** 2 + d2[1] ** 2 + (heading_weight * dtheta) ** 2)

    T = Timer()
    T.tic()
    nodes = start.reshape((-1, 1))
    parents = [0]
    state_trajectories = [start]

    for _ in range(opts["K"]):
        x_rand = sample_free()
        idx_nearest = nearest(x_rand)
        x_nearest = nodes[:, idx_nearest]

        candidate_paths, candidate_distances = steer_candidates(x_nearest, x_rand)
        if not candidate_paths:
            continue

        idx_best = int(np.argmin(candidate_distances))
        best_path = candidate_paths[idx_best]
        x_new = best_path[:, -1]

        nodes = np.column_stack((nodes, x_new))
        parents.append(idx_nearest)
        state_trajectories.append(best_path)

        if opts["eps"] > 0 and distance_fcn(x_new, goal) < opts["eps"]:
            break

    Tplan = T.toc()
    goal_idx = np.argmin(distance_fcn(nodes, goal[:, None]), axis=0)
    return goal_idx, nodes, parents, state_trajectories, Tplan


opts = {
    "beta": 0.05,  # Probability of selecting goal state as target state
    "lambda": 0.05,  # Step size (in time)
    "eps": 1.0,  # Threshold for stopping the search (negative for full search)
    "K": 50000,
    "seed": 42,
}

experiment = "control_set"  # "control_set" (Exercise 4.11) or "heading_weight" (Exercise 4.10)

if experiment == "control_set":
    active_opts = {**opts, "eps": -0.01, "K": 4000}
    experiment_cases = [(1.0, count) for count in (3, 5, 11, 21)]
elif experiment == "heading_weight":
    active_opts = {**opts, "eps": 1.0}
    experiment_cases = [(weight, 11) for weight in (0.0, 0.3, 1.0, 2.0, 3.0)]
else:
    raise ValueError("experiment must be 'control_set' or 'heading_weight'")


# %% Plots

def plot_rrt_result(goal_idx, parents, state_trajectories, weight, num_controls, options, figure_number):
    """Plot one tree and its backtracked solution path for an experiment case."""
    fig, ax = plt.subplots(num=figure_number, clear=True)
    world.draw(ax=ax)

    for idx in range(1, len(state_trajectories)):
        trajectory = state_trajectories[idx]
        ax.plot(trajectory[0], trajectory[1], color="0.65", lw=0.6,
                label="RRT tree" if idx == 1 else None)

    idx = goal_idx
    first_solution_edge = True
    while idx != 0:
        trajectory = state_trajectories[idx]
        ax.plot(trajectory[0], trajectory[1], color="tab:blue", lw=2.5,
                label="Planned path" if first_solution_edge else None)
        first_solution_edge = False
        idx = parents[idx]

    ax.plot(start[0], start[1], "go", ms=7, label="Start")
    ax.plot(goal[0], goal[1], "r*", ms=10, label="Goal")
    ax.set(xlim=(world.xmin, world.xmax), ylim=(world.ymin, world.ymax),
           xlabel="x", ylabel="y")
    ax.set_aspect("equal", adjustable="box")
    distance_label = "position only" if weight == 0 else f"position + heading ({weight:g} m/rad)"
    termination_label = "K iterations" if options["eps"] <= 0 else f"eps = {options['eps']:g}"
    ax.set_title(
        f"RRT car - {distance_label}, {num_controls} controls\n"
        f"{termination_label}, time step = {options['lambda']:g}"
    )
    ax.legend(loc="best")


results = []
for figure_number, (weight, num_controls) in enumerate(experiment_cases, start=99):
    heading_weight = weight
    u_c = np.linspace(-np.pi / 4, np.pi / 4, num_controls)
    goal_idx, nodes, parents, state_trajectories, Tplan = rrt_diff(
        start, goal, u_c, sim_car, world, active_opts
    )
    results.append((weight, num_controls, goal_idx, nodes, parents, state_trajectories, Tplan))
    print(
        f"heading_weight={weight:g}, controls={num_controls}: "
        f"{nodes.shape[1]} nodes, {Tplan:.2f} s"
    )
    plot_rrt_result(
        goal_idx, parents, state_trajectories, weight, num_controls, active_opts, figure_number
    )


# %%
plt.show()
