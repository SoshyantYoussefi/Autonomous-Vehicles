# %% TSFS12 Hand-in Exercise 2: Planning for Vehicles with Differential Motion Constraints --- Motion Planning Using a State Lattice

import numpy as np
import matplotlib.pyplot as plt

# Assumes that you have all your planners in the file planners.py
from planners import breadth_first, depth_first, dijkstra, astar, best_first
from world import BoxWorld
from motionprimitives import MotionPrimitives
from plotters import (
    plot_lattice_plan,
    plot_motion_primitives,
    plot_plan_lengths_vs_planning_times,
    plot_planning_mission,
    plot_planning_time_vs_visited_nodes,
    plot_visited_nodes,
)
import os

# Run instead if you want plots in external windows
# %matplotlib


# Run the ipython magic below to activate automated import of modules. Useful if you write code in external .py files.
# %load_ext autoreload
# %autoreload 2


# %% Motion Primitives

# Run CasADi to pre-compute all motion primitives and save results in a pickle file for later re-use

# Vehicle parameters and constraints. They are also used by the heuristics
# when the motion primitives are loaded from an existing file.
L = 1.5  # Wheel base (m)
v = 15  # Constant velocity (m/s)
u_max = np.pi / 4  # Maximum steering angle (rad)

file_name = "mprims.pickle"
if os.path.exists(file_name):
    mp = MotionPrimitives(file_name)
    print(f"Read motion primitives from file {file_name}")
else:
    # Define the initial states and desired goal states for the motion
    # primitives
    theta_init = np.array(
        [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4]
    )

    x_vec = np.array([3, 2, 3, 3, 3, 1, 3, 3, 3, 2, 3])
    y_vec = np.array([2, 2, 2, 1, 1, 0, -1, -1, -2, -2, -2])
    th_vec = np.array(
        [0, np.pi / 4, np.pi / 2, 0, np.pi / 4, 0, -np.pi / 4, 0, -np.pi / 2, -np.pi / 4, 0]
    )
    state_0 = np.column_stack((x_vec, y_vec, th_vec))

    # Construct a MotionPrimitives object and generate the
    # motion primitives using the constructed lattice and
    # specification of the motion primitives
    mp = MotionPrimitives()
    mp.generate_primitives(theta_init, state_0, L, v, u_max)
    mp.save(file_name)


# Plot the computed motion primitives

# plot_motion_primitives(mp)


# %% Define Planning Mission

# Create world with obstacles using the BoxWorld class

xx = np.arange(-2, 13)
yy = np.arange(-2, 13)
th = np.array(
    [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4]
)

world = BoxWorld((xx, yy, th))

mission_nbr = 4

# Example planning missions

if mission_nbr == 1:
    world.add_box(0, 1, 2, 4)
    world.add_box(0, 6, 6, 4)
    world.add_box(4, 1, 6, 4)
    world.add_box(7, 7, 3, 3)

    start = [0, 0, 0]
    goal = [7, 8, np.pi / 2]
elif mission_nbr == 2:
    world.add_box(0, 1, 3, 4)
    world.add_box(0, 7, 10, 3)
    world.add_box(4, 1, 6, 4)

    start = [0, 0, 0]
    goal = [8, 6, np.pi / 2]
elif mission_nbr == 3:
    world.add_box(-2, 0, 10, 5)
    world.add_box(-2, 6, 10, 4)

    start = [0, 5, 0]
    goal = [0, 6, np.pi]
elif mission_nbr == 4:
    world.add_box(0, 3, 10, 2)
    world.add_box(0, 5, 4, 2)
    world.add_box(6, 5, 4, 2)

    start = [5, 7, 0]
    goal = [5, 6, 0]


# Define the initial and goal state for the graph search by finding the
# node number (column number in world.st_sp) in the world state space

mission = {
    "start": {"id": np.argmin(np.sum((world.st_sp - np.array(start)[:, None]) ** 2, axis=0))},
    "goal": {"id": np.argmin(np.sum((world.st_sp - np.array(goal)[:, None]) ** 2, axis=0))},
}


# Plot world and start and goal positions

plot_planning_mission(world, start, goal)


# %% Define State-Transition Function for Lattice Planner

# Define the state-transition function for the lattice planner


def next_state(x, world, mp, rev=True, tol=1e-5):
    """Input arguments:
     x - current state
     world - description of the map of the world
             using the class BoxWorld
     mp - object with motion primitives of the class MotionPrimitives
     rev - Allow reversing (default: True)
     tol - tolerance for comparison of closeness of states

    Output arguments:
     xi - List containing the indices of N possible next states from current
          state x, considering the obstacles and size of the world model

          To get the state corresponding to the first element in xi,
          world.st_sp[:, xi[0]]
     u - List of indices indicating which motion primitive used for reaching
         the states in xi. Each element in the list contains the two indices of the
         motion primitives used for reaching each state and the driving direction
         (1 forward, -1 reverse).

         If u_i = u[0] (first element), the corresponding motion primitive is
         mp.mprims[u_i[0], u_i[1]]
     d - List with the cost associated with each possible
         transition in xi"""

    state_i = world.st_sp[:, x]
    theta_i = state_i[2]
    mprims = mp.mprims

    xi = []
    u = []
    d = []

    # Iterate through all available primitives compatible with the current
    # angle state. Base set of motion primitives (nominally
    # corresponding to forward driving) is reversed for obtaining reverse
    # driving of the corresponding motion primitive.

    for i, j in mp.with_start_orientation_index(theta_i):
        mpi = mprims[i, j]

        # Create path to next state
        p = state_i[0:2, np.newaxis] + np.vstack((mpi["x"], mpi["y"]))
        state_next = np.vstack((p[:, -1:], mpi["th"][-1]))

        # Check if the path to next state is in the allowed area
        if not world.in_bound(state_next) or not world.obstacle_free(p):
            continue
        else:
            next_idx = np.argmin(np.sum((world.st_sp - state_next) ** 2, axis=0))
            xi.append(next_idx)
            d.append(mpi["ds"])
            u.append([i, j, 1])
    if rev:  # With reverse driving
        for i, j in mp.with_end_orientation_index(theta_i):
            mpi = mprims[i, j]

            # Create path to next state
            p = state_i[0:2, np.newaxis] + np.vstack(
                (np.flip(mpi["x"]) - mpi["x"][-1], np.flip(mpi["y"]) - mpi["y"][-1])
            )
            state_next = np.vstack((p[:, -1:], mpi["th"][0]))

            # Check if the path to next state is in the allowed area
            if not world.in_bound(state_next) or not world.obstacle_free(p):
                continue
            else:
                next_idx = np.argmin(np.sum((world.st_sp - state_next) ** 2, axis=0))
                xi.append(next_idx)
                d.append(mpi["ds"])
                u.append([i, j, -1])

    return (xi, u, d)


# The state-transition function is fully implemented. Apply it to the initial state and interpret the result.

next_state(mission["start"]["id"], world, mp)


# and do not allow reversing

next_state(mission["start"]["id"], world, mp, rev=False)


# %% Call Planners

# Get number of nodes in the state space

n = world.num_nodes()

# Define cost-to-go heuristic for planner

# Euclidian
def cost_to_go(x, xg):
    return np.linalg.norm(world.st_sp[0:2, x] - world.st_sp[0:2, xg])


# Euclidian + orientation penalty
def angle_difference(a, b):
    return abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def h_cost_pose(x, xg, heading_weight=1.0):
    position_cost = np.linalg.norm(
        world.st_sp[:2, x] - world.st_sp[:2, xg]
    )
    heading_cost = heading_weight * angle_difference(
        world.st_sp[2, x], world.st_sp[2, xg]
    )
    return position_cost + heading_cost


def h_cost_curvature_lower_bound(x, xg):
    """Lower bound based on position, heading, and minimum turning radius."""
    position_cost = np.linalg.norm(
        world.st_sp[:2, x] - world.st_sp[:2, xg]
    )
    minimum_turning_radius = L / np.tan(u_max)
    heading_cost = minimum_turning_radius * angle_difference(
        world.st_sp[2, x], world.st_sp[2, xg]
    )
    return max(position_cost, heading_cost)


# Run A* with all heuristics on the same planning mission.
heuristics = {
    "A* - Euclidean": cost_to_go,
    "A* - Euclidean + orientation": h_cost_pose,
    "A* - Curvature lower bound": h_cost_curvature_lower_bound,
}
res = []
for result_name, heuristic in heuristics.items():
    result = astar(
        n,
        mission,
        lambda x: next_state(x, world, mp, rev=True),
        heuristic=heuristic,
        num_controls=3,
    )
    if result:
        result["name"] = result_name
    res.append(result)

for r in res:
    if r:
        print(
            f"Method: {r['name']} \tLength: {r['length']:.3f}"
            f" \tVisited nodes: {r['num_expanded_nodes']}"
        )
    else:
        print("No plan found for a heuristic.")

#opt_length = [r["length"] for r in res if r["name"] == "Dijkstra"][0]  # Dijkstra is optimal
#print(f"Optimal length: {opt_length:.3f}")


# %% Plots and Analysis

for result in res:
    if not result:
        print("No plan found for a planner.")
        continue
    plot_lattice_plan(world, mp, start, goal, result)

plot_plan_lengths_vs_planning_times(res)
plot_planning_time_vs_visited_nodes({f"Mission {mission_nbr}": res})
plot_visited_nodes(res)


# %%
plt.show()
