#!/usr/bin/env python
# coding: utf-8

# %% TSFS12 Hand-in exercise 4: Collaborative Control -- initial example

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from collab_functions import CreateAgent, AgentFormation


# It is necessary to plot in an external windows for the animations to work.
# Run the ipython-command below to activate plotting in external windows.
#
# If you have no default Matplotlib backend, you might have to write
#
# ```%matplotlib tk```
#
# to explicitly state which backend to use (tk is a good choice that should work on all platforms)


# %% Model, controllers, and agent formation
# A first example are 4 agents, all modeled by a single integrator, and all measuring their absolute position.  The controller for each agent is a proportional controller.


def single_integrator(t, x, u, mdlpar):
    """
    System-model functions

     Input arguments are:
     t - time
     x - state vector of the agent (position)
     u - the control input u,
     mdlpar - structure with parameters of the model

    Output:
    dxdt - state-derivative"""

    return u


def g_absolute(y, xref, ctrlpar):
    """
    Control function
    Compute the control signal, in this case it is a P-controller which gives a control signal proportional to the vector between the current position and the reference position.

      Input arguments:
      y - measurement y,
      xref - the reference vector xref (in this case the desired position of the agent)
      ctrlpar - dictionary which contains parameters used by the controller (in this case the proportional gain k).

     Output argument:
     Control signal"""
    Kp = ctrlpar["k"]
    return -Kp * (y - xref)


# Now, formulate the formation graph and position references. The graph is specified by a set of edges. An edge (i, j) represents that agent j meauseras the distance from agent i. If agent i measures its own absolute position, this is represented by a tuple with element i as (i,) and this is the case for all agents in this example.

n = 2  # number of states is 2 (x, y)
G = [
    (0,),  # Agent 0
    (1,),  # Agent 1
    (2,),  # Agent 2
    (3,),  # Agent 3
]

formation_references = [
    lambda t: [np.cos(2 * t), np.sin(2 * t)],  # Moving counter-clockwise on a circle 1
    lambda t: [2 * np.cos(-t), 2 * np.sin(-t)],  # Moving clockwaise on a circle with radius 2
    lambda t: [3, 2],  # Fixed position
    lambda t: [5, 4],  # Fixed position
]


# Create all agents by assigning model and controller using ```CreateAgent``` and then form the formation using ```AgentFormation``` by providing the references.

modelparam = {"m": 1}
ctrl_params_1 = {"k": 1}
ctrl_params_2 = {"k": 10}

agents = [
    CreateAgent(single_integrator, modelparam, g_absolute, ctrl_params_2),
    CreateAgent(single_integrator, modelparam, g_absolute, ctrl_params_1),
    CreateAgent(single_integrator, modelparam, g_absolute, ctrl_params_1),
    CreateAgent(single_integrator, modelparam, g_absolute, ctrl_params_2),
]

formation = AgentFormation(agents, G, formation_references, n)


# %% Simulate and animate
# Define an initial state and evaluate the formatation state-transition function

x0 = np.array([1, 0, 2, 0, 3, 0, 4, 0])  # Initial values of the state vector.


# Simulate the formation

t = np.arange(0, 10, 0.05)  # Time vector for the simulation
x = odeint(formation.ode, x0, t)


# Animate the results

fig, ax = plt.subplots(num=20, clear=True)
ax.axis([-2, 6, -3, 6])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Example 0")

m = tuple([ax.plot(np.nan, np.nan, "bo")[0] for k in range(len(agents))])
text = ax.text(-1.5, 4, "")


def animate(i):
    for idx, mi in enumerate(m):
        mi.set_xdata([x[i, 0 + n * idx]])
        mi.set_ydata([x[i, 1 + n * idx]])
    text.set_text(f"t = {t[i]:.1f}")
    return m + (text,)


ani = animation.FuncAnimation(fig, animate, interval=5, frames=x.shape[0], blit=True, repeat=False)

# %%
plt.show()
