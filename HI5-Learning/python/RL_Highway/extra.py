# %% TSFS12 Hand-in exercise 5, extra assignment: Deep Q-learning for highway driving

# %%# Initial imports

from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from seaborn import despine

import gymnasium
import highway_env

from dqn_agent import DQN_Agent

# %%# Utility function


def sliding_mean(r, M):
    """Compute sliding mean over M samples for r"""
    n = len(r)
    r_M = np.zeros(n)
    for k in range(1, n):
        r_M[k] = np.mean(r[np.max((0, k - M + 1)) : k])
    r_M[0] = r_M[1]
    return r_M


# %%# The simulator

# Create the OpenAI gym environment that will be used throughout the exercise

env = gymnasium.make("highway-v0", render_mode="rgb_array")


# To show how the simulator works, run the simulator with random actions

s, _ = env.reset()
done = False
while not done:
    a = env.action_space.sample()
    s, r, term, trunc, info = env.step(a)
    done = term or trunc
    env.render()
env.close()


# Available actions are

env.unwrapped.get_wrapper_attr("action_type").ACTIONS_ALL


# %%# Code skeletons for exercises

# %%## Exercise A.1

# %%## Exercise A.2

# Define the neural network model for the functional approximation
#
# ![QNN](figs/q_nn.png)


class Q_NN(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(25, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 5),
            nn.ReLU(),
        )

    def forward(self, s):
        q = self.layers(self.flatten(s))
        return q


# Define an explore-exploit strategy


def explore_exploit(k, eps):
    """Return probability for exploration action, a simple strategy

    arguments
      k -- episode number
      eps -- scalar or a tuple (p0, p1, tau)
             if eps is scalar, return constant exploration probability
             if eps is tuple, return exponential decay from p0 to p1 with time-constant tao
    """
    if type(eps) == tuple:
        p0, p1, tau = eps
        p = (p0 - p1) * np.exp(-k / tau) + p1
    else:
        p = eps
    return p


# Plot a sample explore/exploit strategy for a 3,000 episode training session.

k = np.arange(0, 3000)
_, ax = plt.subplots(num=10, figsize=(10, 7))
ax.plot(k, explore_exploit(k, (1.0, 0.05, 200)))
ax.set_xlabel("Episode")
ax.set_ylabel("Probability")
ax.set_title("Exploration/explotation")
despine()


# Load the pre-trained model and display model information

opts = {
    "explore_exploit": lambda k: explore_exploit(k, (1.0, 0.05, 200)),
    "gamma": 0.8,
    "batch_size": 32,
}

pre_trained = DQN_Agent(env, Q_NN, opts)
pre_trained.load("models/pre-trained/last_model.tar")
pre_trained.eval()  # Put the agent into non-training mode


pre_trained.summary()


# %%## Exercise A.3

# Define an agent

opts = {
    "explore_exploit": lambda k: explore_exploit(k, (1.0, 0.05, 200)),
    "gamma": 0.8,
    "batch_size": 32,
}

agent = DQN_Agent(env, Q_NN, opts)


# and train for a few episodes

agent.train(10, display=True)
env.close()


# Data from training is saved in a dictionary ```agent.stats```.

print(agent.stats.keys())


# Plot training statistics for pre-trained model

stats = pre_trained.stats  # Get saved training stats from pre-trained agent
_, ax = plt.subplots(1, 2, num=20, figsize=(14, 7))
ax[0].plot(sliding_mean(stats["episode_rewards"], 100))
ax[0].set_xlabel("Episode")
ax[0].set_title("Cumulative reward")
despine()

ax[1].plot(stats["eps"])
ax[1].set_xlabel("Episode")
ax[1].set_title("Exploration/exploitation")
despine()


# %%## Exercise A.4

# Generate a random observation

s, _ = env.reset()
env.render()
print(s)


# Evaluate the state-value function for the pre-trained model

print(pre_trained.Q(s))


# Define you own observation $s\in \mathbb{R}^{5\times 5}$ and evaluate

s = np.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
print(pre_trained.Q(s))


# Run the pre-trained model (try your own also)

# Generate a random seed between 0 and 1000; save for later if you find an interesting scenario
seed = np.random.randint(0, 1000)
# env.config["vehicles_density"] = 1.25  # Increase density, default = 1.0

s, _ = env.reset(seed=seed)
done = False
tot_reward = 0
while not done:
    a = pre_trained.act(s)
    s, r, term, trunc, info = env.step(a)
    done = term or trunc
    tot_reward += r
    env.render()
env.close()
print(f"Finished episode with total reward {tot_reward:.1f}")


# %%## Exercise A.5

# This exercise evaluates how the value faunction $V(s)$ connects to the action-value functio $Q(s, a)$ in a scenario.

seed = np.random.randint(0, 1000)

s, _ = env.reset(seed=seed)
done = False
tot_reward = 0
s_scenario = []  # List with all states during sceneario execution
while not done:
    s_scenario.append(s)

    a = pre_trained.act(s)
    s, r, term, trunc, info = env.step(a)
    done = term or trunc
    env.render()

    tot_reward += r

env.close()
print(f"Finished episode with total reward {tot_reward:.1f}")


# Use the list of all states visited during the scenario, ```s_scenrio``` to compute and plot ab estimate for ```V(s)``` for the states visited during the scenario. Also, determine an upper bound for the true ```V(s)```.

V = None  # Your code here
V_uppper_bound = 0  # Your code here

_, ax = plt.subplots(num=30, clear=True, figsize=(12, 8))
# Your code here


# %%## Exercise A.6
# %%
plt.show()
