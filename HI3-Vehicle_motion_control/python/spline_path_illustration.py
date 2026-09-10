import numpy as np
import matplotlib.pyplot as plt
from vehiclecontrol import ControllerBase, SingleTrackModel
from splinepath import SplinePath
from seaborn import despine

# Run if you want plots in external windows
# %matplotlib


# Run the ipython magic below to activate automated import of modules. Useful if you write code in external .py files.
# %load_ext autoreload
# %autoreload 2


# %% Generate a path


class MiniController(ControllerBase):
    def __init__(self):
        super().__init__()

    def u(self, t, w):
        a = 0.0
        if t < 10:
            u = [np.pi / 180 * 10, a]
        elif t >= 10 and t < 20:
            u = [-np.pi / 180 * 11, a]
        elif t >= 20 and t < 23:
            u = [-np.pi / 180 * 0, a]
        elif t >= 23 and t < 40:
            u = [-np.pi / 180 * 15, a]
        else:
            u = [-np.pi / 180 * 0, a]
        return u


opts = {"L": 2, "amax": np.inf, "amin": -np.inf, "deltamax": np.pi / 3, "deltamin": -np.pi / 3}

car = SingleTrackModel().set_attributes(opts)
car.Ts = 0.1
car.controller = MiniController()
w0 = np.array([0, 0, 0, 2])
z0 = car.simulate(w0, T=40, dt=0.1, t0=0.0)
t, w, u = z0
M = 10
p = w[::M, 0:2]


plt.figure(10, clear=True)
plt.plot(p[:, 0], p[:, 1], "rx")


# %% Plot path and properties

pl = SplinePath(p)
print(f"Path length: {pl.length:.2f} m")


# With the SplinePath object, you can treat the path as a continuous function

s = np.linspace(0, pl.length, 100)
_, ax = plt.subplots(num=20, clear=True)
ax.plot(pl.x(s), pl.y(s), "b")
despine()


# Plot curvature

_, ax = plt.subplots(num=21, clear=True)
ax.plot(s, pl.c(s), "b")
despine()


s0 = 10
tangent, normal = pl.heading(s0)
print(f"Tangent: {tangent}")
print(f"Normal: {normal}")


L = 5  # Arrow length
_, ax = plt.subplots(num=21, clear=True)
ax.plot(pl.x(s), pl.y(s), "b")
ax.plot(pl.x(s0), pl.y(s0), "ro")
ax.arrow(pl.x(s0), pl.y(s0), L * tangent[0], L * tangent[1], width=0.15)
ax.arrow(pl.x(s0), pl.y(s0), L * normal[0], L * normal[1], width=0.15)
despine()


# %% Projection

# Plot the path and a possible current position of the vehicle

p0 = [17, 15]
_, ax = plt.subplots(num=22, clear=True)
ax.plot(pl.x(s), pl.y(s), "b")
ax.plot(p0[0], p0[1], "ko")
despine()


# Project onto the path at around s=50

s_proj, d = pl.project(p0, 50)
print("s: ", s_proj)
print("d: ", d)
print(f"Projection point at {s_proj:.3f} m, position {pl.p(s_proj)}, {d} m from point {p0}.")


# Visualize the projection. Experiment with different starting points.

p0 = [17, 15]
# s_proj, d = pl.project(p0, 70)
s_proj, d = pl.project(p0, 20)

_, ax = plt.subplots(num=10, clear=True)
ax.plot(pl.x(s), pl.y(s), "b")
ax.plot(p0[0], p0[1], "ko")
ax.plot(pl.x(s_proj), pl.y(s_proj), "ro")
ax.plot([p0[0], pl.x(s_proj)], [p0[1], pl.y(s_proj)], "k--")
ax.set_title(f"Projection point {d:.2f} m from path")
despine()


# %%
plt.show()
