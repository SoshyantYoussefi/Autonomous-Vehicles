import numpy as np
from typing import Callable


def CreateAgent(
    f: Callable[[float, np.ndarray, np.ndarray, dict], np.ndarray],
    mdlpar: dict,
    g: Callable[[np.ndarray, np.ndarray, dict], np.ndarray],
    ctrlpar: dict,
) -> dict[str]:
    """Create an agent

    Representation of a controller and dynamics ofg an agent.

    Arguments
    ---------
        f : Dynamic function for the agent: f(t, x, u, mdlpar)
        mdlpar: Dictionary with model parameters used by f
        g: Control function: g(y, xref, ctrlpar)
        ctrlpar: Dictionary with control parameters used by g

    Returns
    -------
        Dictionary representing dynamics and controller of an agent.
    """
    return {"f": f, "mdlpar": mdlpar, "g": g, "ctrlpar": ctrlpar}


class AgentFormation:
    def __init__(
        self, agents: list[dict], G: list[tuple[int, int]], formation_refs: list[Callable[[float], np.ndarray]], n: int
    ):
        """Initialize formation of agents

        Arguments
        ---------
            agents : list[dict]
                List of agents, created by CreateAgent
            G : list[tuple[imt, int]]
                List of edges in graph
            formation_refs : list[(float)->np.ndarray]
                List of functions, providing reference for corresponding agent.
            n : int
                Size of state (must be same for all agents)

        Returns
        -------
            Agent formation
        """

        def is_loop(edge):
            return len(edge) == 1
            # return edge[0] == edge[1]

        self.n = n
        self.n_agents = len(agents)
        self.agents = agents
        self.formation_references = formation_refs
        self.agent_idx = [range(n * i, n * (i + 1)) for i in range(0, self.n_agents)]
        self.measurement_graph = []
        self.absolute_measurement = np.full(self.n_agents, False)

        for idx in range(self.n_agents):
            agent_edges = [b for b in G if (len(b) == 1 and b[0] == idx) or (len(b) == 2 and b[1] == idx)]
            if len(agent_edges) == 1 and is_loop(agent_edges[0]):
                self.measurement_graph.append(self.agent_idx[agent_edges[0][0]])
                self.absolute_measurement[idx] = True
            else:
                self.measurement_graph.append(
                    np.array([np.vstack((self.agent_idx[edg[0]], self.agent_idx[edg[1]])) for edg in agent_edges])
                )

    def h_state(self, x, meas_idx):
        return x[meas_idx]  # Measure state directly

    def h_relative(self, x, meas_idx):
        # meas_idx: (num_edges, 2, num_states)
        # edge (i, j): measure x[i] - x[j]
        return x[meas_idx[:, 0, :]] - x[meas_idx[:, 1, :]]

    def ode(self, x: np.ndarray, t: float) -> np.ndarray:
        """Ode function for agent formation

        Provides f(x, t) to be used in ODE solver.

        Arguments
        ---------
            x : np.ndarray
                Current state
            t : float
                current time

        Returns
        -------
        np.ndarray
            time derivative of state
        """
        dxdt = np.zeros(self.n * self.n_agents)
        xr = x.reshape((-1, self.n))
        for idx in range(self.n_agents):
            a_i = self.agents[idx]
            xi = xr[idx]
            meas_idx = self.measurement_graph[idx]

            if self.absolute_measurement[idx]:
                y = self.h_state(x, meas_idx)
            else:
                y = self.h_relative(x, meas_idx)

            u = a_i["g"](y, self.formation_references[idx](t), a_i["ctrlpar"])

            dxdt[idx * self.n : (idx + 1) * self.n] = a_i["f"](t, xi, u, a_i["mdlpar"])
        return dxdt
