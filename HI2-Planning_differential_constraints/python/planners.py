import numpy as np
import matplotlib.pyplot as plt
from misc import Timer
from queues import FIFO, LIFO, PriorityQueue


def breadth_first(num_nodes, mission, f_next, heuristic=None, num_controls=0):
    """Breadth-first planner"""
    t = Timer()
    t.tic()

    unvis_node = -1
    previous = np.full(num_nodes, dtype=int, fill_value=unvis_node)
    cost_to_come = np.full(num_nodes, np.inf)
    control_to_come = np.zeros((num_nodes, num_controls), dtype=int)
    expanded_nodes = []

    startNode = mission["start"]["id"]
    goalNode = mission["goal"]["id"]

    q = FIFO()
    q.insert(startNode)
    cost_to_come[startNode] = 0
    foundPlan = False

    while not q.IsEmpty():
        x = q.pop()
        expanded_nodes.append(x)

        if x == goalNode:
            foundPlan = True
            break

        neighbours, u, d = f_next(x)
        for xi, ui, di in zip(neighbours, u, d):
            if previous[xi] == unvis_node and xi != startNode:
                previous[xi] = x
                cost_to_come[xi] = cost_to_come[x] + di
                if num_controls > 0:
                    control_to_come[xi] = ui
                q.insert(xi)

    if not foundPlan:
        return []

    plan = [goalNode]
    control = []
    while plan[0] != startNode:
        if num_controls > 0:
            control.insert(0, control_to_come[plan[0]])
        plan.insert(0, previous[plan[0]])

    return {
        "plan": plan,
        "length": cost_to_come[goalNode],
        "num_expanded_nodes": len(expanded_nodes),
        "name": "BreadthFirst",
        "time": t.toc(),
        "control": control,
        "expanded_nodes": expanded_nodes,
    }


def depth_first(num_nodes, mission, f_next, heuristic=None, num_controls=0):
    """Depth-first planner"""
    t = Timer()
    t.tic()

    unvis_node = -1
    previous = np.full(num_nodes, dtype=int, fill_value=unvis_node)
    cost_to_come = np.full(num_nodes, np.inf)
    control_to_come = np.zeros((num_nodes, num_controls), dtype=int)
    expanded_nodes = []

    startNode = mission["start"]["id"]
    goalNode = mission["goal"]["id"]

    q = LIFO()
    q.insert(startNode)
    cost_to_come[startNode] = 0
    foundPlan = False

    while not q.IsEmpty():
        x = q.pop()
        expanded_nodes.append(x)

        if x == goalNode:
            foundPlan = True
            break

        neighbours, u, d = f_next(x)
        for xi, ui, di in zip(neighbours, u, d):
            if previous[xi] == unvis_node and xi != startNode:
                previous[xi] = x
                cost_to_come[xi] = cost_to_come[x] + di
                if num_controls > 0:
                    control_to_come[xi] = ui
                q.insert(xi)

    if not foundPlan:
        return []

    plan = [goalNode]
    control = []
    while plan[0] != startNode:
        if num_controls > 0:
            control.insert(0, control_to_come[plan[0]])
        plan.insert(0, previous[plan[0]])

    return {
        "plan": plan,
        "length": cost_to_come[goalNode],
        "num_expanded_nodes": len(expanded_nodes),
        "name": "DepthFirst",
        "time": t.toc(),
        "control": control,
        "expanded_nodes": expanded_nodes,
    }


def best_first(num_nodes, mission, f_next, heuristic=None, num_controls=0):
    """Greedy best-first planner"""
    if heuristic is None:
        raise ValueError("best_first requires a heuristic function")

    t = Timer()
    t.tic()

    unvis_node = -1
    previous = np.full(num_nodes, dtype=int, fill_value=unvis_node)
    cost_to_come = np.full(num_nodes, np.inf)
    control_to_come = np.zeros((num_nodes, num_controls), dtype=int)
    expanded_nodes = []

    startNode = mission["start"]["id"]
    goalNode = mission["goal"]["id"]

    q = PriorityQueue()
    q.insert(x=startNode, priority=heuristic(startNode, goalNode))
    cost_to_come[startNode] = 0
    foundPlan = False

    while not q.IsEmpty():
        x, _ = q.pop()
        expanded_nodes.append(x)

        if x == goalNode:
            foundPlan = True
            break

        neighbours, u, d = f_next(x)
        for xi, ui, di in zip(neighbours, u, d):
            if previous[xi] == unvis_node and xi != startNode:
                previous[xi] = x
                cost_to_come[xi] = cost_to_come[x] + di
                if num_controls > 0:
                    control_to_come[xi] = ui
                q.insert(x=xi, priority=heuristic(xi, goalNode))

    if not foundPlan:
        return []

    plan = [goalNode]
    control = []
    while plan[0] != startNode:
        if num_controls > 0:
            control.insert(0, control_to_come[plan[0]])
        plan.insert(0, previous[plan[0]])

    return {
        "plan": plan,
        "length": cost_to_come[goalNode],
        "num_expanded_nodes": len(expanded_nodes),
        "name": "BestFirst",
        "time": t.toc(),
        "control": control,
        "expanded_nodes": expanded_nodes,
    }


def dijkstra(num_nodes, mission, f_next, heuristic=None, num_controls=0):
    t = Timer()
    t.tic()

    unvis_node = -1
    previous = np.full(num_nodes, dtype=int, fill_value=unvis_node)
    cost_to_come = np.full(num_nodes, np.inf)
    control_to_come = np.zeros((num_nodes, num_controls), dtype=int)
    expanded_nodes = []

    startNode = mission["start"]["id"]
    goalNode = mission["goal"]["id"]

    q = PriorityQueue()
    q.insert(x=startNode, priority=0)
    cost_to_come[startNode] = 0
    closed = set()
    foundPlan = False

    while not q.IsEmpty():
        x, _ = q.pop()

        if x in closed:
            continue
        closed.add(x)
        expanded_nodes.append(x)

        if x == goalNode:
            foundPlan = True
            break

        neighbours, u, d = f_next(x)
        for xi, ui, di in zip(neighbours, u, d):
            new_cost = cost_to_come[x] + di

            if new_cost < cost_to_come[xi]:
                previous[xi] = x
                cost_to_come[xi] = new_cost
                if num_controls > 0:
                    control_to_come[xi] = ui
                q.insert(x=xi, priority=new_cost)

    if not foundPlan:
        return []

    plan = [goalNode]
    control = []
    while plan[0] != startNode:
        if num_controls > 0:
            control.insert(0, control_to_come[plan[0]])
        plan.insert(0, previous[plan[0]])

    return {
        "plan": plan,
        "length": cost_to_come[goalNode],
        "num_expanded_nodes": len(expanded_nodes),
        "name": "Dijkstra",
        "time": t.toc(),
        "control": control,
        "expanded_nodes": expanded_nodes,
    }


def astar(num_nodes, mission, f_next, heuristic, num_controls=0):
    t = Timer()
    t.tic()

    unvis_node = -1
    previous = np.full(num_nodes, dtype=int, fill_value=unvis_node)
    cost_to_come = np.zeros(num_nodes)
    control_to_come = np.zeros((num_nodes, num_controls), dtype=int)
    expanded_nodes = []

    startNode = mission["start"]["id"]
    goalNode = mission["goal"]["id"]

    q = PriorityQueue()
    q.insert(x=startNode, priority=heuristic(startNode, goalNode))
    foundPlan = False

    while not q.IsEmpty():
        x, _ = q.pop()
        expanded_nodes.append(x)
        if x == goalNode:
            foundPlan = True
            break
        neighbours, u, d = f_next(x)

        for xi, ui, di in zip(neighbours, u, d):
            new_cost = cost_to_come[x] + di

            if previous[xi] == unvis_node or new_cost < cost_to_come[xi]:
                previous[xi] = x
                q.insert(priority=new_cost + heuristic(xi,goalNode), x=xi)
                cost_to_come[xi] = cost_to_come[x] + di
                if num_controls > 0:
                    control_to_come[xi] = ui


    # Recreate the plan by traversing previous from goal node
    if not foundPlan:
        return []
    else:
        plan = [goalNode]
        length = cost_to_come[goalNode]
        control = []
        while plan[0] != startNode:
            if num_controls > 0:
                control.insert(0, control_to_come[plan[0]])
            plan.insert(0, previous[plan[0]])

        return {
            "plan": plan,
            "length": length,
            "num_expanded_nodes": len(expanded_nodes),
            "name": "AStar",
            "time": t.toc(),
            "control": control,
            "expanded_nodes": expanded_nodes,
        }
