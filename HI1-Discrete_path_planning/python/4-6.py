# %% Övning 4.6: jämför med och utan återanvändning.
import json
import runpy
from pathlib import Path
from time import perf_counter

import numpy as np
from queues import PriorityQueue

# Kör från samma mapp som 4-5.py.
source = runpy.run_path(str(Path(__file__).with_name("4-5.py")))
ara = source["ara"]
num_nodes = source["num_nodes"]
f_next = source["f_next"]
heuristic = source["cost_to_go"]


def astar(mission, c=1.0):
    """A* med fast epsilon och ny sökning."""
    start = mission["start"]["id"]
    goal = mission["goal"]["id"]
    costs = np.full(num_nodes, np.inf)
    previous = np.full(num_nodes, -1, dtype=int)
    costs[start] = 0
    queue = PriorityQueue()
    queue.insert(start, c * heuristic(start, goal))
    expanded = 0
    while not queue.IsEmpty() and costs[goal] > queue.peek()[1]:
        x, _ = queue.pop()
        expanded += 1
        neighbours, _, distances = f_next(x)
        for xi, di in zip(neighbours, distances):
            new_cost = costs[x] + di
            if new_cost < costs[xi]:
                costs[xi] = new_cost
                previous[xi] = x
                priority = new_cost + c * heuristic(xi, goal)
                if queue.ismember(xi):
                    queue.update_key(xi, priority)
                else:
                    queue.insert(xi, priority)  # Öppna även tidigare noder.
    if not np.isfinite(costs[goal]):
        return []
    plan = [goal]
    while plan[0] != start:
        plan.insert(0, previous[plan[0]])
    return {"plan": plan, "length": costs[goal], "num_expanded_nodes": expanded}


def measure(mission, method, factors):
    start = perf_counter()
    if method == "reuse":
        result = ara(num_nodes, mission, f_next, heuristic,
                     c_start=factors[0], c_step=0.5)
        stages = [entry["time"] for entry in result["history"]]
    elif method == "restart":
        stages = []
        for c in factors:
            result = astar(mission, c)
            stages.append(perf_counter() - start)  # Kumulativ tid.
    else:
        result = astar(mission)
        stages = [perf_counter() - start]
    elapsed = perf_counter() - start
    return elapsed, stages, result


if __name__ == "__main__":
    factors = list(np.arange(5.0, 0.99, -0.5))
    repeats = 7
    methods = ["reuse", "restart", "astar"]
    results = []
    expected = [5085.5957, 2646.2140, 1860.7143]
    for i, mission in enumerate(source["pre_mission"]):
        samples = {method: [] for method in methods}
        stage_samples = {method: [] for method in methods}
        for method in methods:
            measure(mission, method, factors)  # Värm upp.
        for repeat in range(repeats):
            order = methods[repeat % 3:] + methods[:repeat % 3]
            for method in order:  # Variera körordningen.
                elapsed, stages, result = measure(mission, method, factors)
                assert abs(result["length"] - expected[i]) < 1e-2
                path = result["plan"]
                length = sum(float(source["osm_map"].distancematrix[a, b])
                             for a, b in zip(path, path[1:]))
                assert abs(length - result["length"]) < 1e-6
                samples[method].append(elapsed * 1000)
                stage_samples[method].append(np.asarray(stages) * 1000)
        medians = {method: float(np.median(samples[method])) for method in methods}
        row = {"mission": i, "median_ms": medians, "samples_ms": samples,
               "stages_ms": {method: np.median(stage_samples[method], axis=0).tolist()
                             for method in methods}}
        row["saved_percent"] = 100 * (1 - medians["reuse"] / medians["restart"])
        row["speedup"] = medians["restart"] / medians["reuse"]
        results.append(row)
        print(f"Mission {i}: ARA* {medians['reuse']:.1f} ms, "
              f"omstarter {medians['restart']:.1f} ms, A* {medians['astar']:.1f} ms; "
              f"sparat {row['saved_percent']:.1f} %, {row['speedup']:.2f}x", flush=True)
    output = {"epsilon": factors, "repeats": repeats, "results": results}
    Path(__file__).with_name("4-6-results.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8")
