"""Check native extensions, checkpoints and (when present) real notebook data."""
import json
import os
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import torch
import pyg_lib
import torch_scatter
import torch_cluster
import torch_geometric
from torch_geometric.nn import radius_graph
from torch_geometric.data import HeteroData
from trajectory_prediction.utils import load_pretrained_model

lab = Path(__file__).resolve().parent
torch.set_num_threads(min(4, os.cpu_count() or 1))
os.chdir(lab)
print(f"Python: {sys.executable}", flush=True)
print(f"torch={torch.__version__}, numpy={np.__version__}, PyG={torch_geometric.__version__}", flush=True)
assert torch.from_numpy(np.zeros(2)).numpy().shape == (2,)
assert radius_graph(torch.tensor([[0., 0.], [1., 0.]]), r=2.).shape == (2, 2)
# A tiny artificial input checks execution only; it is not course/evaluation data.
torch.manual_seed(42)
sample = HeteroData()
sample['agent'].num_nodes = 3
for key in ('inp_pos', 'inp_vel', 'inp_acc'):
    sample['agent'][key] = torch.randn(3, 15, 2)
sample['agent']['inp_yaw'] = torch.zeros(3, 15, 1)
sample['agent']['input_mask'] = torch.ones(3, 15, dtype=torch.bool)
sample['agent']['trg_pos'] = torch.zeros(3, 25, 2)
sample['agent']['batch'] = torch.zeros(3, dtype=torch.long)
sample['map_point'].num_nodes = 4
sample['map_point']['position'] = torch.randn(4, 2)
sample['map_point']['batch'] = torch.zeros(4, dtype=torch.long)
sample['map_point', 'to', 'map_point']['edge_index'] = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
sample['map_point', 'to', 'map_point']['type'] = torch.ones(4, 1, dtype=torch.long)
for extra in ("", "map_mpnn", "map_gat"):
    model = load_pretrained_model(extra=extra).eval()
    with torch.no_grad():
        _, prediction, _, _ = model(sample)
    assert prediction.shape == (3, 25, 2) and torch.isfinite(prediction).all()
    print(f"Checkpoint OK: {extra or 'no map'} ({sum(p.numel() for p in model.parameters())} parameters)", flush=True)

notebook = json.loads((lab / "trajectory_prediction.ipynb").read_text(encoding="utf-8"))
namespace = {}
exec("".join(notebook["cells"][2]["source"]), namespace)
dataset = Path(os.environ.get("HI5_DATASET_PATH", str(lab / "datasets")))
if not (dataset / "sinD" / "test").is_dir():
    print("PACKAGES AND CHECKPOINTS OK; real-data notebook test pending: Lisam sinD/test is missing.")
    sys.exit(2)

# Execute actual notebook code, including plotting and all four models/metrics.
# Timing cells are excluded: these intentionally repeat expensive predictions.
for i, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code" or i == 2:
        continue
    source = "".join(cell["source"])
    if source.lstrip().startswith("%%timeit"):
        continue
    print(f"Notebook cell {i}", flush=True)
    exec(compile(source, f"notebook-cell-{i}", "exec"), namespace)
output = lab / "setup-check"
output.mkdir(exist_ok=True)
import matplotlib.pyplot as plt
for number in plt.get_fignums():
    plt.figure(number).savefig(output / f"figure-{number}.png")
print("PASS: notebook code, real-data predictions, plots and single-batch evaluation.")
