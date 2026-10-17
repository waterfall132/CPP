"""Test evaluation (5-way 1/5-shot, 15 queries, squared Euclidean ProtoNet) on
one fixed set of 10000 episodes shared by every run.

  python eval_protoclr.py --data_root data/mini84 --runs "runs_pc/*" --ckpt best_val.pt
  python eval_protoclr.py --data_root data/mini84 --released path/to/proto_..._best.pth.tar
The second form checks our pipeline against ProtoTransfer's released ProtoCLR
checkpoint (paper, Conv4 miniImageNet, no fine-tuning: about 44.9 / 63.4).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CNN_4Layer, Plain, euclid_episode_accs, read_split  # noqa: E402
from episodes import extract, make_episodes, mean_ci  # noqa: E402


def evaluate(state, test_set, eps, device, workers):
    enc = CNN_4Layer().to(device)
    enc.load_state_dict(state)
    feats = extract(enc, test_set, device, workers=workers)
    return {s: euclid_episode_accs(feats, eps, s) for s in (1, 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--ckpt", default="best_val.pt")
    ap.add_argument("--released", default="")
    ap.add_argument("--episodes", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths, labels = read_split(a.data_root, "test")
    test_set = Plain(paths)
    eps = make_episodes(labels, a.episodes, n_per_class=20, seed=0)

    if a.released:
        ck = torch.load(a.released, map_location=device, weights_only=False)
        r = evaluate(ck["model"], test_set, eps, device, a.workers)
        print(f"released {os.path.basename(a.released)}: " + "  ".join(
            f"{s}-shot {100 * mean_ci(r[s])[0]:.2f}±{100 * mean_ci(r[s])[1]:.2f}" for s in (1, 5)))

    runs = sorted({r.rstrip("/") for p in a.runs for r in glob.glob(p)})
    for run in runs:
        ck = os.path.join(run, a.ckpt)
        if not os.path.exists(ck):
            print(f"skip {run}: no {a.ckpt}")
            continue
        r = evaluate(torch.load(ck, map_location=device), test_set, eps, device, a.workers)
        res = {"ckpt": a.ckpt, "episodes": a.episodes}
        for s in (1, 5):
            res[f"test_{s}shot"], res[f"test_{s}shot_ci"] = mean_ci(r[s])
            np.save(os.path.join(run, f"test_{s}shot_{a.ckpt[:-3]}.npy"), r[s].astype(np.float32))
        json.dump(res, open(os.path.join(run, f"test_{a.ckpt[:-3]}.json"), "w"), indent=2)
        print(f"{run}: 1-shot {100 * res['test_1shot']:.2f}±{100 * res['test_1shot_ci']:.2f}  "
              f"5-shot {100 * res['test_5shot']:.2f}±{100 * res['test_5shot_ci']:.2f}", flush=True)


if __name__ == "__main__":
    main()
