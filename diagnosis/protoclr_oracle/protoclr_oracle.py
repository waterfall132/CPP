"""ProtoCLR (ProtoTransfer's self-supervised stage) with a FALSE-NEGATIVE oracle,
for a sweep over the number of ways N (= batch size).

In ProtoCLR every image in a batch is its own class, so two images of the same
true class in one batch form a false negative. Expected same-class pairs per
batch on miniImageNet (64 balanced classes): C(N,2)/64, i.e. 0.16 at N=5,
3.8 at N=20, 19 at N=50.

  --oracle_neg 0   batch = N images drawn uniformly without replacement (ProtoCLR)
  --oracle_neg 1   batch = N DISTINCT true classes drawn uniformly, one random
                   image of each (no false negatives; same image marginal on a
                   balanced dataset). Requires N <= number of classes (64).

Everything else follows the official code: Conv4, Adam lr 1e-3 halved every
25k iterations, 1 un-augmented support + 3 augmented queries, squared
Euclidean distance, 84x84 images without normalisation. Differences on
purpose: a fixed iteration budget and checkpoint selection on VALIDATION
episodes (the official code early-stops on training accuracy).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CNN_4Layer, Plain, Views, euclid_episode_accs, protoclr_loss, read_split  # noqa: E402
from episodes import extract, make_episodes, mean_ci  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="folder containing mini_imagenet/ (84x84 images)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_way", type=int, default=50)
    ap.add_argument("--n_query", type=int, default=3)
    ap.add_argument("--oracle_neg", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--iters", type=int, default=150000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--decay_step", type=int, default=25000)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--n_val_episodes", type=int, default=600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="auto")
    return ap.parse_args()


class BatchSampler:
    """Infinite stream of index batches."""

    def __init__(self, labels, n_way, oracle, seed):
        self.labels, self.n, self.oracle = labels, n_way, oracle
        self.by_class = [np.flatnonzero(labels == c) for c in range(labels.max() + 1)]
        if oracle and n_way > len(self.by_class):
            raise ValueError(f"oracle needs n_way <= {len(self.by_class)} classes")
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        while True:
            if self.oracle:
                cls = self.rng.choice(len(self.by_class), self.n, replace=False)
                yield [int(self.rng.choice(self.by_class[c])) for c in cls]
            else:
                yield self.rng.choice(len(self.labels), self.n, replace=False).tolist()


def collision_stats(lab):
    """lab: true labels of one batch. Returns (#same-class pairs, any collision)."""
    _, cnt = np.unique(lab, return_counts=True)
    pairs = int((cnt * (cnt - 1) // 2).sum())
    return pairs, pairs > 0


def main():
    a = parse()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if a.device == "auto" else a.device))
    os.makedirs(a.out, exist_ok=True)

    tr_paths, tr_lab = read_split(a.data_root, "train")
    va_paths, va_lab = read_split(a.data_root, "val")
    val_set = Plain(va_paths)
    val_eps = make_episodes(va_lab, a.n_val_episodes, n_per_class=16, seed=0)

    enc = CNN_4Layer().to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=a.decay_step, gamma=0.5)

    resume, log_path = os.path.join(a.out, "resume.pt"), os.path.join(a.out, "log.jsonl")
    start, best, hours = 0, {"val": -1.0, "iter": 0}, 0.0
    if os.path.exists(resume):
        ck = torch.load(resume, map_location=device, weights_only=False)
        enc.load_state_dict(ck["enc"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best, hours = ck["iter"], ck["best"], ck["hours"]
        torch.set_rng_state(ck["torch_rng"].cpu())
        if os.path.exists(log_path):
            keep = [l for l in open(log_path) if json.loads(l)["iter"] <= start]
            open(log_path, "w").writelines(keep)
        print(f"resumed from iteration {start} (best val {best['val']:.4f} at {best['iter']})", flush=True)
    elif os.path.exists(log_path):
        os.remove(log_path)

    sampler = BatchSampler(tr_lab, a.n_way, a.oracle_neg, seed=a.seed * 1000003 + start)
    loader = DataLoader(Views(tr_paths, a.n_query), batch_sampler=sampler, num_workers=a.workers,
                        pin_memory=device.type == "cuda", persistent_workers=a.workers > 0)
    t0, agg = time.time(), {"loss": 0.0, "acc": 0.0, "pairs": 0, "coll": 0, "n": 0}
    enc.train()
    it = start
    for x, idx in loader:
        if it >= a.iters:
            break
        x = x.to(device, non_blocking=True).flatten(0, 1)      # way-major: N*(1+Q)
        loss, acc = protoclr_loss(enc(x), a.n_way, a.n_query)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        it += 1
        p, c = collision_stats(tr_lab[idx.numpy()])
        agg["loss"] += loss.item(); agg["acc"] += acc; agg["pairs"] += p; agg["coll"] += c; agg["n"] += 1

        if it % a.eval_every == 0 or it == a.iters:
            feats = extract(enc, val_set, device, workers=min(a.workers, 4))
            enc.train()
            va, vci = mean_ci(euclid_episode_accs(feats, val_eps, shot=1))
            n = max(agg["n"], 1)
            pairs_total = a.n_way * (a.n_way - 1) / 2
            rec = {"iter": it, "lr": sched.get_last_lr()[0], "loss": agg["loss"] / n, "train_acc": agg["acc"] / n,
                   "fn_pairs_per_batch": agg["pairs"] / n,
                   "fn_fraction_of_negatives": agg["pairs"] / n / pairs_total if pairs_total else 0.0,
                   "batches_with_collision": agg["coll"] / n,
                   "val_1shot": va, "val_ci": vci, "sec": time.time() - t0}
            if va > best["val"]:
                best = {"val": va, "iter": it}
                torch.save(enc.state_dict(), os.path.join(a.out, "best_val.pt"))
            with open(log_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"it {it:6d} loss {rec['loss']:.3f} acc {rec['train_acc']:.3f} "
                  f"FNpairs {rec['fn_pairs_per_batch']:.2f} collide {rec['batches_with_collision']:.3f} "
                  f"val {va:.4f} ({rec['sec']:.0f}s)", flush=True)
            hours += (time.time() - t0) / 3600
            torch.save({"enc": enc.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "iter": it, "best": best, "hours": hours, "torch_rng": torch.get_rng_state()},
                       resume + ".tmp")
            os.replace(resume + ".tmp", resume)
            t0, agg = time.time(), {"loss": 0.0, "acc": 0.0, "pairs": 0, "coll": 0, "n": 0}

    torch.save(enc.state_dict(), os.path.join(a.out, "last.pt"))
    with open(os.path.join(a.out, "config.json"), "w") as fh:
        json.dump({**vars(a), "best_val": best["val"], "best_val_iter": best["iter"], "train_hours": hours},
                  fh, indent=2)
    print(f"done. best val {best['val']:.4f} at iteration {best['iter']}")


if __name__ == "__main__":
    main()
