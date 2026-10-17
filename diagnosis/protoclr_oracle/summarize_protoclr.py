"""Way-number x false-negative-oracle table and pre-registered verdict.

  python summarize_protoclr.py --runs_dir runs_pc --ckpt best_val
Run folders: <runs_dir>/N<n_way>_<base|oracle>_s<seed>/

Pre-registered rule (test 5-way 1-shot, best-on-validation checkpoint):
  * dFN(N=50) >= 1.0 point and dFN grows with N  -> false negatives matter in
    large-way augmentation-based training: keep the two-error unified view
  * dFN(N=50) <  1.0 point                       -> false negatives cost little
    even at N=50: narrow the paper to false positives
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs_pc")
    ap.add_argument("--ckpt", default="best_val")
    a = ap.parse_args()
    acc = defaultdict(dict)       # (N, cfg) -> {seed: (1shot, 5shot)}
    stats = {}
    for d in sorted(glob.glob(os.path.join(a.runs_dir, "N*_*_s*"))):
        m = re.match(r"N(\d+)_(base|oracle)_s(\d+)$", os.path.basename(d))
        tj = os.path.join(d, f"test_{a.ckpt}.json")
        if not m or not os.path.exists(tj):
            continue
        n, cfg, s = int(m.group(1)), m.group(2), int(m.group(3))
        t = json.load(open(tj))
        acc[(n, cfg)][s] = (100 * t["test_1shot"], 100 * t["test_5shot"])
        if cfg == "base" and os.path.exists(os.path.join(d, "log.jsonl")):
            log = [json.loads(l) for l in open(os.path.join(d, "log.jsonl"))]
            stats[n] = (np.mean([r["fn_pairs_per_batch"] for r in log]),
                        np.mean([r["batches_with_collision"] for r in log]))

    ns = sorted({n for n, _ in acc})
    print(f"\n=== ProtoCLR: false-negative oracle vs number of ways (checkpoint {a.ckpt}) ===")
    print(f"{'N':>4}{'FN pairs/batch':>16}{'batches w/ FN':>15}{'base 1s':>10}{'oracle 1s':>11}"
          f"{'dFN 1s':>9}{'95% CI (episodes)':>20}{'seed p':>8}{'base 5s':>10}{'oracle 5s':>11}{'dFN 5s':>9}  seeds")
    d1 = {}
    for n in ns:
        b, o = acc.get((n, "base"), {}), acc.get((n, "oracle"), {})
        seeds = sorted(set(b) & set(o))
        if not seeds:
            continue
        B, O = np.array([b[s] for s in seeds]), np.array([o[s] for s in seeds])
        D = O - B
        d1[n] = D[:, 0].mean()
        fp, fc = stats.get(n, (float("nan"), float("nan")))
        # paired per-episode difference (same 10000 test episodes for every run): evaluation noise only
        ep = []
        for s in seeds:
            f = lambda cfg: os.path.join(a.runs_dir, f"N{n}_{cfg}_s{s}", f"test_1shot_{a.ckpt}.npy")
            if os.path.exists(f("base")) and os.path.exists(f("oracle")):
                ep.append(100 * (np.load(f("oracle")) - np.load(f("base"))))
        if ep:
            e = np.concatenate(ep)
            h = 1.96 * e.std() / np.sqrt(len(e))
            ci = f"[{e.mean() - h:+.2f}, {e.mean() + h:+.2f}]"
        else:
            ci = "-"
        # across-seed paired t-test: training noise (needs >= 2 seeds)
        p = float("nan")
        if len(seeds) >= 2 and D[:, 0].std(ddof=1) > 0:
            from scipy import stats as st
            p = float(st.ttest_1samp(D[:, 0], 0.0).pvalue)
        print(f"{n:>4}{fp:>16.2f}{fc:>15.3f}{B[:, 0].mean():>10.2f}{O[:, 0].mean():>11.2f}{D[:, 0].mean():>+9.2f}"
              f"{ci:>20}{p:>8.3f}{B[:, 1].mean():>10.2f}{O[:, 1].mean():>11.2f}{D[:, 1].mean():>+9.2f}  {seeds}")
    if 50 in d1:
        grows = all(d1[x] <= d1[y] + 0.3 for x, y in zip(sorted(d1), sorted(d1)[1:]))
        if d1[50] >= 1.0 and d1[min(d1)] < d1[50]:
            v = ("false negatives MATTER at large N (dFN(50) = %+.2f)%s: keep the two-error view."
                 % (d1[50], "" if grows else ", though not monotone in N"))
        else:
            v = "false negatives cost < 1 point even at N=50 (%+.2f): narrow the paper to false positives." % d1[50]
        print(f"\nVERDICT (pre-registered, test 1-shot): {v}")

    try:  # validation and loss curves, to see whether the oracle changes optimisation
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        for d in sorted(glob.glob(os.path.join(a.runs_dir, "N*_*_s1"))):
            lp = os.path.join(d, "log.jsonl")
            if not os.path.exists(lp):
                continue
            log = [json.loads(l) for l in open(lp)]
            name = os.path.basename(d)[:-3]
            ls = "-" if "base" in name else "--"
            ax[0].plot([r["iter"] for r in log], [100 * r["val_1shot"] for r in log], ls, label=name)
            ax[1].plot([r["iter"] for r in log], [r["loss"] for r in log], ls, label=name)
        ax[0].set_title("val 5-way 1-shot (%)"); ax[1].set_title("train loss"); ax[1].set_yscale("log")
        for x in ax:
            x.set_xlabel("iteration")
        ax[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(a.runs_dir, "curves_s1.png"), dpi=120)
        print(f"saved {os.path.join(a.runs_dir, 'curves_s1.png')}")
    except Exception as ex:
        print(f"(plot skipped: {ex})")


if __name__ == "__main__":
    main()
