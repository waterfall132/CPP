"""Few-shot test episodes and nearest-prototype evaluation.

Only XPL images of rock types that are disjoint from training are used; a shot is one image.
  cross-slide  (primary) support images of a class come from ONE slide, its queries are all XPL images
               of its OTHER slides -> recognition of a rock type, not re-identification of a field
  image-level  (prior work) support and 15 queries per class drawn from all slides of the type pooled
"""
import math
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from .data import xpl_keys


def make_episodes(slides, rock, types, n_ep, shot, protocol, seed, n_way=5):
    """Fixed list of episodes; each is (support files per class, query files per class)."""
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for s in slides:
        if rock[s] in types:
            by_type[rock[s]].append(s)
    ok = [t for t in sorted(by_type) if len(by_type[t]) >= 2]
    xpl = lambda s: [slides[s][k] for k in xpl_keys(slides[s])]  # noqa: E731
    n_way = min(n_way, len(ok))
    eps = []
    for _ in range(n_ep):
        sup, que = [], []
        for t in rng.sample(ok, n_way):
            if protocol == "cross-slide":
                s0 = rng.choice(by_type[t])
                pool = xpl(s0)
                sup.append(rng.sample(pool, min(shot, len(pool))))
                que.append([f for s in by_type[t] if s != s0 for f in xpl(s)])
            elif protocol == "image-level":
                pool = [f for s in by_type[t] for f in xpl(s)]
                rng.shuffle(pool)
                sup.append(pool[:shot])
                que.append(pool[shot:shot + 15])
            else:
                raise ValueError(protocol)
        eps.append((sup, que))
    return eps


def evaluate(feat_of, eps):
    """Cosine nearest-prototype accuracy. feat_of: {file: normalised feature}.
    Returns (mean accuracy %, 95 % confidence half-width over episodes)."""
    accs = []
    for sup, que in eps:
        proto = F.normalize(torch.stack([torch.stack([feat_of[f] for f in s]).mean(0) for s in sup]), dim=-1)
        q = torch.stack([feat_of[f] for qs in que for f in qs])
        y = torch.tensor([i for i, qs in enumerate(que) for _ in qs])
        accs.append(((q @ proto.t()).argmax(1) == y).float().mean().item())
    accs = np.array(accs)
    return 100 * accs.mean(), 100 * 1.96 * accs.std() / math.sqrt(len(accs))
