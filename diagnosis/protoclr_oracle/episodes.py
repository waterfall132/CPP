"""Fixed few-shot episodes and fast evaluation on precomputed features.

The encoder is in eval mode, so a sample's feature does not depend on which
episode it appears in; extracting every image once and indexing the features
is identical to running the encoder per episode, and far faster.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def make_episodes(labels, n_episodes, n_way=5, n_per_class=20, seed=0):
    """Each episode: n_way classes x n_per_class image indices. The first `shot`
    of each class are the support, the next 15 the query, so the same episode
    list serves 1-shot and 5-shot evaluation."""
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    per_class = [np.flatnonzero(labels == c) for c in range(labels.max() + 1)]
    eps = np.zeros((n_episodes, n_way, n_per_class), dtype=np.int64)
    for e in range(n_episodes):
        cls = rng.choice(len(per_class), n_way, replace=False)
        for j, c in enumerate(cls):
            eps[e, j] = rng.choice(per_class[c], n_per_class, replace=False)
    return eps


@torch.no_grad()
def extract(enc, ds, device, workers=4, bs=256):
    was_training = enc.training
    enc.eval()
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers)
    out = torch.cat([enc(x.to(device)).float().cpu() for x, _ in dl])
    if was_training:
        enc.train()
    return out


@torch.no_grad()
def episode_accs(feats, eps, shot, n_query=15):
    """Semi-normalized prototype classifier, as in PETD's `finetunning`:
    normalized prototypes, unnormalized queries, dot product."""
    eps = torch.as_tensor(eps)
    out = []
    for s in range(0, len(eps), 250):                          # chunks keep memory small
        ch = eps[s:s + 250]
        E, N, _ = ch.shape
        z = feats[ch[:, :, : shot + n_query]]                  # E,N,shot+q,D
        proto = F.normalize(z[:, :, :shot].mean(2), dim=-1)    # E,N,D
        q = z[:, :, shot:].reshape(E, N * n_query, -1)          # E,Nq,D
        pred = torch.einsum("end,eqd->eqn", proto, q).argmax(-1)
        y = torch.arange(N).repeat_interleave(n_query)
        out.append((pred == y).float().mean(1))
    return torch.cat(out).numpy()


def mean_ci(accs):
    accs = np.asarray(accs)
    return float(accs.mean()), float(1.96 * accs.std() / math.sqrt(len(accs)))
