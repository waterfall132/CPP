"""Shared pieces for the ProtoCLR (ProtoTransfer) oracle study.

Model, augmentations and loss follow the official ProtoTransfer code
(https://github.com/indy-lab/ProtoTransfer, MIT licence, see
LICENSE_ProtoTransfer): Conv4 with 1600-d output, no input normalisation,
un-augmented support view + augmented query views, squared Euclidean
distance.
"""
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG = 84


def conv3x3(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2))


class CNN_4Layer(nn.Module):
    """Same layer names as ProtoTransfer's CNN_4Layer, so its released
    checkpoints load directly."""

    def __init__(self, in_channels=3, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(conv3x3(in_channels, hidden), conv3x3(hidden, hidden),
                                     conv3x3(hidden, hidden), conv3x3(hidden, hidden))

    def forward(self, x):
        return self.encoder(x).flatten(1)


def query_transform():  # ProtoTransfer get_custom_transform
    jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    return transforms.Compose([transforms.RandomResizedCrop(IMG, scale=(0.5, 1.0)),
                               transforms.RandomHorizontalFlip(0.5), transforms.RandomVerticalFlip(0.5),
                               transforms.RandomApply([jitter], p=0.8), transforms.RandomGrayscale(0.2),
                               transforms.ToTensor()])


def plain_transform():  # ProtoTransfer identity_transform
    return transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor()])


def read_split(data_root, split):
    root = os.path.join(data_root, "mini_imagenet")
    with open(os.path.join(root, "split", f"{split}.csv")) as f:
        rows = list(csv.DictReader(f))
    wnids = sorted({r["label"] for r in rows})
    lab = {w: i for i, w in enumerate(wnids)}
    paths = [os.path.join(root, "images", r["filename"]) for r in rows]
    return paths, np.array([lab[r["label"]] for r in rows])


class Views(Dataset):
    """Item = [1 un-augmented support view + n_query augmented views]."""

    def __init__(self, paths, n_query):
        self.paths, self.n_query = paths, n_query
        self.sup, self.q = plain_transform(), query_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        return torch.stack([self.sup(im)] + [self.q(im) for _ in range(self.n_query)]), i


class Plain(Dataset):
    def __init__(self, paths):
        self.paths, self.t = paths, plain_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.t(Image.open(self.paths[i]).convert("RGB")), 0


def protoclr_loss(z, n_way, n_query):
    """z: (n_way*(1+n_query), D), ordered way-major. 1-shot prototypes are
    the support embeddings; logits = -squared Euclidean distance."""
    z = z.view(n_way, 1 + n_query, -1)
    proto = z[:, 0]
    q = z[:, 1:].reshape(n_way * n_query, -1)
    logits = -torch.cdist(q, proto).pow(2)
    y = torch.arange(n_way, device=z.device).repeat_interleave(n_query)
    return F.cross_entropy(logits, y), (logits.argmax(1) == y).float().mean().item()


@torch.no_grad()
def euclid_episode_accs(feats, eps, shot, n_query=15):
    """ProtoNet with squared Euclidean distance (ProtoTransfer's evaluation
    without fine-tuning)."""
    eps = torch.as_tensor(eps)
    out = []
    for s in range(0, len(eps), 250):
        ch = eps[s:s + 250]
        E, N, _ = ch.shape
        z = feats[ch[:, :, : shot + n_query]]
        proto = z[:, :, :shot].mean(2)
        q = z[:, :, shot:].reshape(E, N * n_query, -1)
        pred = torch.cdist(q, proto).argmin(-1)
        y = torch.arange(N).repeat_interleave(n_query)
        out.append((pred == y).float().mean(1))
    return torch.cat(out).numpy()
