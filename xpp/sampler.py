"""Training pseudo-task sampler.

Every item is one N-way pseudo-task with (n_shot + n_query) shots per way, returned as a tensor of
N * (n_shot + n_query) images in way-major order.

Modes
  views       pseudo-class = one training slide (no rock type is used). All shots of a way are views
              of ONE anchor box in ONE anchor XPL image of that slide, drawn from the enabled sources:
                aug     the anchor box in the anchor XPL image (always enabled)
                angle   the anchor box in another XPL image of the series (other stage angle)
                ppl     the anchor box in the PPL image of the same field   <- XPP
                region  another box of the slide (applied with probability p_region)
              With --schedule flat every enabled image source has weight 1, so XPP
              (--sources aug,ppl) draws each shot from the XPL or the PPL image with probability 1/2.
              With --schedule progressive the angle / ppl / region sources are ramped in over training.
  nn          label-free neighbour pseudo-tasks (the PETD idea): pseudo-class = slide, but with
              probability rising linearly to --nn_pmax a shot is taken from the slide's nearest other
              slide in the current feature space (table refreshed at every validation step).
  supervised  pseudo-class = identified rock type (reference that uses labels); --n_label_types k keeps
              only k randomly chosen identified training rock types.
"""
import random
from collections import defaultdict

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .data import PPL, augment, random_box, to_tensor, xpl_keys


def ramp(u, onset, length):
    """0 before `onset`, then linear up to 1 over `length` (u = training progress in [0, 1])."""
    return min(1.0, max(0.0, (u - onset) / length))


class PseudoTasks(IterableDataset):
    def __init__(self, a, slides, rock, train_slides, images, progress, nn_table):
        """a: parsed arguments of scripts/train.py; progress: shared mp.Value with the training
        progress in [0, 1]; nn_table: shared mp.Array, nearest other training slide (mode nn)."""
        self.a, self.slides, self.rock, self.images = a, slides, rock, images
        self.train_slides = train_slides
        self.progress, self.nn_table = progress, nn_table
        self.src = set(a.sources.split(","))
        self.on = dict(zip(["region", "angle", "ppl"], map(float, a.onsets.split(","))))
        self.by_type = defaultdict(list)
        for s in train_slides:
            self.by_type[rock[s]].append(s)
        types = sorted(self.by_type)
        if a.mode == "supervised" and a.n_label_types:
            if a.n_label_types > len(types):
                raise ValueError(f"--n_label_types {a.n_label_types} > {len(types)} training rock types")
            types = random.Random(a.split_seed + 1).sample(types, a.n_label_types)
        self.types = types

    def xpl(self, s):
        return xpl_keys(self.slides[s])

    def aug(self, patch, rng):
        return augment(patch, rng, strong_colour=self.a.color_jitter)

    def weight(self, name, u):
        if name not in self.src:
            return 0.0
        return 1.0 if self.a.schedule == "flat" else ramp(u, self.on[name], self.a.ramp)

    def view(self, s, k_a, box, u, rng, counts):
        """One shot of slide s (mode views): choose the image source, then crop and augment."""
        a = self.a
        choices = [("same", 1.0), ("angle", self.weight("angle", u)), ("ppl", self.weight("ppl", u))]
        r, src = rng.random() * sum(w for _, w in choices), "same"
        for name, w in choices:
            if r < w:
                src = name
                break
            r -= w
        if src == "angle":
            others = [k for k in self.xpl(s) if k != k_a]
            k = rng.choice(others) if others else k_a
        elif src == "ppl":
            k = PPL
        else:
            k = k_a
        im = self.images[self.slides[s][k]]
        if rng.random() < a.p_region * self.weight("region", u):
            box = random_box(im.shape[0], im.shape[1], rng, a.crop_min, a.crop_max)
            counts["region"] += 1
        counts[src] += 1
        y, x, c = box
        return self.aug(im[y:y + c, x:x + c], rng)

    def __iter__(self):
        a = self.a
        wi = get_worker_info()
        rng = random.Random(a.seed * 1000 + (wi.id if wi else 0))
        m = a.n_shot + a.n_query
        while True:
            u = self.progress.value
            counts = defaultdict(int)
            views = []
            if a.mode == "supervised":
                cls = rng.sample(self.types, min(a.n_way, len(self.types)))
                for t in cls:
                    for _ in range(m):
                        s = rng.choice(self.by_type[t])
                        im = self.images[self.slides[s][rng.choice(self.xpl(s))]]
                        y, x, c = random_box(im.shape[0], im.shape[1], rng, a.crop_min, a.crop_max)
                        views.append(self.aug(im[y:y + c, x:x + c], rng))
                n_way = len(cls)
            else:
                cls = rng.sample(self.train_slides, min(a.n_way, len(self.train_slides)))
                for s in cls:
                    k_a = rng.choice(self.xpl(s))                       # anchor XPL image
                    im = self.images[self.slides[s][k_a]]
                    box = random_box(im.shape[0], im.shape[1], rng, a.crop_min, a.crop_max)
                    for _ in range(m):
                        if a.mode == "nn" and rng.random() < a.nn_pmax * u:
                            o = self.train_slides[self.nn_table[self.train_slides.index(s)]]
                            im2 = self.images[self.slides[o][rng.choice(self.xpl(o))]]
                            y, x, c = random_box(im2.shape[0], im2.shape[1], rng, a.crop_min, a.crop_max)
                            views.append(self.aug(im2[y:y + c, x:x + c], rng))
                            counts["nn"] += 1
                        elif a.mode == "views":
                            views.append(self.view(s, k_a, box, u, rng, counts))
                        else:
                            y, x, c = box
                            views.append(self.aug(im[y:y + c, x:x + c], rng))
                            counts["same"] += 1
                n_way = len(cls)
            yield torch.stack([to_tensor(v) for v in views]), n_way, dict(counts)
