"""Dataset access, rock-type splits and the image augmentation used for training.

The prepared dataset (see scripts/prepare_data.py) is a folder with
    img/<slide id>_<k>.jpg     k = 1 plane-polarized light (PPL), k = 2..8 cross-polarized (XPL) series
    index.csv                  columns slide, k, file, rock, group
Slide ids are opaque (s0001, ...). The columns `rock` and `group` are read only to split rock
types into train / validation / test and to score evaluation; the training sampler never uses the
rock type of a training slide unless --mode supervised is chosen.
"""
import csv
import math
import os
import random
import re
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

# ImageNet normalisation (the encoder is ImageNet-pretrained)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# File names of the original Nanjing University collection start with the Chinese character of the
# rock class: \u706b (igneous), \u53d8 (metamorphic), \u6c89 (sedimentary). They are written as
# escapes so that every source file stays plain ASCII.
GROUP_CHARS = "\u706b\u53d8\u6c89"
GROUP_NAMES = {"\u706b": "Igneous", "\u53d8": "Metamorphic", "\u6c89": "Sedimentary"}
# <class char><rock number><rock name><slide number>-<k>.<ext>, e.g. "<igneous>6<name>1-2.jpg"
FILENAME_PATTERN = re.compile(r"^([" + GROUP_CHARS + r"])(\d+)(.+?)(\d+)-(\d+)\.(jpg|jpeg|png|tif|bmp)$", re.I)

PPL, XPL_FIRST, XPL_LAST = 1, 2, 8


def xpl_keys(slide_images):
    """k values of the XPL images of one slide (dict k -> file)."""
    return [k for k in sorted(slide_images) if XPL_FIRST <= k <= XPL_LAST]


def load_index(root):
    """Returns slides: {slide id: {k: file}} and rock: {slide id: rock type code}."""
    with open(os.path.join(root, "index.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    slides = defaultdict(dict)
    rock = {}
    for r in rows:
        slides[r["slide"]][int(r["k"])] = r["file"]
        rock[r["slide"]] = r["rock"]
    return slides, rock


def split_rocks(rock, seed, sizes):
    """Random split of rock TYPES into (train, val, test) lists with the given sizes.

    The primary split of the paper is seed 0 with sizes 78/10/20. If the dataset holds a different
    number of types (e.g. the synthetic test data), the sizes are scaled proportionally.
    """
    types = sorted(set(rock.values()))
    rng = random.Random(seed)
    rng.shuffle(types)
    a, b, c = sizes
    if a + b + c != len(types):
        n = len(types)
        b = max(2, round(n * b / (a + b + c)))
        c = max(5, round(n * c / (a + b + c)))
        a = n - b - c
    return types[:a], types[a:a + b], types[a + b:a + b + c]


class Images:
    """Decodes every image once into memory as uint8 arrays (shared with forked data workers)."""

    def __init__(self, root, files, verbose=True):
        self.arr = {}
        files = sorted(set(files))
        for i, f in enumerate(files):
            self.arr[f] = np.asarray(Image.open(os.path.join(root, "img", f)).convert("RGB"))
            if verbose and (i + 1) % 500 == 0:
                print(f"loaded {i + 1}/{len(files)} images", flush=True)

    def __getitem__(self, f):
        return self.arr[f]


def to_tensor(a):
    """uint8 HxWx3 array -> normalised float tensor 3xHxW."""
    return torch.from_numpy(((a.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1).copy())


def resize(a, s):
    """Resize a uint8 array to s x s pixels (bicubic)."""
    return np.asarray(Image.fromarray(a).resize((s, s), Image.BICUBIC))


def color_jitter(p, rng):
    """Control condition only (--color_jitter): SimCLR-style strong colour change.

    Hue shift of up to +-180 degrees and saturation x0.3-1.7 with probability 0.8, greyscale with
    probability 0.2. It tests whether the PPL view acts merely as a colour augmentation.
    """
    im = Image.fromarray(np.clip(p, 0, 255).astype(np.uint8))
    if rng.random() < 0.8:
        h, s_, v = [np.asarray(c, dtype=np.int16) for c in im.convert("HSV").split()]
        h = (h + int(rng.uniform(-0.5, 0.5) * 255)) % 256
        s_ = np.clip(s_ * rng.uniform(0.3, 1.7), 0, 255)
        im = Image.merge("HSV", [Image.fromarray(c.astype(np.uint8)) for c in (h, s_, v)]).convert("RGB")
    if rng.random() < 0.2:
        im = im.convert("L").convert("RGB")
    return np.asarray(im).astype(np.float32)


def augment(patch, rng, strong_colour=False):
    """Physics-aware augmentation of one square patch, returns a 224 x 224 uint8 array.

    Random sub-crop of 60-100 % of the area (aspect 3/4-4/3), random 90-degree rotation and
    horizontal flip, brightness and contrast jitter of +-20 %. Hue and saturation are NOT changed,
    because interference colours and pleochroism are diagnostic (unless strong_colour=True).
    """
    h, w = patch.shape[:2]
    area = rng.uniform(0.6, 1.0) * h * w
    ar = math.exp(rng.uniform(math.log(3 / 4), math.log(4 / 3)))
    ch, cw = min(h, int(round(math.sqrt(area / ar)))), min(w, int(round(math.sqrt(area * ar))))
    y, x = rng.randint(0, h - ch), rng.randint(0, w - cw)
    p = resize(patch[y:y + ch, x:x + cw], 224).astype(np.float32)
    p = np.rot90(p, rng.randint(0, 3))
    if rng.random() < 0.5:
        p = p[:, ::-1]
    m = p.mean()
    p = (p - m) * rng.uniform(0.8, 1.2) + m * rng.uniform(0.8, 1.2)
    if strong_colour:
        p = color_jitter(p, rng)
    return np.clip(p, 0, 255).astype(np.uint8)


def random_box(H, W, rng, lo, hi):
    """Square box (y, x, side) whose side is a random fraction in [lo, hi] of the short image side."""
    c = int(rng.uniform(lo, hi) * min(H, W))
    return rng.randint(0, H - c), rng.randint(0, W - c), c
