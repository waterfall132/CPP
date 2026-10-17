# XPP: cross-polarization pairing for label-free few-shot rock thin-section classification

Code, experiment scripts and figure scripts for the article

> *Cross-polarization pairing for label-free few-shot rock thin-section classification*,
> submitted to *Computers & Geosciences*.

Few-shot thin-section classifiers are usually meta-trained on dozens of rock types that a
petrographer has identified. XPP needs **no rock identification at all**. It uses only what the
microscope records anyway: which slide an image comes from, and whether it was taken in plane-polarized
(PPL) or cross-polarized (XPL) light. Each training slide is one pseudo-class, and its shots pair the
XPL and PPL views of the **same box** of the same field, so a match must rely on grain geometry and
texture rather than colour. An ImageNet ResNet-50 is fine-tuned gently (last stage only) on these
pseudo-tasks and then recognizes **unseen rock types** from one or five images.

![XPP framework](docs/images/framework.png)

On the Nanjing University collection (108 rock types), primary split, cross-slide protocol:

| Method | Rock identifications used | 5-way 1-shot (%) | 5-way 5-shot (%) |
|---|---|---|---|
| ImageNet features | 0 | 84.96 | 86.43 |
| Instance training | 0 | 87.32 +- 0.19 | 88.27 +- 0.20 |
| **XPP** | **0** | **89.32 +- 0.27** | **90.48 +- 0.30** |
| Supervised meta-training | 78 types | 88.85 +- 0.36 | 89.79 +- 0.34 |

Mean +- standard deviation over three seeds. All tables of the paper, including the results on two
further random splits, are listed with the commands that produce them in


## Contents

```
xpp/                  the method as an importable package
  data.py             dataset index, rock-type splits, physics-aware augmentation
  sampler.py          pseudo-task sampler (XPP views, neighbour pseudo-tasks, supervised)
  model.py            ResNet-50 encoder, partial fine-tuning, whole-image features
  evaluation.py       cross-slide and image-level few-shot episodes, nearest-prototype accuracy
  presets.py          one named configuration per row of the paper's tables

diagnosis/            miniImageNet diagnosis of Section 4 (PETD and ProtoCLR oracles)

```

## Installation

Python 3.10 or newer and PyTorch with torchvision.

```bash
git clone https://github.com/<user>/xpp-petrography.git
cd xpp-petrography
pip install -r requirements.txt          # or: conda env create -f environment.yml && conda activate xpp
bash tests/smoke_test.sh                 # synthetic data, CPU only; ends with "SMOKE TEST PASSED"
```

For a CUDA build of PyTorch follow the selector at https://pytorch.org first, then install the
remaining requirements. The ImageNet weights of ResNet-50 (about 100 MB) are downloaded by torchvision on
the first run.

## Computing resources

| Task | Hardware used | Time |
|---|---|---|
| Data preparation (2634 images) | any CPU | about 5 min |
| One training run (1500 iterations, 32-way pseudo-tasks) | 1 x RTX 4090 (24 GB), 8 CPU cores | about 6-8 min |
| `experiments/run_main.sh` (Tables 6.2 and 6.3, 30 runs, 3 in parallel) | 1 x RTX 4090 | about 1.5 h |
| All thin-section experiments (`experiments/run_all.sh`) | 1 x RTX 4090 | about 5 h |
| Positive-source audit (`scripts/audit_positives.py`) | GPU / CPU | 20 min / 1-2 h |
| miniImageNet diagnosis (`diagnosis/`, 100 epochs per run) | 1 x RTX 4090 | about 6-10 h per run |
| Smoke test | CPU | about 25 min on 2 cores |

Memory: about 6 GB of GPU memory per training run and about 6 GB of RAM (all images are decoded into
memory once). A smaller GPU works with `--n_way 16`, which changes results slightly.

## Data

The photographs are the public Nanjing University teaching collection (Lai et al., 2020, *China
Scientific Data* 5(3)): 108 rock types (40 igneous, 40 metamorphic, 28 sedimentary), three slides per
type, one PPL and seven XPL photographs of one field per slide. See [docs/DATA.md](docs/DATA.md) for
download, the file-name format and the preparation.

```bash
python scripts/prepare_data.py --src data/raw --out data/rock
```

## Quick start

```bash
# XPP, primary split, seed 1 (writes runs/split0/xpp_s1/test.json)
python scripts/train.py --data data/rock --out runs/split0/xpp_s1 --preset xpp --seed 1
# the baselines of Table 6.2
python scripts/train.py --data data/rock --out runs/split0/imagenet_s1 --preset imagenet
python scripts/train.py --data data/rock --out runs/split0/instance_s1 --preset instance --seed 1
python scripts/summarize.py --runs runs/split0
```

A step-by-step walk-through, including how to apply a trained encoder to your own slides, is in
[docs/TUTORIAL.md](docs/TUTORIAL.md). Every input, output and option is described in
[docs/USER_GUIDE.md](docs/USER_GUIDE.md).

## Reproducing the paper

```bash
DATA=data/rock bash experiments/run_all.sh        # Tables 6.2-6.7 and the analysis figures
python scripts/audit_positives.py --root data/raw # Table 4.5
python paper_figures/make_figures.py              # result figures from the reported numbers
```

The miniImageNet diagnosis (Tables 4.2-4.4) needs the PETD and ProtoTransfer code bases; see
[diagnosis/README.md](diagnosis/README.md). Training uses non-deterministic GPU kernels and parallel
data loading, so re-runs agree with the paper within the reported seed-to-seed variation rather than to
the last digit.

## License

MIT, see [LICENSE](LICENSE). `diagnosis/protoclr_oracle/` adapts parts of ProtoTransfer (MIT). The
PETD repository has no license; no PETD code is included here, it is imported from a separate clone.
The thin-section photographs are not redistributed; they remain under the terms of their publisher.

## Citation

Please also cite the dataset (Lai et al., 2020).

## Contact

Questions and bug reports: please open a GitHub issue.
