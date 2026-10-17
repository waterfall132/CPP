# Diagnosis on miniImageNet (Section 4 of the paper)

These experiments explain why generic unsupervised pseudo-tasks transfer poorly. They use ground-truth
labels on purpose ("oracles"), which is why they run on miniImageNet, where all training labels are
known. They are not needed to train or use XPP.

```
petd_oracle/      PETD with oracle switches and label-free neighbour selectors (Tables 4.2-4.4)
protoclr_oracle/  ProtoCLR way-number sweep with a false-negative oracle (Section 4.2)
```

## Third-party code

* **PETD** (Zhao et al., 2025, *Neurocomputing* 653, 131176): https://github.com/ZWB-1224171056/PETD.
  The repository has no license, so none of its code is included. `petd_oracle.py` imports the PETD
  data loader and model from a clone passed with `--petd_root`; the task construction is
  re-implemented in `petd_oracle.py`.
* **ProtoTransfer / ProtoCLR** (Medina et al., 2020): https://github.com/indy-lab/ProtoTransfer, MIT
  License. `protoclr_oracle/common.py` follows its model, augmentation and loss; the license is in
  `protoclr_oracle/LICENSE_ProtoTransfer`.

## Setup

```bash
pip install tqdm pandas scipy                      # in addition to requirements.txt
mkdir -p third_party
git clone https://github.com/ZWB-1224171056/PETD third_party/PETD
git clone https://github.com/indy-lab/ProtoTransfer third_party/ProtoTransfer   # ProtoCLR only

# miniImageNet, Ravi-Larochelle split: <src>/images/*.jpg, <src>/{train,val,test}.csv
python diagnosis/petd_oracle/prepare_mini.py --src data/miniImageNet--ravi --out data/mini_raw --size 0
python diagnosis/petd_oracle/prepare_mini.py --src data/miniImageNet--ravi --out data/mini84  --size 84
```

`data/mini_raw` keeps the original images (used for all PETD runs in the paper); `data/mini84` holds
84 x 84 images (used by the ProtoCLR runs, as in the official ProtoTransfer code). All scripts take the
paths from the environment variables `PETD`, `DATA_RAW`, `DATA` (defaults: `third_party/PETD`,
`data/mini_raw`, `data/mini84`, relative to the repository root).

Compute: one PETD run (100 epochs) takes several hours on one RTX 4090; scripts start runs in parallel
(`PAR`) and resume an interrupted run from `<run>/resume.pt`.

## PETD oracles (`petd_oracle/`)

Common settings follow the official PETD code: online constrained k-means with 512 clusters,
progressive replacement of query views by co-clustered samples with probability 0.6 * epoch / epochs,
semi-normalised similarity, Conv4, Adam 0.002, 32 x 256 tasks per epoch, 5-way 1-shot 5-query.
Deliberate differences: the checkpoint is selected on validation episodes, all runs share 10 000 fixed
test episodes, and training runs for 100 full epochs.

| Option of `petd_oracle.py` | Meaning | Paper |
|---|---|---|
| `--oracle_pos 1` | keep only co-clustered neighbours of the anchor's true class (false positives removed) | Table 4.2 |
| `--oracle_neg 1` | the N ways of a task have different true classes (false negatives removed) | Table 4.2 |
| `--pmax 0` | no replacement (own views only) | Table 4.2 |
| `--oracle_prec q` | each neighbour is correct with probability q | Table 4.3 |
| `--oracle_cover c` | only a fraction c of anchors use neighbours | Table 4.3 |
| `--select knn --knn_pool 10` | nearest co-clustered sample in a feature bank (label-free) | Table 4.4 |
| `--gate 1` | keep a neighbour only if it is closest to its own anchor's prototype (label-free) | Table 4.4 |
| `--select gknn --knn_pool 5` | global nearest neighbours (label-free) | Table 4.4 |
| `--cover c` | only the densest fraction c of anchors use neighbours (label-free) | Table 4.4 |

Labels are used by the label-free selectors only for logging (`nb_used_precision`, `nb_used_frac`,
`gate_precision_kept`, `gate_precision_rejected`, `gknn_table_precision`, `cover_table_precision` in
`<run>/log.jsonl`).

```bash
cd diagnosis/petd_oracle
OUT=runs_raw SEEDS=1 PAR=4 WORKERS=5 bash run_oracle.sh   # Table 4.2: baseline, oracle_pos, oracle_neg, both
python petd_oracle.py --petd_root ../../third_party/PETD --data_root ../../data/mini_raw \
       --out runs_raw/nonbr_s1 --pmax 0 --seed 1  # Table 4.2: no replacement
bash start_dose.sh                                # Table 4.3
bash start_method.sh; bash start_sel.sh           # Table 4.4
python eval_test.py --petd_root ../../third_party/PETD --data_root ../../data/mini_raw \
       --runs runs_raw/*/ runs_dose/*/ runs_method/*/ runs_sel/*/ --ckpt best_val.pt
python summarize_oracle.py --runs_dir runs_raw --ckpt best_val
```

Offline audits of a trained run (no training; Section 4.4):

```bash
python audit_neighbors.py --petd_root ../../third_party/PETD --data_root ../../data/mini_raw \
       --resume runs_method/gate_s1/resume.pt --tta 8
python audit_diffusion.py --petd_root ../../third_party/PETD --data_root ../../data/mini_raw \
       --resume runs_method/gate_s1/resume.pt
```

## ProtoCLR way sweep (`protoclr_oracle/`)

ProtoCLR treats every image of a batch of N as its own class. Two images of the same true class form a
false negative; the expected number per batch is C(N, 2) / 64 on miniImageNet (0.16 for N = 5, 19.1 for
N = 50). `--oracle_neg 1` draws N different true classes per batch.

```bash
cd diagnosis/protoclr_oracle
# check the evaluation with the released checkpoint (ProtoTransfer paper: 44.89 / 63.35)
python eval_protoclr.py --data_root ../../data/mini84 --released \
  ../../third_party/ProtoTransfer/omni-mini/prototransfer/checkpoints/protoclr/proto_miniimagenet_conv4_euclidean_1supp_3query_50bs_best.pth.tar
WAYS="5 10 20 50" SEEDS=1 PAR=4 WORKERS=5 bash run_protoclr.sh
```
