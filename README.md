# Enhance for Human

New QE takes seven adjacent luma frames `[B,7,1,H,W]` and enhances the current
frame `[B,1,H,W]`. Its 449,253 parameters implement a shared encoder, key-feature
selection, three-branch aggregation, alignment, gated fusion, reconstruction
and an input residual.

Each Kaggle notebook trains **one QP**: 22, 27, 32 or 37. Training calls the
trainer directly; no orchestration package or approval command is required.

## Kaggle: setup once

Add the [public dataset](https://www.kaggle.com/datasets/anginhok/enhance-for-human-mfqe-vvc-trainval-qp22-37),
enable GPU and Internet, then run this notebook cell:

```bash
%%bash
set -e
git clone https://github.com/dangdinh17/Enhance-for-Human.git /kaggle/working/Enhance-for-Human
cd /kaggle/working/Enhance-for-Human
python -m pip install -q -r requirements.txt
python -B scripts/bootstrap_kaggle.py
```

Bootstrap finds the matching dataset under `/kaggle/input`, including nested
mount paths. If it is missing, use **Add Input** to attach the public dataset
linked above. You can also pass `--dataset-root /path/to/bundle-folder`.
Bootstrap verifies all five LMDB checksums
on first use, reading approximately 73.70 GiB, then creates read-only data links.
LMDB data stays on Kaggle Input. Repeated bootstrap calls reuse verification
while the bundle checksum and local file identities match.

## Comet credentials

Comet logs to [dangdinh17/qe-for-human](https://www.comet.com/dangdinh17/qe-for-human).
`comet_ml` is installed by requirements.txt. Before training, add a Kaggle Secret
named `COMET_API_KEY`, enable it once for the notebook, then run this Python
cell. It loads the key automatically without an input prompt:

```python
import os
from kaggle_secrets import UserSecretsClient

if not os.environ.get("COMET_API_KEY"):
    os.environ["COMET_API_KEY"] = UserSecretsClient().get_secret("COMET_API_KEY").strip()
```

A separate experiment key is generated per run and saved in `comet.json` for
resume. API keys are never embedded in source, notebook outputs or Git. Training
logs loss, PSNR-Y, delta PSNR, gradient norm, AMP scale/retries and timing every
100 updates; validation logs loss and PSNR every 2,000 updates. The startup
prints the Comet experiment URL; `comet.json` records connection status.
If the service fails, training continues with CSV logs and prints a warning.

## Train the selected QP

Change only `QP` in the next cell:

```bash
%%bash
set -e
QP=22
cd /kaggle/working/Enhance-for-Human
python -u scripts/train_mfqev2_window.py --config "configs/kaggle/QP${QP}.yml" --device cuda:0
```

The ready-made notebook is [notebooks/train_kaggle.ipynb](notebooks/train_kaggle.ipynb).
Python 3.10+ and a matched PyTorch/torchvision CUDA pair are required.

| QP | Seed | Successful optimizer updates |
|---|---:|---:|
| 22 | 29 | 300,000 |
| 27 | 34 | 300,000 |
| 32 | 39 | 300,000 |
| 37 | 44 | 300,000 |

All models start from scratch in a new run directory. Training uses pure
Charbonnier loss with epsilon 1e-6, without a JDE backbone. Batch size is 8,
crop size is 128, Adam LR is fixed at 1e-4, AMP retries the same batch up to
8 times, and gradient clipping uses norm 1. Crops and augmentation are
synchronized across all seven inputs and the target. Epoch sampling shuffles
without replacement and drops incomplete batches. CSV logging is enabled;
Comet is enabled in Kaggle configs for `dangdinh17/qe-for-human`.

**Best checkpoint:** highest mean validation PSNR-Y (`enhanced_psnr/max`).
All 540 validation samples are evaluated every 2,000 updates and at the end.
Training logs every 100 updates and saves `last.pt` every 1,000 updates.

## Outputs and resume

For QP22, outputs are under:

```text
runs/research/newqe_kaggle_comet_charb_300k_v1/QP22/
├── checkpoints/best.pth
├── checkpoints/last.pt
├── train.csv
├── validation.csv
├── comet.json
├── config.resolved.yaml
├── STARTED.json
└── DONE.json  # created only when all 300k updates finish
```

This Comet retraining version uses new run directories; previous runs are
left intact. Each other QP uses its own directory. Rerunning the same training command
resumes `last.pt` automatically, restoring model, optimizer, scaler, RNG,
checkpoint selection and sampler position. Changed config/data/source is
rejected. Stop gracefully to save the current state; abrupt termination loses
updates since the last saved checkpoint.

Before ending a Kaggle session, save the entire selected run directory.
In the next session, restore it at the same project path, use the same commit
and dependency versions, run setup/bootstrap, then run the same training cell.
Do not overwrite existing checkpoints when starting a different experiment.

## Dataset and validation limits

GT and four LQ LMDBs contain lossless 8-bit PNG-Y frames. Each QP has 5,407
training windows from 108 sequences and 540 validation samples from 18
sequences. VVC reconstruction uses randomaccess_faster at 50 fps and
QP22/27/32/37. See THIRD_PARTY_NOTICES.md for source attribution and video rights.

**Validation overlap:** the first 30 frames per test18 sequence are used for
checkpoint selection; this is not an independent held-out benchmark. Full
test18 (7,980 frames/QP), YUV videos and historical weights are excluded from
the Kaggle bundle. Training does not run tracking or full-video evaluation.

## Tests

After attaching the real dataset:

```bash
python -B -m unittest discover -s tests -v
```

Bounded smoke checks are diagnostic, never research quality results. Local
GPU/notebook execution remains unverified when the NVIDIA driver is unavailable.
Original lab datasets, contracts, checkpoints and runs are preserved separately.

## Dataset publication

Publication dependencies are separate from training:

```bash
python -m pip install -r requirements-publish.txt
```

`scripts/package_training_bundle.py` exports existing verified LMDBs without
encoding. `scripts/publish_kaggle_dataset.py` uploads the public bundle and
verifies metadata, filenames, sizes and the downloaded bundle receipt checksum.
Credentials are read from private Kaggle configuration and are never bundled.
