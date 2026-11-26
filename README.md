# Enhance for Human

New QE accepts seven adjacent luma frames [B,7,1,H,W] and enhances the current
frame [B,1,H,W]. Its 449,253 parameters implement a shared encoder, key-feature
selection, three-branch aggregation, alignment, gated fusion, reconstruction
and an input residual. Training uses pure Charbonnier; there is no JDE backbone.

Four scratch models: QP22/27/32/37, seeds29/34/39/44, 300,000 successful updates
each. Batch8, crop128, Adam LR1e-4, AMP retry up to8, gradient clipping1.
Crops and augmentation are synchronized across seven inputs and their target;
epoch sampling shuffles without replacement and drops incomplete batches.

**Best checkpoint:** highest mean validation PSNR-Y (enhanced_psnr/max).
Validation uses all540 samples/QP every2000 updates and at the final update.
Loss remains logged. last.pt every1000 updates includes optimizer, scaler,
RNG, best-selection state and exact sampler position.

## Dataset

[Public Kaggle training dataset](https://www.kaggle.com/datasets/anginhok/enhance-for-human-mfqe-vvc-trainval-qp22-37)
contains GT and four LQ LMDBs, manifests and checksums, approximately73.70GiB.
Each QP has5407 train windows from108 sequences and540 validation samples
from18 sequences. Stored pixels are lossless8-bit PNG-Y. VVC reconstruction
uses randomaccess_faster at50fps and QP22/27/32/37.

**Validation overlap:** the first30 frames of each test18 sequence are used
for checkpoint selection; this is not an independent held-out benchmark.
Full test18 (7980frames/QP), YUV videos and historical weights are excluded.
See THIRD_PARTY_NOTICES.md for source attribution and original video rights.

## Kaggle setup

Use Python3.10+ and a working matched PyTorch/torchvision pair. Attach the dataset,
enable a GPU, then open notebooks/train_kaggle.ipynb or run:

    git clone https://github.com/dangdinh17/Enhance-for-Human.git
    cd Enhance-for-Human
    python -m pip install -r requirements.txt
    python -m pip install -e tools/researchctl
    python -B scripts/bootstrap_kaggle.py

Supply --dataset-root if Kaggle mounts the dataset at another location.
The first bootstrap reads all LMDB bytes to verify SHA-256, then attaches them
with symlinks. LMDBs remain on read-only Kaggle Input; no second73.70GiB copy
is created. Later calls reuse verification only while the bundle checksum
and current local file identities match.

## Checks and bounded smoke

All experiment scheduling uses researchctl. Smoke metrics and checkpoints are
diagnostics, never research results or initialization for main training.

    researchctl --profile smoke status
    researchctl --profile smoke validate
    researchctl --profile smoke plan --campaign newqe_kaggle_charb_300k_v1
    researchctl --profile smoke preflight --campaign newqe_kaggle_charb_300k_v1
    researchctl --profile smoke run --campaign newqe_kaggle_charb_300k_v1 --until smoke

Smoke runs two updates/QP, reloads after update1 and validates eight real samples.
For a CPU-only check, substitute campaign newqe_kaggle_cpu_smoke_v1 above.
CPU smoke does not qualify a GPU production environment.

    python -B -m unittest discover -s tests -v

## Main training: explicit opt-in

Bootstrap never starts main training. The notebook defaults to RUN_MAIN=False.
Kaggle-profile smoke is also required because approval is bound to that profile.

    researchctl --profile kaggle validate
    researchctl --profile kaggle plan --campaign newqe_kaggle_charb_300k_v1
    researchctl --profile kaggle preflight --campaign newqe_kaggle_charb_300k_v1
    researchctl --profile kaggle run --campaign newqe_kaggle_charb_300k_v1 --until smoke
    researchctl --profile kaggle approve --campaign newqe_kaggle_charb_300k_v1 --stage main
    researchctl --profile kaggle run --campaign newqe_kaggle_charb_300k_v1 --until main

The four runs execute sequentially on GPU0. Main requires full validation and
rejects smoke validation limits. Logs, checkpoints and receipts are stored under
runs/research/newqe_kaggle_charb_300k_v1/QP*/. The portable campaign does not
invoke JDE, tracking or full-video test. Kaggle logs to CSV; Comet is disabled
by default. No API credentials are included.

## Stop, preserve and resume

Before ending a session, cancel gracefully, then save/download runs/:

    researchctl --profile kaggle cancel --campaign newqe_kaggle_charb_300k_v1

Keep the exact repository commit, dependency versions, metadata and checkpoints.
Restore runs/ at the same project path in a later session, rerun bootstrap, then:

    researchctl --profile kaggle resume --campaign newqe_kaggle_charb_300k_v1 --until main

Resume restores optimizer, scaler, RNG and sampler; incompatible source/config/
data is rejected. Environment changes may require renewed preflight, smoke and
approval. Abrupt session termination loses updates since the last durable save.
The notebook can archive runs/ for download.

## Local provenance

The original local-data verifier is retained. Portable verification is selected
only by dataset.bundle_receipt in configs/kaggle/. Historical lab runs are
untouched. Local campaign definitions are archived under
.research/campaigns/legacy/ because they need lab source videos and old runs.
The portable release discovers only the two Kaggle campaigns.

Bundled researchctl retains explicit approval and Linux isolation checks;
preflight fails if the host lacks required isolation support. Credentials,
dataset files, generated manifests, checkpoints, approvals and conversation
diagnostics are excluded from Git. Reusable entry points stay in scripts/;
one-off helpers belong in scripts/temp/<run>/.
