#!/usr/bin/env python3
"""Train adjacent-window MFQE on one QP or a balanced mixed-QP dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.window_mixed import (  # noqa: E402
    BalancedWindowBatches, EpochWindowBatches, MixedQPWindows, SharedWindowDataset,
    close_window_environments,
)
from src.loss.charbonnier import MFQEv2CombinedLoss  # noqa: E402
from src.utils.hashing import file_sha256  # noqa: E402
from src.models.mfqev2_window import MFQEv2Window  # noqa: E402
from src.train.metrics import MetricAccumulator  # noqa: E402
from src.train.checkpoint_selection import CheckpointSelection  # noqa: E402
from src.train.window_trainer import WindowTrainer  # noqa: E402
from src.utils.experiment import (  # noqa: E402
    ExperimentLock, atomic_write_json, atomic_write_redacted_yaml,
    canonical_config_hash, load_yaml_mapping, CometMetricLogger,
)
from src.utils.training import (  # noqa: E402
    capture_random_state, restore_random_state, save_checkpoint,
    seed_data_worker, seed_everything,
)


def worker_init(worker_id: int) -> None:
    """Limit decoder/BLAS threads inside each spawned data worker."""
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    seed_data_worker(worker_id)


def components(config: dict[str, Any], device: str) -> tuple[Any, ...]:
    """Construct verified data readers, model, optimizer and pure pixel loss."""
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    seed_everything(config["seed"])
    from src.data.human_data import verify_training_inputs
    verify_training_inputs(config)
    manifests = [ROOT / value for value in config["dataset"]["manifests"]]
    dataset = MixedQPWindows(manifests, config["dataset"]["crop_size"])
    sampling = config["dataset"].get("sampling", "balanced")
    if not dataset.qps or not set(dataset.qps) <= {22, 27, 32, 37}:
        raise ValueError("Expected supported QPs 22,27,32,37")
    if sampling == "epoch_shuffle":
        if dataset.qps != [config.get("qp")]:
            raise ValueError("Epoch shuffle requires exactly the configured single QP")
    elif sampling != "balanced" or dataset.qps != [22, 27, 32, 37]:
        raise ValueError("Balanced sampling requires QPs 22,27,32,37")
    train_sequences = {entry["sequence"] for ds in dataset.datasets for entry in ds.entries}
    validations = [SharedWindowDataset(path, "validation", config["dataset"]["crop_size"], False)
                   for path in manifests]
    val_sequences = {entry["sequence"] for ds in validations for entry in ds.entries}
    if train_sequences & val_sequences:
        raise ValueError("Training and validation sequences overlap")
    model = MFQEv2Window(**config["network"]).to(device)
    criterion = MFQEv2CombinedLoss(**config["loss"])
    optimizer = torch.optim.Adam(model.parameters(), **config["optimizer"])
    trainer = WindowTrainer(model, optimizer, criterion, device,
                            config["train"]["mixed_precision"],
                            config["train"].get("gradient_clip_norm", 1.0))
    provenance = {
        "manifest_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in manifests},
        "loss_mode": "charbonnier_only",
        "dataset_receipt_sha256": file_sha256(ROOT / config["dataset"]["completion"]),
        "samples_per_qp": {str(ds.qp): {"train": len(ds), "validation": len(vds)}
                           for ds, vds in zip(dataset.datasets, validations)},
        "train_sequences": sorted(train_sequences), "validation_sequences": sorted(val_sequences),
    }
    return dataset, validations, trainer, provenance


def make_sampler(dataset: MixedQPWindows, config: dict[str, Any],
                 start: int, stop: int) -> Any:
    """Select the configured deterministic batch stream."""
    if config["dataset"].get("sampling") == "epoch_shuffle":
        return EpochWindowBatches(len(dataset.datasets[0]), config["dataset"]["batch_size"],
                                  config["seed"], start, stop)
    return BalancedWindowBatches([len(ds) for ds in dataset.datasets],
                                 config["dataset"]["batch_size"], config["seed"], start, stop)


def train_loader(dataset: MixedQPWindows, config: dict[str, Any],
                 start: int, stop: int) -> DataLoader:
    """Create reproducible batches independent of worker scheduling."""
    workers = config["dataset"]["workers"]
    options = {"multiprocessing_context": "spawn", "persistent_workers": True} if workers else {}
    sampler = make_sampler(dataset, config, start, stop)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                      pin_memory=True, worker_init_fn=worker_init,
                      generator=torch.Generator().manual_seed(config["seed"]), **options)


def validate(trainer: WindowTrainer, datasets: list[SharedWindowDataset],
             batch_size: int,
             should_stop: Callable[[], bool] | None = None,
             max_samples: int | None = None) -> dict[str, Any]:
    """Evaluate every validation sample and give each QP equal selection weight."""
    results = {}
    for ds in datasets:
        selected = ds if max_samples is None else Subset(ds, range(min(len(ds), max_samples)))
        def batches() -> Any:
            for batch in DataLoader(selected, batch_size=batch_size, num_workers=0):
                if should_stop is not None and should_stop():
                    raise InterruptedError("Validation interrupted by a stop request")
                yield batch
        results[str(ds.qp)] = trainer.validate(batches()).to_dict()
    macro = sum(row["total_loss"] for row in results.values()) / len(results)
    psnr = sum(row["enhanced_psnr"] for row in results.values()) / len(results)
    return {"macro_total_loss": macro, "macro_enhanced_psnr": psnr, "per_qp": results}


def source_hashes() -> dict[str, str]:
    """Bind new training states to the code that defines their updates/data."""
    names = ('scripts/train_mfqev2_window.py', 'src/data/window_mixed.py', 'src/data/stdr_mfqev2_lmdb.py', 'src/data/human_data.py', 'src/train/window_trainer.py', 'src/train/base.py', 'src/train/metrics.py', 'src/models/mfqev2_window.py', 'src/models/qe_alignment_fusion.py', 'src/loss/charbonnier.py', 'src/utils/training.py', 'src/utils/experiment.py', 'src/utils/hashing.py')
    names += ('src/train/checkpoint_selection.py',)
    names += ('src/data/training_bundle.py',)
    return {name: file_sha256(ROOT / name) for name in names}


def trim_resumed_logs(directory: Path, iteration: int) -> None:
    """Remove rows beyond the last durable checkpoint after a process failure."""
    for name in ("train.csv", "validation.csv"):
        path = directory / name
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = [row for row in reader if int(row["iteration"]) <= iteration]
        if not fields:
            continue
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    """Append a flushed row, writing the header only once."""
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run(config: dict[str, Any], device: str, stop_after: int | None = None) -> None:
    """Train/resume a run and save durable state before an orderly stop."""
    selection = CheckpointSelection.from_config(config)
    validation_limit = config["train"].get("validation_max_samples")
    if validation_limit is not None and (
        config["train"]["iterations"] > 2 or not 1 <= validation_limit <= 8
    ):
        raise ValueError("Validation subsampling is only permitted for <=2-update smoke")
    directory = ROOT / config["run_dir"]
    directory.mkdir(parents=True, exist_ok=True)
    with ExperimentLock(directory / ".lock"):
        fingerprint = canonical_config_hash(config)
        identity_path = directory / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text())["config_sha256"] != fingerprint:
            raise ValueError("Refusing to mix changed configuration with existing training")
        atomic_write_json(identity_path, {"config_sha256": fingerprint, "name": config["name"]})
        atomic_write_redacted_yaml(directory / "config.resolved.yaml", config)
        dataset, validations, trainer, provenance = components(config, device)
        checkpoint_dir = directory / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        last = checkpoint_dir / "last.pt"
        iteration, best, best_iteration = 0, selection.initial_value, 0
        best_validation = None
        resume_lineage = None
        single_qp = config["dataset"].get("sampling") == "epoch_shuffle"
        sources = source_hashes()
        sampler = make_sampler(dataset, config, 0, config["train"]["iterations"])

        def sampler_state(next_iteration: int) -> dict[str, Any]:
            if isinstance(sampler, EpochWindowBatches):
                return sampler.state_at(next_iteration)
            return {"kind": "balanced", "next_iteration": next_iteration}

        start_receipt = directory / "STARTED.json"
        if not start_receipt.exists():
            if last.exists():
                raise ValueError("Refusing checkpoint without scratch-start provenance")
            atomic_write_json(start_receipt, {"started_iteration": 0, "sampler_state": sampler_state(0),
                                             "config_sha256": fingerprint, "source_sha256": sources,
                                             "initialization": "scratch", "seed": config["seed"]})
        elif json.loads(start_receipt.read_text())["config_sha256"] != fingerprint:
            raise ValueError("Scratch origin/config mismatch")

        if last.is_file():
            saved = torch.load(last, map_location="cpu", weights_only=False)
            if saved["config_sha256"] != fingerprint or saved["provenance"] != provenance:
                raise ValueError("Resume config/data provenance mismatch")
            selection.check_resume(saved["checkpoint_selection"], dataset.qps)
            if single_qp and (
                saved.get("model") != "MFQEv2Window"
                or saved.get("training_qps") != dataset.qps
                or saved.get("source_sha256") != sources
                or saved.get("sampler_state") != sampler_state(saved["iteration"])
            ):
                raise ValueError("Resume model/QP/source/sampler identity mismatch")
            trainer.model.load_state_dict(saved["state_dict"], strict=True)
            trainer.optimizer.load_state_dict(saved["optimizer_state_dict"])
            trainer.scaler.load_state_dict(saved["scaler_state_dict"])
            iteration, best, best_iteration = saved["iteration"], saved["best_value"], saved["best_iteration"]
            best_validation = saved.get("best_validation")
            resume_lineage = saved.get("resume_lineage")
            restore_random_state(saved["random_state"])
            if single_qp:
                trim_resumed_logs(directory, iteration)
        started_iteration = iteration
        requested_stop = []
        def stop(signum: int, frame: Any) -> None:
            del frame
            requested_stop.append(signum)
        previous_handlers = {signum: signal.signal(signum, stop)
                             for signum in (signal.SIGTERM, signal.SIGINT)}
        def checkpoint() -> dict[str, Any]:
            return {"model": "MFQEv2Window", "network_options": config["network"],
                    "iteration": iteration, "training_qps": dataset.qps,
                    "state_dict": trainer.model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "scaler_state_dict": trainer.scaler.state_dict(),
                    "random_state": capture_random_state(), "config_sha256": fingerprint,
                    "best_value": best, "best_iteration": best_iteration,
                    "best_validation": best_validation,
                    "provenance": provenance, "source_sha256": sources,
                    "resume_lineage": resume_lineage,
                    "sampler_state": sampler_state(iteration),
                    "checkpoint_selection": selection.metadata(dataset.qps, best)}
        def best_summary() -> dict[str, Any]:
            return {
                "best_value": selection.metadata(dataset.qps, best)["value"],
                "checkpoint_selection": selection.metadata(dataset.qps, best),
                "best_validation_total_loss": (
                    best_validation["macro_total_loss"] if best_validation else None),
                "best_validation_psnr": (
                    best_validation["macro_enhanced_psnr"] if best_validation else None),
            }
        def status(state: str, **extra: Any) -> None:
            atomic_write_json(directory / "status.json", {
                "state": state, "pid": os.getpid(), "iteration": iteration,
                "num_iterations": config["train"]["iterations"], "best_iteration": best_iteration,
                **best_summary(),
                "training_qps": dataset.qps, "seed": config["seed"],
                "started_iteration": started_iteration,
                "sampler_state": sampler_state(iteration),
                "updated_unix": time.time(), **extra})
        total = config["train"]["iterations"]
        stop_iteration = min(total, stop_after) if stop_after is not None else total
        if stop_iteration < iteration:
            raise ValueError("--stop-after is earlier than the saved iteration")
        comet = CometMetricLogger(config, directory)
        accumulator = MetricAccumulator()
        norm_sum, norm_max, norm_count = 0.0, 0.0, 0
        amp_retries = 0
        last_print, last_step = time.monotonic(), iteration
        status("running")
        loader = None
        iterator = None
        try:
            loader = train_loader(dataset, config, iteration, stop_iteration)
            iterator = iter(loader)
            for batch in iterator:
                if requested_stop:
                    break
                if sorted(set(batch["qp"].tolist())) != sorted(dataset.qps):
                    raise ValueError("Training batch QP does not match configured run")
                accumulator.update(trainer.train_batch(batch))
                iteration += 1
                amp_retries += trainer.last_amp_retries
                norm_sum += trainer.last_gradient_norm
                norm_max = max(norm_max, trainer.last_gradient_norm)
                norm_count += 1
                if iteration == 1 or iteration % config["train"]["log_every"] == 0:
                    seconds = time.monotonic() - last_print
                    diagnostics = {k: v.cpu().tolist() for k, v in trainer.model.last_diagnostics.items()}
                    row = {"iteration": iteration, **accumulator.compute().to_dict(),
                           "seconds_per_step": seconds / (iteration - last_step)}
                    if single_qp:
                        row.update(gradient_norm_mean=norm_sum / norm_count,
                                   gradient_norm_max=norm_max,
                                   epoch=sampler_state(iteration)["next_epoch"] + 1,
                                   qp=dataset.qps[0], amp_retries=amp_retries,
                                   loss_scale=trainer.scaler.get_scale())
                    append_csv(directory / "train.csv", row)
                    comet.log_metrics("train", row, iteration)
                    print(json.dumps({"train": row, "selection": diagnostics}), flush=True)
                    status("running", seconds_per_step=row["seconds_per_step"], diagnostics=diagnostics)
                    accumulator = MetricAccumulator()
                    norm_sum, norm_max, norm_count = 0.0, 0.0, 0
                    amp_retries = 0
                    last_print, last_step = time.monotonic(), iteration
                if requested_stop:
                    break
                if iteration % config["train"]["validate_every"] == 0 or iteration == total:
                    status("validating")
                    result = validate(trainer, validations, config["dataset"]["batch_size"],
                                      should_stop=lambda: bool(requested_stop),
                                      max_samples=validation_limit)
                    for qp, metrics in result["per_qp"].items():
                        append_csv(directory / "validation.csv", {"iteration": iteration, "qp": qp, **metrics})
                    comet.log_metrics("validation", result["per_qp"][str(dataset.qps[0])], iteration)
                    candidate = selection.score(result)
                    if selection.improves(candidate, best):
                        best, best_iteration = candidate, iteration
                        best_validation = result
                        save_checkpoint({"model": "MFQEv2Window", "network_options": config["network"],
                                         "state_dict": trainer.model.state_dict(), "iteration": iteration,
                                         "training_qps": dataset.qps, "validation": result,
                                         "config_sha256": fingerprint, "provenance": provenance,
                                         "source_sha256": sources,
                                         "resume_lineage": resume_lineage,
                                         "checkpoint_selection": selection.metadata(dataset.qps, best)},
                                        checkpoint_dir / "best.pth")
                    print(json.dumps({"validation": result, "iteration": iteration}), flush=True)
                    last_print, last_step = time.monotonic(), iteration
                    status("running")
                if iteration % config["train"]["checkpoint_every"] == 0 or iteration == stop_iteration or requested_stop:
                    save_checkpoint(checkpoint(), last)
            if requested_stop:
                save_checkpoint(checkpoint(), last)
                status("interrupted", signal=requested_stop[-1])
            elif iteration == total:
                atomic_write_json(directory / "DONE.json", {
                    "iteration": iteration, "best_iteration": best_iteration,
                    **best_summary(), "config_sha256": fingerprint,
                    "checkpoint_sha256": file_sha256(checkpoint_dir / "best.pth"),
                    "last_checkpoint_sha256": file_sha256(last),
                    "training_qps": dataset.qps, "source_sha256": sources,
                    "training_samples": iteration * config["dataset"]["batch_size"]})
                status("complete")
            elif not requested_stop:
                status("stopped_early")
        except InterruptedError:
            if not requested_stop:
                raise
            save_checkpoint(checkpoint(), last)
            status("interrupted", signal=requested_stop[-1])
        except Exception as error:
            save_checkpoint(checkpoint(), last)
            status("failed", error=repr(error))
            raise
        finally:
            if iterator is not None:
                shutdown = getattr(iterator, "_shutdown_workers", None)
                if shutdown is not None:
                    shutdown()
            del iterator, loader
            close_window_environments()
            comet.close()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
        if requested_stop:
            raise SystemExit(128 + requested_stop[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-after", type=int, help="Save and exit at this iteration; next invocation resumes")
    args = parser.parse_args()
    config = load_yaml_mapping(args.config)
    run(config, args.device, args.stop_after)


if __name__ == "__main__":
    main()
