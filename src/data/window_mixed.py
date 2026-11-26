"""Read-only mixed-QP windows with restartable, step-indexed augmentation."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

import lmdb
from torch.utils.data import Dataset, Sampler

from .stdr_mfqev2_lmdb import STDRMFQEv2LMDBDataset


_ENVIRONMENTS: dict[tuple[int, str], lmdb.Environment] = {}


class SharedWindowDataset(STDRMFQEv2LMDBDataset):
    """Share identical GT LMDB handles across QPs inside each worker process."""

    def _environment(self, client: Literal["gt", "lq"]) -> lmdb.Environment:
        root = self.gt_root if client == "gt" else self.lq_root
        key = (os.getpid(), str(root.resolve()))
        if key not in _ENVIRONMENTS:
            if not root.is_dir():
                raise FileNotFoundError(root)
            _ENVIRONMENTS[key] = lmdb.open(
                str(root), readonly=True, lock=False, readahead=False,
                meminit=False, max_readers=512,
            )
        return _ENVIRONMENTS[key]


def close_window_environments() -> None:
    """Close pooled handles owned by this process after all readers finish."""
    for key in list(_ENVIRONMENTS):
        if key[0] == os.getpid():
            _ENVIRONMENTS.pop(key).close()


class MixedQPWindows(Dataset):
    """Dispatch (QP-index, sample-index, augmentation-seed) to adjacent windows."""

    def __init__(self, manifests: Sequence[str | Path], crop_size: int) -> None:
        self.datasets = [SharedWindowDataset(path, "train", crop_size, True)
                         for path in manifests]
        self.qps = [dataset.qp for dataset in self.datasets]
        if len(set(self.qps)) != len(self.qps):
            raise ValueError("Mixed-QP manifests must have unique QPs")

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, index: tuple[int, int, int]) -> dict[str, Any]:
        qp_index, sample_index, augmentation_seed = index
        state = random.getstate()
        try:
            random.seed(augmentation_seed)
            return self.datasets[qp_index][sample_index]
        finally:
            random.setstate(state)


class BalancedWindowBatches(Sampler):
    """Exactly equal samples/QP; worker prefetch cannot alter the resume stream."""

    def __init__(self, lengths: Sequence[int], batch_size: int, seed: int,
                 start: int, stop: int) -> None:
        if not lengths or min(lengths) < 1:
            raise ValueError("Each QP needs nonempty training samples")
        if batch_size < len(lengths) or batch_size % len(lengths):
            raise ValueError("batch_size must be a positive multiple of QP count")
        if start < 0 or stop < start:
            raise ValueError("Invalid iteration interval")
        self.lengths = list(lengths)
        self.per_qp = batch_size // len(lengths)
        self.seed, self.start, self.stop = seed, start, stop

    def __len__(self) -> int:
        return self.stop - self.start

    def __iter__(self) -> Iterator[list[tuple[int, int, int]]]:
        for step in range(self.start, self.stop):
            rng = random.Random(self.seed + 1000003 * step)
            batch = [(qp, rng.randrange(length), rng.randrange(2 ** 63))
                     for qp, length in enumerate(self.lengths)
                     for _ in range(self.per_qp)]
            rng.shuffle(batch)
            yield batch


class EpochWindowBatches(Sampler):
    """Shuffle a single QP without replacement, dropping incomplete batches.

    Epoch permutations and per-sample augmentation seeds depend on the global
    iteration, so prefetch and worker scheduling cannot alter a resumed stream.
    The emitted tuple format is shared with ``MixedQPWindows``.
    """

    def __init__(self, length: int, batch_size: int, seed: int,
                 start: int, stop: int) -> None:
        if batch_size < 1 or length < batch_size or start < 0 or stop < start:
            raise ValueError("Invalid single-QP epoch sampler configuration")
        self.length = length
        self.batch_size = batch_size
        self.seed = seed
        self.start, self.stop = start, stop
        self.steps_per_epoch = length // batch_size

    def __len__(self) -> int:
        return self.stop - self.start

    def state_at(self, iteration: int) -> dict[str, Any]:
        """Describe the next batch, rather than the prefetched iterator cursor."""
        epoch, batch = divmod(iteration, self.steps_per_epoch)
        return {
            "kind": "epoch_shuffle", "length": self.length,
            "batch_size": self.batch_size, "seed": self.seed,
            "next_iteration": iteration, "next_epoch": epoch,
            "next_batch_in_epoch": batch,
        }

    def __iter__(self) -> Iterator[list[tuple[int, int, int]]]:
        previous_epoch = None
        order: list[int] = []
        for step in range(self.start, self.stop):
            epoch, batch = divmod(step, self.steps_per_epoch)
            if epoch != previous_epoch:
                order = list(range(self.length))
                random.Random(self.seed + 1000003 * epoch).shuffle(order)
                previous_epoch = epoch
            indices = order[batch * self.batch_size:(batch + 1) * self.batch_size]
            rng = random.Random(self.seed + 2000003 * step)
            yield [(0, index, rng.randrange(2 ** 63)) for index in indices]
