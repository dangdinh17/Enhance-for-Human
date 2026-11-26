"""Regressions for the real Human architecture, loss and sampling protocol."""
from __future__ import annotations

import sys
import unittest

import torch

from src.data.human_data import manifest
from src.data.window_mixed import EpochWindowBatches
from src.loss.charbonnier import MFQEv2CombinedLoss
from src.models.mfqev2_window import MFQEv2Window


class HumanModelTests(unittest.TestCase):
    """Exercise architecture and loss without a feature extractor."""

    def test_original_architecture_identity_and_backward(self) -> None:
        torch.set_num_threads(2)
        model = MFQEv2Window()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 449253)
        window = torch.rand(1, 7, 1, 32, 32)
        target = window[:, 3] * .97
        output = model(window)
        self.assertEqual(tuple(output.shape), (1, 1, 32, 32))
        torch.testing.assert_close(output, window[:, 3], rtol=0, atol=0)
        criterion = MFQEv2CombinedLoss()
        self.assertEqual(list(criterion.parameters()), [])
        loss = criterion(output, target)
        expected = ((output.float() - target).square() + 1e-12).sqrt().mean()
        torch.testing.assert_close(loss.total, expected, rtol=0, atol=0)
        self.assertEqual(float(loss.feature), 0)
        loss.total.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
        self.assertFalse(any("jde" in name.lower() for name in sys.modules))

    def test_manifests_own_unique_samples_and_same_temporal_windows(self) -> None:
        reference = None
        for qp in (22, 27, 32, 37):
            _, value = manifest(qp)
            windows = [(r["sequence"], r["lq_keys"]) for r in value["splits"]["train"]]
            if reference is not None:
                self.assertEqual(windows, reference)
            reference = windows

    def test_epoch_shuffle_resume_and_dropped_tail(self) -> None:
        full = list(EpochWindowBatches(5407, 8, 29, 0, 677))
        resumed = list(EpochWindowBatches(5407, 8, 29, 674, 677))
        self.assertEqual(resumed, full[674:])
        # Every epoch has 675 full batches; seven samples are dropped.
        first = [item[1] for batch in full[:675] for item in batch]
        self.assertEqual(len(first), 5400)
        self.assertEqual(len(set(first)), 5400)


if __name__ == "__main__":
    unittest.main()
