"""Exercise real GradScaler overflow, retry and fail-closed behavior on CPU."""

import unittest
import warnings

import torch
from torch import nn

from src.loss.charbonnier import LossComponents
from src.train.window_trainer import WindowTrainer


class HalfPrecisionHead(nn.Module):
    """Small differentiable FP16 boundary exposing scaled-gradient overflow."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        return (window[:, 3] * self.weight).to(torch.float16)


class MeanLoss(nn.Module):
    """Analytic mean reconstruction loss for exact optimizer-step assertions."""

    def forward(self, prediction: torch.Tensor,
                target: torch.Tensor) -> LossComponents:
        value = (prediction.float() - target).mean()
        return LossComponents(value, value, value.new_zeros(()))


class WindowAMPRecoveryTests(unittest.TestCase):
    def trainer(self, scale: float, amp: bool = True) -> WindowTrainer:
        """Use a real CPU scaler, independent of GPU availability."""
        model = HalfPrecisionHead()
        trainer = WindowTrainer(model, torch.optim.Adam(model.parameters(), lr=1e-4),
                                MeanLoss(), 'cpu', False, 1.0)
        trainer.amp_enabled = amp
        trainer.scaler = torch.amp.GradScaler('cpu', init_scale=scale, enabled=amp)
        return trainer

    def batch(self) -> dict:
        return {'lq': torch.ones(1, 7, 1, 4, 4),
                'target': torch.zeros(1, 1, 4, 4)}

    def test_overflow_retries_same_batch_and_updates_adam_once(self) -> None:
        recovering = self.trainer(1048576)
        reference = self.trainer(524288)
        with self.assertWarnsRegex(RuntimeWarning, 'retrying same batch'):
            result = recovering.train_batch(self.batch())
        reference.train_batch(self.batch())
        self.assertEqual(recovering.last_amp_retries, 1)
        self.assertEqual(recovering.scaler.get_scale(), 524288)
        self.assertEqual(result.samples, 1)
        self.assertEqual(recovering.last_gradient_norm, 1.0)
        torch.testing.assert_close(recovering.model.weight, reference.model.weight,
                                   rtol=0, atol=0)
        actual = recovering.optimizer.state[recovering.model.weight]
        expected = reference.optimizer.state[reference.model.weight]
        self.assertEqual(float(actual['step']), 1)
        for name in ('exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    def test_persistent_nonfinite_gradients_stop_without_weight_update(self) -> None:
        trainer = self.trainer(1048576)
        handle = trainer.model.weight.register_hook(
            lambda gradient: torch.full_like(gradient, float('inf')))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                with self.assertRaisesRegex(FloatingPointError, 'after 8 AMP retries'):
                    trainer.train_batch(self.batch())
            self.assertEqual(float(trainer.model.weight), 1)
            self.assertFalse(trainer.optimizer.state)
            self.assertIsNone(trainer.model.weight.grad)
        finally:
            handle.remove()

    def test_nonfinite_fp32_gradients_fail_immediately(self) -> None:
        trainer = self.trainer(1, amp=False)
        handle = trainer.model.weight.register_hook(
            lambda gradient: torch.full_like(gradient, float('nan')))
        try:
            with self.assertRaisesRegex(FloatingPointError, 'after 0 AMP retries'):
                trainer.train_batch(self.batch())
            self.assertFalse(trainer.optimizer.state)
        finally:
            handle.remove()

    def test_nonfinite_loss_is_not_hidden_as_amp_overflow(self) -> None:
        trainer = self.trainer(1048576)
        batch = self.batch()
        batch['target'].fill_(float('nan'))
        with self.assertRaisesRegex(FloatingPointError, 'training loss'):
            trainer.train_batch(batch)
        self.assertFalse(trainer.optimizer.state)


if __name__ == '__main__':
    unittest.main()
