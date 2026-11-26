"""Discover attached dataset paths using the real published bundle receipt."""

import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.bootstrap_kaggle import discover_dataset_root

RECEIPT = Path(__file__).parent / 'fixtures/training_bundle_receipt.json'


class KaggleDiscoveryTests(unittest.TestCase):
    """Cover nested mounts, missing input and explicit path selection."""

    def attach(self, root: Path, relative: str) -> Path:
        """Place the actual receipt at a different filesystem mount layout."""
        folder = root / relative
        folder.mkdir(parents=True)
        shutil.copyfile(RECEIPT, folder / 'bundle.json')
        return folder.resolve()

    def test_flat_and_nested_mounts(self) -> None:
        for relative in (
            'enhance-for-human-mfqe-vvc-trainval-qp22-37',
            'datasets/anginhok/enhance-for-human-mfqe-vvc-trainval-qp22-37/versions/1',
        ):
            with self.subTest(layout=relative), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                attached = self.attach(root, relative)
                self.assertEqual(discover_dataset_root(input_root=root), attached)

    def test_missing_dataset_explains_add_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError, 'Add Input'):
                discover_dataset_root(input_root=Path(temp))

    def test_multiple_mounts_require_explicit_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self.attach(root, 'first')
            self.attach(root, 'second')
            with self.assertRaisesRegex(ValueError, '--dataset-root'):
                discover_dataset_root(input_root=root)
            self.assertEqual(discover_dataset_root(first, input_root=root), first)

    def test_explicit_path_does_not_silently_fall_back(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.attach(root, 'attached')
            with self.assertRaisesRegex(FileNotFoundError, '--dataset-root'):
                discover_dataset_root(root / 'wrong', input_root=root)


if __name__ == '__main__':
    unittest.main()
