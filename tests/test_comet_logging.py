"""Exercise the real Comet SDK offline without publishing test measurements."""

import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.utils.experiment import (
    CometMetricLogger, canonical_config_hash, load_yaml_mapping,
)

ROOT = Path(__file__).resolve().parents[1]


class CometLoggingTests(unittest.TestCase):
    """Protect experiment identity, real metric output and missing-key behavior."""

    def config(self, qp: int = 37) -> dict:
        """Use the actual QP config with SDK network communication disabled."""
        config = load_yaml_mapping(ROOT / f'configs/kaggle/QP{qp}.yml')
        config['comet_logging']['online'] = False
        return config

    def test_offline_metrics_distinct_qps_and_resume_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            config = self.config()
            initial_hash = canonical_config_hash(config)
            logger = CometMetricLogger(config, directory / 'QP37')
            self.assertTrue(logger.enabled)
            key = logger.config['comet_logging']['experiment_key']
            row = json.loads((ROOT / 'tests/fixtures/qp37_recorded_validation.json')
                             .read_text())['rows'][0]
            logger.log_metrics('validation', {
                name: float(row[name]) for name in ('enhanced_psnr', 'total_loss')
            }, int(row['iteration']))
            logger.close()
            state = json.loads((directory / 'QP37/comet.json').read_text())
            self.assertEqual(state['status'], 'closed')
            self.assertEqual(state['project_name'], 'qe-for-human')
            self.assertEqual(state['workspace'], 'dangdinh17')
            self.assertEqual(state['experiment_key'], key)
            self.assertEqual(len(key), 32)
            self.assertNotIn('experiment_key', config['comet_logging'])
            self.assertEqual(canonical_config_hash(config), initial_hash)

            archives = list((directory / 'QP37/comet-offline').glob('*.zip'))
            self.assertTrue(archives)
            with zipfile.ZipFile(archives[0]) as archive:
                contents = '\n'.join(archive.read(name).decode('utf-8', errors='replace')
                                     for name in archive.namelist())
            self.assertIn('validation/enhanced_psnr', contents)
            self.assertIn('validation/total_loss', contents)

            resumed = CometMetricLogger(config, directory / 'QP37')
            self.assertTrue(resumed.enabled)
            self.assertEqual(resumed.config['comet_logging']['experiment_key'], key)
            resumed.close()
            other = CometMetricLogger(self.config(22), directory / 'QP22')
            self.assertTrue(other.enabled)
            self.assertNotEqual(other.config['comet_logging']['experiment_key'], key)
            other.close()

    def test_changed_project_does_not_overwrite_saved_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            config = self.config()
            logger = CometMetricLogger(config, directory)
            logger.close()
            before = (directory / 'comet.json').read_bytes()
            changed = copy.deepcopy(config)
            changed['comet_logging']['project_name'] = 'another-project'
            with self.assertRaisesRegex(ValueError, 'project_name differs'):
                CometMetricLogger(changed, directory)
            self.assertEqual(before, (directory / 'comet.json').read_bytes())

    def test_missing_api_key_records_failure_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = self.config()
            config['comet_logging'].update(
                online=True, api_key_env='NEWQE_TEST_UNCONFIGURED_COMET_API_KEY')
            logger = CometMetricLogger(config, Path(temp))
            self.assertFalse(logger.enabled)
            state = json.loads((Path(temp) / 'comet.json').read_text())
            self.assertEqual(state['status'], 'failed')
            self.assertIn('Kaggle Secrets', state['error'])


if __name__ == '__main__':
    unittest.main()
