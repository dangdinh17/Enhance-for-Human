"""Resource-bound subprocess execution tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from researchctl.executor import execute, task_temp_directory
from researchctl.journal import Journal


class ExecutorTest(unittest.TestCase):
    def test_short_temp_allows_unix_socket_with_long_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / ('experiment_' * 15) / 'long_trial_name' / 'output'
            context = {'project.root': str(root),
                       'trial.output_dir': str(output),
                       'paths.temp': str(root/'short')}
            ipc = task_temp_directory(context)
            self.assertEqual(ipc.parent, root/'short')
            self.assertNotEqual(ipc, output/'tmp')
            context['runtime.writable_roots'] = [str(output), str(ipc)]
            journal = Journal(root/'journal.sqlite')
            journal.ensure_node({'node_id': 'ipc', 'campaign_id': 'campaign',
                                 'campaign_hash': 'hash', 'trial_id': 'trial',
                                 'stage': 'test'})
            try:
                result = execute(
                    node_id='ipc',
                    task={'argv': [sys.executable, '-c',
                          'from multiprocessing.connection import Listener; '
                          'listener=Listener(family="AF_UNIX"); '
                          'print(listener.address); listener.close()'],
                          'timeout_seconds': 10, 'network': 'deny',
                          'resources': {'gpu_count': 0}},
                    context=context, environment={}, journal=journal,
                    limits={'max_node_gb': 1}, lock_root=root/'locks',
                    safety={'filesystem': {}},
                )
                self.assertEqual(result.state, 'COMPLETE',
                                 (output/'stderr.log').read_text())
                address = (output/'stdout.log').read_text().strip()
                self.assertLess(len(address.encode()), 108)
                self.assertTrue(address.startswith(str(ipc)))
            finally:
                journal.close()

    def _run(self, task: dict, node_id: str) -> str:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            journal = Journal(root / "journal.sqlite")
            journal.ensure_node(
                {
                    "node_id": node_id,
                    "campaign_id": "campaign",
                    "campaign_hash": "hash",
                    "trial_id": "trial",
                    "stage": "test",
                }
            )
            result = execute(
                node_id=node_id,
                task=task,
                context={
                    "project.root": str(root),
                    "trial.output_dir": str(output),
                },
                environment={},
                journal=journal,
                limits={"max_node_gb": 1, "rlimits": {"max_processes": 32}},
                lock_root=root / "locks",
            )
            journal.close()
            return result.state

    def test_timeout_terminates_process_group(self) -> None:
        state = self._run(
            {
                "argv": [sys.executable, "-c", "import time; time.sleep(5)"],
                "timeout_seconds": 1,
                "resources": {"gpu_count": 0},
            },
            "timeout",
        )
        self.assertEqual(state, "TIMED_OUT")

    def test_output_quota_terminates_process(self) -> None:
        state = self._run(
            {
                "argv": [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; import time; "
                    "Path(r'{trial.output_dir}/large.bin').write_bytes(b'x' * 4096); "
                    "time.sleep(5)",
                ],
                "timeout_seconds": 10,
                "max_output_gb": 0.0000001,
                "resources": {"gpu_count": 0},
            },
            "quota",
        )
        self.assertEqual(state, "QUOTA_EXCEEDED")


if __name__ == "__main__":
    unittest.main()
