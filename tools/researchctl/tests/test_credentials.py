"""Credential persistence tests use isolated account directories."""

from __future__ import annotations

import io
import json
import os
import stat
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from researchctl.cli import main
from researchctl.config import ConfigError, redact_text
from researchctl.credentials import (
    credential_path,
    load_credentials,
    read_credentials,
    save_comet_key,
)


class CredentialsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.account = Path(self.temporary.name)
        self.home = patch(
            "researchctl.credentials.Path.home", return_value=self.account
        )
        self.home.start()
        self.addCleanup(self.home.stop)

    def test_private_roundtrip_and_atomic_replacement(self) -> None:
        self.assertEqual(read_credentials(), {})
        path = save_comet_key("unit-test-token")
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)
        save_comet_key("replacement-token")
        self.assertEqual(read_credentials(), {"COMET_API_KEY": "replacement-token"})
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_environment_precedence_and_redaction(self) -> None:
        save_comet_key("stored-unit-token")
        with patch.dict(os.environ, {}, clear=True):
            load_credentials()
            self.assertEqual(os.environ["COMET_API_KEY"], "stored-unit-token")
            self.assertNotIn("stored-unit-token", redact_text("stored-unit-token"))
        with patch.dict(os.environ, {"COMET_API_KEY": "explicit-token"}, clear=True):
            load_credentials()
            self.assertEqual(os.environ["COMET_API_KEY"], "explicit-token")

    def test_rejects_readable_file_and_shared_directory(self) -> None:
        path = save_comet_key("unit-test-token")
        path.chmod(0o644)
        with self.assertRaises(ConfigError):
            read_credentials()
        path.chmod(0o600)
        path.parent.chmod(0o755)
        with self.assertRaises(ConfigError):
            read_credentials()

    def test_rejects_symlinks_without_modifying_target(self) -> None:
        path = save_comet_key("unit-test-token")
        target = self.account / "original.json"
        path.rename(target)
        path.symlink_to(target)
        with self.assertRaises(OSError):
            save_comet_key("replacement-token")
        self.assertEqual(
            json.loads(target.read_text())["COMET_API_KEY"], "unit-test-token"
        )
        path.unlink()
        path.parent.rmdir()
        path.parent.symlink_to(self.account, target_is_directory=True)
        with self.assertRaises(ConfigError):
            read_credentials()

    def test_invalid_values_never_appear_in_errors(self) -> None:
        for value in ("", "private\ninvalid", "private token", 123, "a" * 513):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(ConfigError) as raised:
                    save_comet_key(value)
                if value:
                    self.assertNotIn(str(value), str(raised.exception))
        self.assertFalse(credential_path().exists())

    def test_cli_status_works_outside_project_and_never_prints_key(self) -> None:
        save_comet_key("unit-test-private-token")
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["credentials", "status"]), 0)
        self.assertTrue(json.loads(output.getvalue())["comet_stored"])
        self.assertNotIn("unit-test-private-token", output.getvalue())

    def test_rejects_unknown_fields_and_oversized_store(self) -> None:
        path = save_comet_key("unit-test-token")
        path.write_text(json.dumps({"UNSUPPORTED_TOKEN": "value"}))
        with self.assertRaises(ConfigError):
            read_credentials()
        path.write_text("x" * 4097)
        with self.assertRaises(ConfigError):
            read_credentials()


if __name__ == "__main__":
    unittest.main()
