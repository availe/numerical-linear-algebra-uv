"""Exercise cleanup and the installed Git filter/hook without editing notebooks."""

import copy
from contextlib import chdir
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

from scripts.notebook_clean import clean, read_notebook, refresh_status


ROOT = Path(__file__).resolve().parents[1]


def executed_notebook() -> dict[str, Any]:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": [
            {
                "cell_type": "code",
                "id": "code-cell",
                "source": ["print('hello')\n", "raise ValueError('failed')"],
                "metadata": {"keep_output": True, "tags": ["keep_output"], "init_cell": True},
                "execution_count": 3,
                "outputs": [
                    {"output_type": "stream", "name": "stdout", "text": "hello\n"},
                    {"output_type": "error", "ename": "ValueError", "evalue": "failed", "traceback": ["failed"]},
                ],
            },
            {
                "cell_type": "markdown",
                "id": "markdown-cell",
                "source": "Résumé ![image](attachment:figure.png)",
                "metadata": {"arbitrary": "noise"},
                "attachments": {"figure.png": {"image/png": "aGVsbG8="}},
            },
        ],
    }


class CleanerTests(unittest.TestCase):
    def test_cleans_success_and_error_outputs_without_changing_content(self):
        original = executed_notebook()
        expected = copy.deepcopy(original)
        expected["metadata"] = {}
        for cell in expected["cells"]:
            cell["metadata"] = {}
            if cell["cell_type"] == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
        result = clean(json.dumps(original).encode())
        self.assertEqual(json.loads(result), expected)
        self.assertEqual(clean(result), result)

    def test_does_not_repair_invalid_notebooks(self):
        notebook = executed_notebook()
        del notebook["cells"][0]["id"]
        with self.assertRaises(ValueError):
            read_notebook(json.dumps(notebook))


class GitIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="notebook git test ")
        self.addCleanup(self.directory.cleanup)
        self.repo = Path(self.directory.name)
        self.env = os.environ.copy()
        # Isolate tests from the user's Git config, index and shell PATH.
        for key in list(self.env):
            if key.startswith("GIT_"):
                del self.env[key]
        self.env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1", PATH=os.defpath)
        self.run_command("git", "init", "-q")
        for name in ("scripts/notebook_clean.py", ".githooks/pre-commit", ".gitattributes"):
            target = self.repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)

    def run_command(self, *args, data=None, check=True):
        result = subprocess.run(
            args, input=data, cwd=self.repo, env=self.env, capture_output=True, check=False,
        )
        if check and result.returncode:
            self.fail(f"{args!r} failed:\n{result.stderr.decode()}")
        return result

    def install(self, **kwargs):
        return self.run_command(sys.executable, "scripts/notebook_clean.py", "install", **kwargs)

    def stage_blob(self, *, filtered):
        args = ["git", "hash-object", "-w", "--stdin"]
        if filtered:
            args.append("--path=Executed notebook.IPYNB")
        oid = self.run_command(*args, data=json.dumps(executed_notebook()).encode()).stdout.strip().decode()
        self.run_command("git", "update-index", "--add", "--cacheinfo", "100644", oid, "Executed notebook.IPYNB")
        return self.run_command("git", "cat-file", "blob", oid).stdout

    def test_install_is_repeatable_and_filters_staged_content_without_uv_on_path(self):
        self.install()
        self.install()
        staged = self.stage_blob(filtered=True)
        self.assertEqual(staged, clean(json.dumps(executed_notebook()).encode()))
        self.assertEqual(self.run_command("git", "config", "--get", "core.hooksPath").stdout.strip(), b".githooks")
        self.run_command("git", "hook", "run", "pre-commit")

    def test_hook_rejects_output_that_bypassed_filter(self):
        self.install()
        self.stage_blob(filtered=False)
        result = self.run_command("git", "hook", "run", "pre-commit", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"metadata must be empty", result.stderr)

    def test_required_filter_rejects_invalid_input(self):
        self.install()
        result = self.run_command(
            "git", "hash-object", "--stdin", "--path=broken.ipynb", data=b"not JSON", check=False,
        )
        self.assertNotEqual(result.returncode, 0)

    def test_preserves_custom_hook_configuration(self):
        self.run_command("git", "config", "core.hooksPath", "custom-hooks")
        result = self.install(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.run_command("git", "config", "--get", "core.hooksPath").stdout.strip(), b"custom-hooks")
        self.assertEqual(self.run_command("git", "config", "--get", "filter.nbstripout.clean", check=False).returncode, 1)

    def refresh_fixture(self):
        # Exercise real Git filtering with a JSON fixture, without writing .ipynb files.
        self.install()
        with (self.repo / ".gitattributes").open("a") as attributes:
            attributes.write("\n*.data filter=nbstripout\n")
        file = self.repo / "Executed notebook.data"
        file.write_bytes(clean(json.dumps(executed_notebook()).encode()))
        self.run_command("git", "add", "--", file.name)
        file.write_text(json.dumps(executed_notebook()))
        return file

    def refresh(self, file):
        with chdir(self.repo), patch.dict(os.environ, self.env, clear=True):
            return refresh_status([file.name])

    def test_refresh_clears_output_only_status_and_preserves_local_outputs(self):
        file = self.refresh_fixture()
        staged = self.run_command("git", "ls-files", "--stage", "-v").stdout
        original = file.read_bytes()
        self.assertTrue(self.run_command("git", "status", "--porcelain", "--", file.name).stdout.startswith(b"AM "))
        self.assertEqual(self.refresh(file), 1)
        self.assertEqual(self.run_command("git", "ls-files", "--stage", "-v").stdout, staged)
        self.assertTrue(self.run_command("git", "status", "--porcelain", "--", file.name).stdout.startswith(b"A  "))
        self.assertEqual(file.read_bytes(), original)

    def test_refresh_preserves_partial_staging_and_unstaged_source_edits(self):
        file = self.refresh_fixture()
        notebook = executed_notebook()
        notebook["cells"][0]["source"] = "print('staged edit')"
        file.write_text(json.dumps(notebook))
        self.run_command("git", "add", "--", file.name)
        original_index = (self.repo / ".git/index").read_bytes()
        notebook["cells"][0]["source"] = "print('unstaged edit')"
        file.write_text(json.dumps(notebook))
        self.assertEqual(self.refresh(file), 0)
        self.assertEqual((self.repo / ".git/index").read_bytes(), original_index)
        self.assertTrue(self.run_command("git", "status", "--porcelain", "--", file.name).stdout.startswith(b"AM "))

    def test_refresh_does_not_stage_source_edits_made_during_refresh(self):
        file = self.refresh_fixture()
        original_index = (self.repo / ".git/index").read_bytes()
        check_output = subprocess.check_output

        def edit_during_add(args, **kwargs):
            if "add" in args:
                notebook = executed_notebook()
                notebook["cells"][0]["source"] = "print('concurrent edit')"
                file.write_text(json.dumps(notebook))
            return check_output(args, **kwargs)

        with patch("scripts.notebook_clean.subprocess.check_output", side_effect=edit_during_add):
            self.assertEqual(self.refresh(file), 0)
        self.assertEqual((self.repo / ".git/index").read_bytes(), original_index)
        self.assertFalse((self.repo / ".git/index.lock").exists())

    def test_refresh_respects_existing_index_lock(self):
        file = self.refresh_fixture()
        lock = self.repo / ".git/index.lock"
        lock.write_bytes(b"another Git operation")
        self.assertEqual(self.refresh(file), 0)
        self.assertEqual(lock.read_bytes(), b"another Git operation")

    def test_refresh_invalid_input_leaves_index_unchanged_and_unlocks(self):
        file = self.refresh_fixture()
        original_index = (self.repo / ".git/index").read_bytes()
        file.write_bytes(b"invalid JSON")
        with self.assertRaises(ValueError):
            self.refresh(file)
        self.assertEqual((self.repo / ".git/index").read_bytes(), original_index)
        self.assertFalse((self.repo / ".git/index.lock").exists())


if __name__ == "__main__":
    unittest.main()
