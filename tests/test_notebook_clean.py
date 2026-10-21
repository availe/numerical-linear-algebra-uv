"""Exercise cleanup and the installed Git filter/hook without editing notebooks."""

import copy
from contextlib import chdir
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any
import unittest
from unittest.mock import patch
from xml.etree import ElementTree

from scripts.notebook_clean import check_notebook, clean, main, read_notebook, refresh_status


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
                    {
                        "output_type": "execute_result",
                        "execution_count": 3,
                        "metadata": {"execution": {"timestamp": "noise"}},
                        "data": {"text/plain": ["array([1., 2.])"], "text/html": "<table><tr><td>1</td></tr></table>"},
                    },
                    {
                        "output_type": "display_data",
                        "metadata": {"arbitrary": "noise"},
                        "data": {"image/png": "aGVsbG8=", "text/plain": "<Figure>"},
                    },
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

    def test_homework_keeps_results_and_images_but_removes_metadata_and_counts(self):
        original = executed_notebook()
        expected = copy.deepcopy(original)
        expected["metadata"] = {}
        for cell in expected["cells"]:
            cell["metadata"] = {}
        code = expected["cells"][0]
        code["execution_count"] = None
        code["outputs"][2]["metadata"] = {}
        code["outputs"][2]["execution_count"] = None
        code["outputs"][3]["metadata"] = {}

        result = clean(json.dumps(original).encode(), path="hw/3.ipynb")
        self.assertEqual(read_notebook(result), expected)
        self.assertEqual(clean(result, path="hw/3.ipynb"), result)
        check_notebook(read_notebook(result), path="hw/3.ipynb")
        with self.assertRaisesRegex(ValueError, "outputs must be empty"):
            check_notebook(read_notebook(result), path="nbs/3.ipynb")

    def test_only_root_homework_folder_keeps_outputs(self):
        data = json.dumps(executed_notebook()).encode()
        for path, expected in (
            ("hw/3.ipynb", True), ("hw/nested/3.IPYNB", True),
            ("./hw/3.ipynb", True), (None, False), ("3.ipynb", False),
            ("nbs/3.ipynb", False), ("hw-old/3.ipynb", False),
            ("nbs/hw/3.ipynb", False),
        ):
            with self.subTest(path=path):
                result = read_notebook(clean(data, path=path))
                self.assertEqual(bool(result["cells"][0]["outputs"]), expected)


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
        for name in (
            "scripts/notebook_clean.py", "scripts/pycharm-notebook-watcher.xml",
            ".githooks/pre-commit", ".gitattributes", ".gitignore",
            ".idea/watcherTasks.xml", ".idea/externalDependencies.xml",
        ):
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

    def stage_blob(self, *, filtered, path="Executed notebook.IPYNB", notebook=None):
        args = ["git", "hash-object", "-w", "--stdin"]
        if filtered:
            args.append("--path=" + path)
        data = json.dumps(executed_notebook() if notebook is None else notebook).encode()
        oid = self.run_command(*args, data=data).stdout.strip().decode()
        self.run_command("git", "update-index", "--add", "--cacheinfo", "100644", oid, path)
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

    def test_git_filter_and_hook_keep_homework_outputs_with_quoted_paths(self):
        self.install()
        path = "hw/nested/Executed 'notebook' $result.IPYNB"
        staged = self.stage_blob(filtered=True, path=path)
        self.assertEqual(staged, clean(json.dumps(executed_notebook()).encode(), path=path))
        self.assertEqual(len(read_notebook(staged)["cells"][0]["outputs"]), 4)
        self.run_command("git", "hook", "run", "pre-commit")

    def test_hook_rejects_homework_metadata_and_execution_counts(self):
        self.install()
        path = "hw/3.ipynb"
        cleaned = clean(json.dumps(executed_notebook()).encode(), path=path)
        for location, key, value, message in (
            ("notebook", "metadata", {"noise": True}, b"metadata must be empty"),
            ("cell", "metadata", {"noise": True}, b"metadata must be empty"),
            ("cell", "execution_count", 42, b"execution_count must be null"),
            ("result", "metadata", {"noise": True}, b"metadata must be empty"),
            ("result", "execution_count", 42, b"execution_count must be null"),
            ("display", "metadata", {"noise": True}, b"metadata must be empty"),
        ):
            with self.subTest(location=location, key=key):
                notebook = read_notebook(cleaned)
                cell = notebook["cells"][0]
                target = {
                    "notebook": notebook, "cell": cell,
                    "result": cell["outputs"][2], "display": cell["outputs"][3],
                }[location]
                target[key] = value
                self.stage_blob(filtered=False, path=path, notebook=notebook)
                result = self.run_command("git", "hook", "run", "pre-commit", check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

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

    def test_shared_watcher_bootstraps_a_fresh_clone_without_manual_install(self):
        # Commit only the portable setup, then exercise the exact watcher command
        # in a real clone with no inherited Git settings and a path with spaces.
        self.run_command("git", "add", ".")
        self.run_command(
            "git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "commit", "-qm", "Shared notebook setup",
        )
        clone = self.repo / "fresh clone"
        self.run_command("git", "clone", "-q", "--no-local", str(self.repo), str(clone))
        self.repo = clone
        self.assertEqual(self.run_command("git", "config", "--get", "filter.nbstripout.clean", check=False).returncode, 1)
        self.assertEqual(
            self.run_command("git", "ls-files", ".idea").stdout.splitlines(),
            [b".idea/externalDependencies.xml", b".idea/watcherTasks.xml"],
        )
        self.assertEqual(self.run_command("git", "check-ignore", ".idea/workspace.xml").returncode, 0)

        # Reuse the test environment's dependencies instead of downloading the
        # course's large packages. No .ipynb working file is created or changed.
        (clone / ".venv").symlink_to(sys.prefix, target_is_directory=True)
        watcher = ElementTree.parse(clone / ".idea/watcherTasks.xml").find("component/TaskOptions")
        assert watcher is not None
        self.assertEqual(watcher.attrib["isEnabled"], "true")
        template = ElementTree.parse(clone / "scripts/pycharm-notebook-watcher.xml").find("TaskOptions")
        assert template is not None
        options = {item.attrib["name"]: item.attrib["value"] for item in watcher.findall("option")}
        self.assertEqual(options, {item.attrib["name"]: item.attrib["value"] for item in template.findall("option")})
        self.assertEqual(options["fileExtension"], "ipynb")

        def expand(value):
            return value.replace("$ProjectFileDir$", str(clone)).replace("$FilePath$", str(clone / "Notebook with spaces.ipynb"))

        command = [expand(options["program"]), *shlex.split(expand(options["arguments"]))]
        before_index = self.run_command("git", "ls-files", "--stage", "-v").stdout
        result = self.run_command(*command)
        self.assertIn(b"Installed notebook filter", result.stdout)
        self.assertEqual(self.run_command("git", "ls-files", "--stage", "-v").stdout, before_index)
        self.assertEqual(self.run_command("git", "config", "--get", "notebook-clean.python").stdout.strip(), os.fsencode(command[0]))
        config = clone / ".git/config"
        config_before = (config.read_bytes(), config.stat().st_mtime_ns)
        self.assertNotIn(b"Installed", self.run_command(*command).stdout)
        self.assertEqual((config.read_bytes(), config.stat().st_mtime_ns), config_before)

        staged = self.stage_blob(filtered=True)
        self.assertEqual(staged, clean(json.dumps(executed_notebook()).encode()))
        self.run_command("git", "hook", "run", "pre-commit")

    def test_refresh_repairs_stale_interpreter_paths_and_missing_settings(self):
        self.install()
        self.run_command("git", "config", "filter.nbstripout.clean", "/old/clone/.venv/bin/python scripts/notebook_clean.py filter")
        self.run_command("git", "config", "notebook-clean.python", "/old/clone/.venv/bin/python")
        self.run_command("git", "config", "--unset", "filter.nbstripout.required")
        self.run_command(sys.executable, "scripts/notebook_clean.py", "refresh")
        self.assertEqual(self.run_command("git", "config", "--get", "notebook-clean.python").stdout.strip(), os.fsencode(sys.executable))
        staged = self.stage_blob(filtered=True)
        self.assertEqual(staged, clean(json.dumps(executed_notebook()).encode()))
        self.run_command("git", "hook", "run", "pre-commit")

    def test_refresh_upgrades_old_filter_to_pass_notebook_paths(self):
        self.install()
        old_filter = shlex.join([sys.executable, "scripts/notebook_clean.py", "filter"])
        self.run_command("git", "config", "filter.nbstripout.clean", old_filter)
        result = self.run_command(sys.executable, "scripts/notebook_clean.py", "refresh")
        self.assertIn(b"Installed notebook filter", result.stdout)
        staged = self.stage_blob(filtered=True, path="hw/3.ipynb")
        self.assertEqual(len(read_notebook(staged)["cells"][0]["outputs"]), 4)
        self.run_command("git", "hook", "run", "pre-commit")

    def test_automatic_install_preserves_custom_hooks(self):
        self.run_command("git", "config", "core.hooksPath", "custom-hooks")
        config_before = (self.repo / ".git/config").read_bytes()
        result = self.run_command(sys.executable, "scripts/notebook_clean.py", "refresh", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"integrate the notebook check", result.stderr)
        self.assertEqual((self.repo / ".git/config").read_bytes(), config_before)

    def test_automatic_install_preserves_active_default_pre_commit(self):
        hook = self.repo / ".git/hooks/pre-commit"
        hook.write_text("#!/bin/sh\nexit 0\n")
        hook.chmod(0o755)
        config_before = (self.repo / ".git/config").read_bytes()
        result = self.run_command(sys.executable, "scripts/notebook_clean.py", "refresh", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"an active pre-commit hook exists", result.stderr)
        self.assertEqual((self.repo / ".git/config").read_bytes(), config_before)

    def refresh_fixture(self, path="Executed notebook.data"):
        # Exercise real Git filtering with a JSON fixture, without writing .ipynb files.
        self.install()
        with (self.repo / ".gitattributes").open("a") as attributes:
            attributes.write("\n*.data filter=nbstripout\n")
        file = self.repo / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(clean(json.dumps(executed_notebook()).encode(), path=path))
        self.run_command("git", "add", "--", path)
        file.write_text(json.dumps(executed_notebook()))
        return file

    def refresh(self, file):
        with chdir(self.repo), patch.dict(os.environ, self.env, clear=True):
            return refresh_status([file.relative_to(self.repo).as_posix()])

    def test_homework_refresh_clears_metadata_only_status_and_keeps_local_outputs(self):
        path = "hw/nested/Executed notebook.data"
        file = self.refresh_fixture(path)
        staged = self.run_command("git", "ls-files", "--stage", "-v").stdout
        original = file.read_bytes()
        self.assertTrue(self.run_command("git", "status", "--porcelain", "--", path).stdout.startswith(b"AM "))
        self.assertEqual(self.refresh(file), 1)
        self.assertEqual(self.run_command("git", "ls-files", "--stage", "-v").stdout, staged)
        self.assertTrue(self.run_command("git", "status", "--porcelain", "--", path).stdout.startswith(b"A  "))
        self.assertEqual(file.read_bytes(), original)

    def test_homework_refresh_preserves_unstaged_output_changes(self):
        path = "hw/Executed notebook.data"
        file = self.refresh_fixture(path)
        notebook = executed_notebook()
        notebook["cells"][0]["outputs"][0]["text"] = "a different result\n"
        file.write_text(json.dumps(notebook))
        original_index = (self.repo / ".git/index").read_bytes()
        self.assertEqual(self.refresh(file), 0)
        self.assertEqual((self.repo / ".git/index").read_bytes(), original_index)
        self.assertTrue(self.run_command("git", "status", "--porcelain", "--", path).stdout.startswith(b"AM "))

    def test_strip_keeps_homework_results_and_cleans_metadata(self):
        path = "hw/Executed notebook.data"
        file = self.refresh_fixture(path)
        with (
            chdir(self.repo), patch.dict(os.environ, self.env, clear=True),
            patch("scripts.notebook_clean.tracked_notebooks", return_value=[(path, "unused")]),
            patch.object(sys, "argv", ["notebook_clean.py", "strip"]),
        ):
            main()
        result = read_notebook(file.read_bytes())
        self.assertEqual(len(result["cells"][0]["outputs"]), 4)
        check_notebook(result, path=path)

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
