"""Strip notebook Git noise, or validate the exact blobs in the Git index."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import nbformat
from nbformat.validator import iter_validate
from nbstripout import strip_output


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def read_notebook(data):
    # Do not use nbformat.read: it can repair IDs and normalize the input.
    notebook = json.loads(
        data, object_pairs_hook=unique_object, parse_constant=invalid_constant
    )
    if not isinstance(notebook, dict) or notebook.get("nbformat") != 4:
        raise ValueError("expected an nbformat 4 notebook; convert older formats explicitly")
    minor = notebook.get("nbformat_minor")
    if type(minor) is not int or not 0 <= minor <= nbformat.v4.nbformat_minor:
        raise ValueError("unsupported notebook minor version")
    error = next(iter_validate(notebook, version=4, version_minor=minor), None)
    if error is not None:
        raise ValueError(f"invalid notebook structure: {error.message}")
    ids = [cell["id"] for cell in notebook["cells"] if "id" in cell]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate cell IDs")
    return notebook


def check_notebook(notebook):
    if notebook["metadata"] != {}:
        raise ValueError("notebook metadata must be empty")
    for index, cell in enumerate(notebook["cells"]):
        if cell["metadata"] != {}:
            raise ValueError(f"cell {index}: metadata must be empty")
        if cell["cell_type"] == "code":
            if cell["outputs"] != []:
                raise ValueError(f"cell {index}: outputs must be empty")
            if cell["execution_count"] is not None:
                raise ValueError(f"cell {index}: execution_count must be null")


def clean(data):
    notebook = nbformat.from_dict(read_notebook(data))
    # Clear all keys BEFORE nbstripout sees keep_output, tags or init_cell.
    notebook.metadata = {}
    for cell in notebook.cells:
        cell.metadata = {}
    strip_output(
        notebook, keep_output=False, keep_count=False, keep_id=True,
        strip_init_cells=True,
    )
    check_notebook(notebook)
    # Preserve the original source representation, IDs, attachments and version.
    return (json.dumps(notebook, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8")


def git(*args):
    return subprocess.check_output(["git", *args])


def install():
    """Configure this clone using the interpreter that ran the installer."""
    hook_config = subprocess.run(
        ["git", "config", "--get", "core.hooksPath"],
        stdout=subprocess.PIPE, check=False,
    )
    if hook_config.returncode not in (0, 1):
        raise ValueError("could not read core.hooksPath")
    hooks_path = os.fsdecode(hook_config.stdout).strip()
    if hooks_path and hooks_path != ".githooks":
        raise ValueError(
            f"core.hooksPath is already {hooks_path!r}; integrate the notebook "
            "check into that hook before switching to .githooks"
        )
    if not hooks_path:
        default_hook = Path(os.fsdecode(git("rev-parse", "--git-path", "hooks/pre-commit")).strip())
        if default_hook.is_file() and os.access(default_hook, os.X_OK):
            raise ValueError("an active pre-commit hook exists; integrate it into .githooks first")

    # Do not resolve symlinks: a venv's Python can point to the base interpreter.
    python = os.path.abspath(sys.executable)
    command = shlex.join([python, "scripts/notebook_clean.py", "filter"])
    settings = {
        "filter.nbstripout.clean": command,
        "filter.nbstripout.smudge": "cat",
        "filter.nbstripout.required": "true",
        "notebook-clean.python": python,
        "core.hooksPath": ".githooks",
    }
    for key, value in settings.items():
        git("config", "--local", key, value)
    print(f"Installed notebook filter and pre-commit hook using {python}.")
    print("Re-run install after moving the clone or changing its Python environment.")


def tracked_notebooks():
    for entry in git("ls-files", "--stage", "-z").split(b"\0"):
        if not entry:
            continue
        info, name = entry.split(b"\t", 1)
        path = os.fsdecode(name)
        if not path.lower().endswith(".ipynb"):
            continue
        mode, oid, stage = info.decode("ascii").split()
        if stage != "0" or mode not in ("100644", "100755"):
            raise ValueError(f"{path!r}: notebook must be a regular, conflict-free file")
        yield path, oid


def refresh_status(paths):
    """Refresh file stats without changing any staged content or index flags."""
    index = Path(os.fsdecode(git("rev-parse", "--git-path", "index")).rstrip("\n")).absolute()
    if not index.exists() or not paths:
        return 0
    lock_path = Path(str(index) + ".lock")
    try:
        lock = lock_path.open("xb")
    except FileExistsError:
        print("Git index is busy; refresh skipped. Retry after the Git operation finishes.")
        return 0
    owns_lock = True
    try:
        with lock, tempfile.TemporaryDirectory(prefix="notebook-refresh-", dir=index.parent) as folder:
            original = index.read_bytes()
            temporary_index = Path(folder) / "index"
            temporary_index.write_bytes(original)
            env = dict(os.environ, GIT_INDEX_FILE=str(temporary_index), GIT_OPTIONAL_LOCKS="0")

            def temporary_git(*args):
                # User hooks must not run against our private index. Keep flags
                # unchanged even if core.ignorestat is enabled in another clone.
                return subprocess.check_output([
                    "git", "-c", f"core.hooksPath={os.devnull}",
                    "-c", "core.fsmonitor=false", "-c", "core.ignorestat=false",
                    "--literal-pathspecs", *args,
                ], env=env)

            before = temporary_git("ls-files", "--stage", "-v", "-z")
            candidates = []
            for path in paths:
                file = Path(path)
                if not file.is_file() or file.is_symlink():
                    continue
                # Compare with the index, not HEAD: retain partial staging.
                staged = temporary_git("show", ":" + path)
                if clean(file.read_bytes()) == staged:
                    candidates.append(path)
            if not candidates:
                return 0
            temporary_git("add", "--", *candidates)
            if temporary_git("ls-files", "--stage", "-v", "-z") != before:
                # A source/mode change during the comparison must never get staged.
                print("Notebook content changed during refresh; Git index left untouched.")
                return 0
            if index.read_bytes() != original:
                raise ValueError("Git index changed during refresh; retry")
            lock.write(temporary_index.read_bytes())
            lock.flush()
            os.fsync(lock.fileno())
        os.replace(lock_path, index)
        owns_lock = False
        return len(candidates)
    finally:
        if owns_lock:
            lock_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("install", "filter", "strip", "check", "refresh"))
    parser.add_argument("--path", action="append", default=[], help="Only refresh this notebook (repeatable)")
    args = parser.parse_args()
    requested = {Path(path).absolute() for path in args.path}
    if requested and args.command != "refresh":
        parser.error("--path is only supported with refresh")
    if args.command == "filter":
        sys.stdout.buffer.write(clean(sys.stdin.buffer.read()))
        return

    os.chdir(os.fsdecode(git("rev-parse", "--show-toplevel").rstrip(b"\n")))
    if args.command == "install":
        install()
        return
    if args.command == "refresh":
        paths = [path for path, _ in tracked_notebooks() if not requested or Path(path).absolute() in requested]
        count = refresh_status(paths)
        print(f"Refreshed {count} notebooks; staged content and local outputs are unchanged.")
        return
    failures = 0
    count = 0
    for path, oid in tracked_notebooks():
        count += 1
        try:
            if args.command == "check":
                # Blob IDs read the staged snapshot, including partially staged files.
                check_notebook(read_notebook(git("cat-file", "blob", oid)))
            else:
                file = Path(path)
                if file.is_symlink():
                    raise ValueError("refusing to strip a symlink")
                original = file.read_bytes()
                cleaned = clean(original)
                if original != cleaned:
                    file.write_bytes(cleaned)
        except (ValueError, OSError) as error:
            print(f"{path!r}: {error}", file=sys.stderr)
            failures += 1
    if failures:
        raise ValueError(
            "clean the affected notebooks and stage them again; "
            "see README.md for setup and repair commands"
        )
    print(f"{'Checked staged' if args.command == 'check' else 'Stripped'} {count} notebooks.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
