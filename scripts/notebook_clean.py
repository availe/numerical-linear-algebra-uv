"""Strip notebook Git noise, or validate the exact blobs in the Git index."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("filter", "strip", "check"))
    args = parser.parse_args()
    if args.command == "filter":
        sys.stdout.buffer.write(clean(sys.stdin.buffer.read()))
        return

    os.chdir(os.fsdecode(git("rev-parse", "--show-toplevel").rstrip(b"\n")))
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
