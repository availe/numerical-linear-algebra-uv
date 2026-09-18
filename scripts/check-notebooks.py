"""Validate lessons; execute copies, leaving deliberate exercises unsolved."""
import argparse
from pathlib import Path
import tempfile
import shutil
import nbformat
from nbclient import NotebookClient
from IPython.core.inputtransformer2 import TransformerManager

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Execute notebooks (downloads data; can take hours).')
    parser.add_argument('--notebook', action='append', help='Filename glob; defaults to all notebooks.')
    parser.add_argument('--timeout', type=int, default=1800, help='Per-cell timeout in seconds.')
    parser.add_argument('--write-outputs', action='store_true', help='Save executed outputs locally in the lesson files (requires --execute).')
    args = parser.parse_args()
    if args.write_outputs and not args.execute:
        parser.error('--write-outputs requires --execute')
    paths = sorted({p for pattern in args.notebook or ['*.ipynb'] for p in (ROOT/'nbs').glob(pattern)})
    if not paths:
        parser.error('No notebooks matched.')
    transformer = TransformerManager()
    failures = []
    # One scratch tree per run, rather than copying the large graph for each lesson.
    with tempfile.TemporaryDirectory(prefix='linear-algebra-') as temp:
        scratch = Path(temp)/'nbs'
        if args.execute:
            shutil.copytree(ROOT/'nbs', scratch)
        for path in paths:
            nb = nbformat.read(path, as_version=4)
            nbformat.validate(nb)
            for i, cell in enumerate(nb.cells):
                if cell.cell_type == 'code':
                    compile(transformer.transform_cell(cell.source), f'{path.name}:cell {i}', 'exec')
            print(f'VALID {path.name}', flush=True)
            if not args.execute:
                continue
            skipped = [c for c in nb.cells if 'exercise-dependent' in c.metadata.get('tags', [])]
            for c in skipped:
                print(f'  EXERCISE {c.id}: {c.metadata["skip_reason"]}', flush=True)
            def progress(cell, cell_index):
                if cell.cell_type == 'code':
                    print(f'  cell {cell_index + 1}/{len(nb.cells)} ({cell.id})', flush=True)
            try:
                NotebookClient(nb, timeout=args.timeout, kernel_name='python3',
                               skip_cells_with_tag='exercise-dependent',
                               on_cell_start=progress,
                               resources={'metadata': {'path': str(scratch)}}).execute()
            except Exception as exc:
                failures.append(path.name)
                print(f'FAIL {path.name}: {exc}', flush=True)
            else:
                print(f'PASS {path.name} ({len(skipped)} exercise-dependent cells skipped)', flush=True)
                if args.write_outputs:
                    nbformat.write(nb, path)
            finally:
                report = ROOT/'notebook-reports'/path.name
                report.parent.mkdir(exist_ok=True)
                nbformat.write(nb, report)
    if failures:
        raise SystemExit('Failed notebooks: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
