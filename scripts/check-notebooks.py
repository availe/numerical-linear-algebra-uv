"""Validate all notebooks, optionally execute copies without overwriting lessons."""
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
    args = parser.parse_args()
    paths = sorted({p for pattern in args.notebook or ['*.ipynb'] for p in (ROOT/'nbs').glob(pattern)})
    if not paths:
        parser.error('No notebooks matched.')
    transformer = TransformerManager()
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                compile(transformer.transform_cell(cell.source), f'{path.name}:cell {i}', 'exec')
        print(f'VALID {path.name}', flush=True)
        if args.execute:
            skipped = [c for c in nb.cells if 'exercise-dependent' in c.metadata.get('tags', [])]
            for c in skipped:
                print(f'  EXERCISE {c.id}: {c.metadata["skip_reason"]}', flush=True)
            # Copy assets/data into a scratch directory: execution must not modify lessons.
            with tempfile.TemporaryDirectory(prefix='linear-algebra-') as temp:
                shutil.copytree(ROOT/'nbs', Path(temp)/'nbs')
                def progress(cell, cell_index):
                    if cell.cell_type == 'code' and cell_index % 20 == 0:
                        print(f'  cell {cell_index + 1}/{len(nb.cells)}', flush=True)

                NotebookClient(nb, timeout=args.timeout, kernel_name='python3',
                               skip_cells_with_tag='exercise-dependent',
                               on_cell_start=progress,
                               resources={'metadata': {'path': str(Path(temp)/'nbs')}}).execute()
            report = ROOT/'notebook-reports'/path.name
            report.parent.mkdir(exist_ok=True)
            nbformat.write(nb, report)
            print(f'PASS {path.name} ({len(skipped)} exercise-dependent cells skipped)', flush=True)


if __name__ == '__main__':
    main()
