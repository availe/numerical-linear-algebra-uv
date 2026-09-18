# Modernization and fidelity notes

This fork targets Python 3.14.7 with the exact dependency versions in `uv.lock`. Use `uv sync --locked`. Updating the lockfile is a separate, deliberate upgrade. The course baseline is `baccd5e44049a7265cc4708cce662f4942f14f09`; the README baseline is `8c8efd33151d92494a555e71119f0d6328c85a6b`.

## Repeatability switch

Edit `REPRODUCIBLE = True` in [`../nbs/course_config.py`](../nbs/course_config.py). Set it to `False` to disable the added repeatability controls, then restart notebook kernels and run from the top. `SEED = 42` is configurable in the same file. This controls lessons 2–8 and `gradient-descent-intro.ipynb` (eight notebooks), including NumPy, Python random, PyTorch initialization, and the added scikit-learn split/SVD settings. Explicit seeds that were already part of the original teaching examples remain intact; disabling our switch restores their original behavior, including NMF's original `random_state=1`. The switch does not guarantee identical timings, floating-point results across hardware/library versions, or deterministic GPU algorithms. Running a later random cell repeatedly still advances the generator; restart and run from the beginning to repeat a run.

## Git policy

Run `scripts/setup-git.sh` once per clone after environment setup. It installs a repository-local clean filter. The notebooks on disk keep your outputs, but Git stores code, Markdown, cell order, attachments, IDs, and lesson metadata without outputs, execution counters, timings, widget state, and editor execution state. The script locates the repository from its own path (including paths with spaces) and checks the repository root without comparing platform-specific path spellings. It uses the already synchronized environment and does not download packages during Git operations. `scripts/check-notebooks.py` validates and optionally executes notebooks; it is not a Git filter.

Changing from the old output-preserving filter requires one normalization of the tracked notebooks. That change needs to be committed with the filter policy. Running everything once does not eliminate output differences: timings, renderings, and random values can change on every run. Even with seeds enabled, output filtering remains necessary. Code edits and changes to Markdown or cell order must still appear in Git.

Some original comment-only exercise cells contain instructor reference outputs. Running those cells cannot recreate those examples. We preserve the original revision/history and link to the [original course notebooks](https://github.com/fastai/numerical-linear-algebra/tree/master/nbs), also rendered by the README's nbviewer links. For an immutable local reference, open the notebook from commit `baccd5e44049a7265cc4708cce662f4942f14f09`. Execution order within the file is preserved except for the explicitly documented premature sparse-component plot below. Execution counters are not a reliable record of a notebook's dependencies.

## Decisions beyond library compatibility

The earlier migration made unnecessary scope changes. Synthetic video, a default 100,000-line graph sample, rewriting the homework-3 stub, and rewriting the README were not needed for modernization. Those choices have been removed. These are the retained or corrected changes, rather than undisclosed lesson rewrites:

| Material | Decision and reason |
|---|---|
| Lesson 2, NMF NumPy penalty | Keep elementwise `minimum(M-mu, 0)`. The original `np.min(..., 0)` reduces over rows, producing the wrong gradient. This repairs the supplied implementation, not an exercise. See the [course discussion of the penalty](https://forums.fast.ai/t/gradient-formula-used-in-nmf/23882). |
| Lesson 2, CPU/GPU | Select CUDA when available, otherwise CPU. Modern tensor/autograd APIs replace obsolete `Variable`, `.data` mutation/access and CUDA-only constructors. Two explanatory Markdown cells are updated to match; the optimization lesson is retained. |
| Lesson 3, original video | Remove the synthetic fallback. Download the course's `Video_003.avi` from the [mirror linked by the course forum](https://forums.fast.ai/t/bmc-2012-dataset-missing/31535), pinned to a commit and SHA-256. |
| Lesson 3, resizing | Replace removed SciPy `imresize` with Pillow bilinear resizing and reproduce its historical per-image byte scaling, including rounding. A direct uint8 cast in the earlier migration did not preserve that normalization. See [SciPy 1.2.1 implementation](https://github.com/scipy/scipy/blob/v1.2.1/scipy/misc/pilutil.py). |
| Lesson 3, plots | Derive frame dimensions from the video; scale crop coordinates with image resolution; restore the sparse-component plot and move that one cell immediately after `S` is computed. The original empty crop/undefined `S` were [reported as bugs upstream](https://github.com/fastai/numerical-linear-algebra/issues/8). Plot rows follow the actual recorded iterations. |
| Lesson 3, PCP | Correct the `mu` clamp (the original accidentally assigns `m`) and select frame columns/rows according to the transpose flag. The [frame orientation issue is documented upstream](https://github.com/fastai/numerical-linear-algebra/issues/12). Keep the original algorithm and convergence rule. |
| Lesson 4 | Add the missing NumPy import; repair `18 x 128 x 128 x 128` to multiplication; save the three generated figures under ignored `data/figures/` so execution does not overwrite teaching assets. |
| Lesson 6 | Close HDF stores with a context manager. When the deliberately rank-deficient example raises a singular-matrix error, print it and record NaNs so later comparisons can run. Retain the intentionally unstable inverse method, rather than substituting a better solver. |
| Lesson 7, data | Restore the full original DBpedia 3.5.1 archives and original example pages. The corrected redirect chains change numeric page IDs; derive those IDs while retaining the Cincinnati Reds/W711-2 worked example, with three small Markdown corrections to explain the lookups. Keep the original edge position in the pinned full dataset. Verify their content hashes. Store edge lists in compact numeric arrays rather than Python object lists to reduce memory without sampling or changing edge order. This remains a large, resource-intensive lesson. |
| Lesson 7, implementation bugs | Size the graph by pages, not edges; inspect the last page without deleting it; resolve redirect chains against a populated mapping; make a real sparse matrix copy before normalization. These prevent corrupt graph construction or invalid sparse operations. Convert one prose-only code cell to Markdown. |
| Lesson 8 | Call the provided Householder routine before using its results. Repair `implicit_Qx`: use the supplied reflector vectors, apply them in reverse order, avoid mutating the input, and return the result. A reconstruction test checks `Q @ R == A`. This provided function was broken, not labeled as a student exercise. |
| Homework 2 | Add the missing NumPy import. Leave pivoting and in-place LU assignments unsolved. |
| Homework 3 | Restore the original `practical_qr` stub verbatim. Do not implement shifted QR. |
| Convolution introduction | Restore missing `pool`/`pool8` definitions from the upstream fast.ai convolution notebook: **7×7 max pooling**, not the earlier guessed 2×2. The [missing definitions were reported upstream](https://github.com/fastai/numerical-linear-algebra/issues/7); its discussion links the original source. |
| Randomness | Make the added seeds optional through one shared switch rather than scattering hard-coded settings. |
| README | Restore the user's version, with only locked setup commands, the accurate Git-filter paragraph, and a short pointer to these notes/the toggle. |

Other compatibility changes replace removed NumPy aliases, matrix conversions, scikit-learn feature-name/SVD/joblib/MNIST APIs, pandas scalar assignment, MoviePy video methods, SciPy image loading/imports, Matplotlib `basey`, and obsolete lasso arguments. MNIST is pinned to OpenML data ID 554. Removed imports were unused. Stable cell IDs and a generic Python kernel support current notebook tooling. Language-version metadata is runtime noise; the Python pin and lockfile define the environment. The user's later `scipy-stubs` dependency is retained.

[The complete cell-by-cell comparison](notebook-changes.md) lists every source/type change against the original fork, including the exact code. It also identifies cell movement and exercise tags. The images, spreadsheet, and other original teaching files have not been rewritten.

## Exercises and execution

The course logistics explicitly say parts are left for students to fill. The validator marks and skips only 14 downstream cells dependent on those missing answers (one in lesson 2, six in lesson 3, seven in lesson 7), with a reason printed for each. It does not solve the assignments or swallow arbitrary runtime errors. Ordinary Run All in an interactive notebook can still stop at an unfinished exercise; complete the exercise to proceed. The original numerically unstable examples remain part of the lesson.

From the repository root:

```bash
uv run --locked pytest -q
uv run --locked python scripts/check-notebooks.py
uv run --locked python scripts/check-notebooks.py --execute --timeout 3600
```

The last command runs fresh kernels in a temporary copy and saves executed reports under ignored `notebook-reports/`. Add `--write-outputs` to also save successful results into the local lesson notebooks. It continues to other notebooks after execution errors and exits unsuccessfully if any fail. Network access is needed for the first dataset downloads; the DBpedia archives alone are about 934 MB compressed. Dataset hashes and provenance are in `nbs/course-data.json`. No synthetic or sampled data is substituted. Tests use small synthetic fixtures where appropriate; those fixtures are not the teaching dataset.

## Verified results (2026-09-18)

- Python 3.14.7, macOS/Apple Silicon; the locked project environment synchronized successfully. The full-run environment had the same runtime package versions as the project; the project's three additional typing/stub packages were retained and the final tests ran in its own environment.
- All 14 notebooks executed successfully with fresh kernels, using the original video and full DBpedia data. Exactly 14 exercise-dependent cells were skipped as documented above. This validates the supplied demonstrations and assignment scaffolds, not unimplemented student answers.
- PageRank was rerun after correcting the worked-example lookups: 119,077,682 links, 9,990,822 pages, and the original W711-2 example with 47 outgoing links. No sampling or synthetic teaching data was used.
- All 31 focused tests passed in the actual project environment. All 14 notebooks passed schema and transformed-Python compilation checks.
- After one-time normalization, lesson 1 was executed again in the actual project. `git diff --exit-code -- nbs/*.ipynb` succeeded: no execution-only unstaged notebook changes. Original images, spreadsheet, lockfile and dependency declarations were unchanged by this update.
- The final comparison records 87 changed source/type cells against the original course (including five Markdown source changes), one moved plot, no added/deleted cells, and 82 byte-identical original non-notebook files. Changes to notebook IDs/kernel/runtime metadata are separate from those source counts.

The changes and normalization are staged for review, not committed. Commit this normalization with the filter policy to establish the new shared baseline. Local execution outputs remain on disk. Other operating systems are configured in CI but were not tested locally.
