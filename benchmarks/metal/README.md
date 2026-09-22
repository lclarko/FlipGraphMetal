# Metal developer tools

These optional tools support correctness comparisons, source snapshots and performance investigation. Python is not part of the production runtime. Read [development](../../docs/development.md) for normal tests and [performance](../../docs/performance.md) for the historical results and limits.

## Prerequisites and supervision

Use Apple silicon with accessible Metal hardware, macOS 15 or later, Xcode Command Line Tools and Python 3.9+. Matched CPU profiling also needs OpenMP (`libomp`); the current profiling build expects its Homebrew prefix at `/opt/homebrew/opt/libomp`. Instruments captures require Xcode's tracing tools. External CPU application comparisons need an explicitly selected source tree, executable and build manifest; that repository is not included here.

Run GPU work serially. The guards use a 45-second process-group timeout and sampled 3 GiB system wired-memory cutoff, followed by TERM and KILL after two seconds. Do not nest another process-group supervisor around `screen.py` or `application.py`, which already own their supervision. Retain incomplete attempts and choose a new output directory for retries.

## Complete-walk comparison

`build.py` compiles selected Metal source as the C++ reference and builds the selected GPU implementation. This reference is distinct from the separate `ternary_flip_graph` CPU application. Set both source directories explicitly: omitting them can select the same source for both sides. The general kernel is the build default, so packed validation must select `randomWalkCompactKernel` explicitly.

From the checkout root, with `REFERENCE_METAL_SOURCE` set to a reviewed compatible frozen source directory and `ATTEMPT` set to a fresh absolute output directory:

```sh
python3 benchmarks/metal/guard.py --output "$ATTEMPT/build-guard" -- \
  python3 benchmarks/metal/build.py \
    --source "$REFERENCE_METAL_SOURCE" --gpu-source "$PWD/src/metal" \
    --gpu-kernel randomWalkCompactKernel --rank-capacity 350 \
    --output "$ATTEMPT/profile"

python3 benchmarks/metal/screen.py \
  --builds "$ATTEMPT/profile" --output "$ATTEMPT/naive" \
  --expected-kernel randomWalkCompactKernel \
  --populations 33 512 --seeds 7 19 --repeats 1 --mode matched

python3 benchmarks/metal/screen.py \
  --builds "$ATTEMPT/profile" --output "$ATTEMPT/rank23" \
  --expected-kernel randomWalkCompactKernel \
  --populations 33 512 --seeds 7 19 --repeats 1 --mode matched \
  --fixture "$PWD/tests/metal/fixtures/rank23_3x3.txt"
```

Verify reference hashes before building. The manifest must identify reference source, GPU source, runtime, kernel and rank capacity. `--expected-kernel` is required by the screen and checks the manifest and every walk dispatch. Output directories must be new.

Matched mode runs six rounds of 1000 iterations. Acceptance requires all rounds, exact current/best state, ordered candidates, RNG and counters, successful tensor checks, independently verified exports and positive finite GPU timings. A successful general-kernel comparison cannot stand in for the packed path. Frozen sources must use compatible layouts; the tool does not translate arbitrary revisions.

## Application snapshots and comparisons

Freeze a coherent project root, including its parser dependencies:

```sh
python3 benchmarks/metal/freeze_application.py \
  --project-root "$PWD" --output "$ATTEMPT/application"
```

The snapshot resolves the parser only within the selected project root, supporting the original or current layout and rejecting ambiguous dependencies. It records source identities, build arguments and executable hashes. Its executable uses the snapshot's absolute shader directory; keep that directory intact.

`application.py` compares external CPU and baseline/candidate Metal applications. Supply explicit binary, source and build-manifest arguments for each selected backend; inspect `--help` for their names. Use `--timing source-elapsed` for source-clock timing, and a new output directory. Custom fixtures require `--fixtures custom --fixture-path PATH` with one raw signed 3×3 scheme, no scheme-count prefix. The runner validates the tensor, records the hash and rechecks it before cases.

`summarize_application.py DIRECTORY` rechecks retained records and timing. Preserve CPU thread counts, work sizes and policy differences in any report. GPU command time, application time and wall time must remain distinct. The historical CPU has no round cap, so asynchronously stopped CPU exports may include work after the measured interval.

## Diagnostics

| Tools | Purpose |
|---|---|
| `build.py`, `profile.cpp`, `kernels.metal`, `screen.py` | Matched walks and diagnostic kernels |
| `freeze_application.py`, `application.py`, `summarize_application.py` | Frozen production application comparisons |
| `guard.py` | Bounded process supervision and retained logs |
| `storage.py`, `simd.py` | Optional layout and SIMD variants |
| `algorithms.py` | Optional search-policy variants |
| `capture.py`, `counters.py` | Instruments capture and exported counter analysis |

Storage, SIMD and algorithm generators create diagnostic snapshots rather than changing production defaults. Select external inputs explicitly where required. Policy variants need their own reference checks and search-quality evaluation; their timings are not interchangeable with production measurements.

Captures must use the minimal subprocess environment and exclude sensitive environment metadata. Counter summaries describe the exported samples; they do not by themselves establish occupancy, frequency changes or causal explanations of speedups.
