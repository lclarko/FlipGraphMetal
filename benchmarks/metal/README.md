# Metal developer tools

These optional tools support correctness comparisons, source snapshots and performance investigation. Python is not part of the production runtime. Read [development](../../docs/development.md) for normal tests and [performance](../../docs/performance.md) for measured results and their limits.

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

`application.py` compares external CPU and baseline/candidate Metal applications. Supply explicit binary, source and build-manifest arguments for each selected backend; inspect `--help` for their names. Use `--timing source-elapsed` for source-clock timing, and a new output directory. Custom fixtures require `--fixtures custom --fixture-path PATH` with one raw signed scheme and no scheme-count prefix. Dimensions must be 1..16, each factor width at most 64, and rank 1..350. The runner checks the exact coefficient count, restricts coefficients to −1, 0 or 1, independently verifies the tensor, records the hash and rechecks it before cases. Use `--expected-kernel` to bind GPU evidence to the intended search kernel.

`summarize_application.py DIRECTORY` rechecks retained records and timing. Report per-fixture throughput ratios and confidence intervals. Summaries contain descriptive estimates and uncertainty, without fixed improvement targets or promotion judgments. Preserve CPU thread counts, work sizes and policy differences in any report. GPU command time, application time and wall time must remain distinct. The historical CPU has no round cap, so asynchronously stopped CPU exports may include work after the measured interval.

## Signed 4×4 benchmark

The checked-in `tests/metal/fixtures/naive_4x4.txt` and `strassen_4x4.txt` start at ranks 64 and 49. Their adjacent provenance files describe schoolbook multiplication and the tensor square of the seven-product Strassen formulas. Both satisfy all 4096 exact-integer tensor equations. To reproduce their bytes in a new directory:

```sh
python3 benchmarks/metal/make_4x4_fixtures.py --output "$ATTEMPT/generated-fixtures"
```

Use the general kernel for 4×4. With the same explicit frozen reference and fresh `ATTEMPT` conventions above, build and check both fixtures:

```sh
python3 benchmarks/metal/guard.py --output "$ATTEMPT/general-build-guard" -- \
  python3 benchmarks/metal/build.py \
    --source "$REFERENCE_METAL_SOURCE" --gpu-source "$PWD/src/metal" \
    --gpu-kernel randomWalkKernel --rank-capacity 350 \
    --output "$ATTEMPT/general-profile"

for fixture in naive_4x4 strassen_4x4; do
  python3 benchmarks/metal/screen.py \
    --builds "$ATTEMPT/general-profile" --output "$ATTEMPT/check-$fixture" \
    --expected-kernel randomWalkKernel \
    --populations 33 512 --seeds 7 19 --repeats 1 --mode matched \
    --fixture "$PWD/tests/metal/fixtures/$fixture.txt" || exit 1
done
```

Inspect the six `ACTIVITY` rows as well as each exact `MATCH`: changed states demonstrate activity; rank ranges and candidate counts describe the resulting walks. Failed-flip recovery can expand a scheme even when optional expansion probability is zero, so search ranks can change from the starting rank.

For application measurements, first create the frozen application snapshot above. Set `CPU_BINARY`, `CPU_SOURCE` and `CPU_BUILD_MANIFEST` to a compatible, pinned external CPU build. The manifest must match its source and executable; the CPU must emit the source-clock report format accepted by the runner.

Before confirmation, run both fixtures with six reports, seed 7 and two repeats at populations 512 and 2048, each in a fresh pilot directory. The measured protocol selected 2048 only when every pilot completed, maximum process duration multiplied by 17/6 was below 30 seconds, and sampled wired memory was below 2.75 GiB. These are headroom checks within the hard guards, not permanent resource guarantees. If the intended report count fails the checks, stop and declare a shorter protocol before collecting confirmation data. Preserve all pilots and incomplete attempts; exclude pilot timings from confirmation statistics.

After those checks, the 120-process confirmation design is:

```sh
for fixture in naive_4x4 strassen_4x4; do
  python3 benchmarks/metal/application.py \
    --output "$ATTEMPT/confirmation-$fixture" --backends cpu candidate \
    --cpu "$CPU_BINARY" --cpu-source "$CPU_SOURCE" \
    --cpu-build-manifest "$CPU_BUILD_MANIFEST" \
    --candidate "$ATTEMPT/application/flip_graph" \
    --candidate-source "$ATTEMPT/application/source" \
    --candidate-build-manifest "$ATTEMPT/application/build.json" \
    --fixtures custom --fixture-path "$PWD/tests/metal/fixtures/$fixture.txt" \
    --populations 2048 --seeds 7 19 41 73 101 --repeats 6 --rounds 17 \
    --timing source-elapsed --expected-kernel randomWalkKernel || exit 1
  python3 benchmarks/metal/summarize_application.py \
    "$ATTEMPT/confirmation-$fixture" || exit 1
done
```

The runner uses eight CPU threads, 1000 attempted iterations per scheme per round, and alternating backend order across repetitions. Timing uses report 17 minus report 1, excluding the first round. Attempted iterations are not successful flips or equal candidate work, and CPU/Metal policies differ. Report each fixture separately without interpreting throughput as equal search quality.

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

### Relocated evidence

The default summarizer verifies the original executable, source and build-manifest paths. For an offline bundle, use `python3 benchmarks/metal/summarize_application.py BUNDLE/RUN --archive-map BUNDLE/archive-map.json`. No executable is run by this mode, and relocated frozen binaries are not standalone installations.

The separate map has version 1, a `paths` object mapping each original absolute binary, source directory, build manifest and Metal shader directory to its bundled relative path, and a `campaigns` object keyed by each bundled run directory. Every campaign entry contains `files`, mapping every original run-relative file path (including config, results, seals, logs and exports) to its SHA-256 digest, and `directories`, listing every original run-relative subdirectory. Verification requires an exact inventory match, rejecting changed or resealed records, missing files and extra files or directories. All mapped paths and case artifacts must stay inside the map's directory; symlinks and parent traversal are rejected. Include every source file and case artifact covered by the original inventories. Preserve original config, manifests, records and logs byte-for-byte. Source inventories, binary and manifest hashes, shader bindings, result seals and artifact hashes are still verified against the retained records.

Create this mapping from the preserved evidence and retain its digest through a separately trusted channel. The mapping must capture the complete original campaign population before relocation, including incomplete attempts; do not construct a new trusted inventory from an untrusted submitted bundle. Mapping hashes detect changes relative to that record; they do not authenticate authorship or prove that measurements occurred. Changing mappings or resealing records cannot establish independent verification. Archive verification does not sanitize private evidence: original paths, identifiers in older machine records, fixtures and logs need a separate sharing review. Never overwrite the private originals with sanitized copies.

New runs retain only model, chip, CPU/GPU core, memory and Metal-support fields from the hardware profiler. Raw profiler output and error text are discarded. Compiler and thermal diagnostics remain separate records and should also be reviewed before sharing.
