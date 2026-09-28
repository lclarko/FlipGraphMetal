# Metal developer tools

These optional tools support correctness comparisons, source snapshots and performance investigation. Python is not part of the production runtime. Read [development](../../docs/development.md) for normal tests and [performance](../../docs/performance.md) for measured results and their limits.

## Prerequisites and supervision

Use Apple silicon with accessible Metal hardware, macOS 15 or later, Xcode with the Metal Toolchain and Python 3.9+. Matched CPU profiling also needs OpenMP (`libomp`); the current profiling build expects its Homebrew prefix at `/opt/homebrew/opt/libomp`. Instruments captures require Xcode's tracing tools. External CPU application comparisons need an explicitly selected source tree, executable and build manifest; that repository is not included here.

Run GPU work serially. The guards use a 45-second process-group timeout and sampled 3 GiB system wired-memory cutoff, followed by TERM and KILL after two seconds. Do not nest another process-group supervisor around `screen.py` or `application.py`, which already own their supervision. Retain incomplete attempts and choose a new output directory for retries.

## Pinned workload baseline

The workflow benchmark adapters cover signed packed 3×3 search, two signed general 4×4 searches, F2 search, both minimizers and fixed/flip-enabled signed reduction. They use repository fixtures and require compiled shader libraries. A fresh checkout can prepare the baseline without an existing measurement bundle.

With `ATTEMPT` set to a new absolute output directory:

```sh
python3 benchmarks/workflow/baseline.py --freeze "$ATTEMPT/baseline"
python3 benchmarks/metal/guard.py --output "$ATTEMPT/baseline-build" -- \
  make -C "$ATTEMPT/baseline/source" METAL_LIBRARY_MODE=metallib metal
python3 benchmarks/workflow/baseline.py \
  --source "$ATTEMPT/baseline/source" --output "$ATTEMPT/baseline-pilot"
```

Source preparation pins `2f91a88` and records the Git tree and archive digest. It does not build or measure anything. The separate build and measurement steps retain their own receipts. Keep the resulting source, binaries, libraries and receipts unchanged after qualification.

The pilot starts with six rounds per workload. Use baseline-only pilots to select work amounts that give useful timings within the guards, normally around 5–10 seconds per process. Each run writes `protocol.json`; save calibrated settings as a separately versioned protocol before evaluating a candidate. Pass that file with `--protocol FILE`, and use `--repetitions` for repeated baseline observations. `baseline.py --summarize RUN/qualification.json --output NEW_SUMMARY` reports repeatability diagnostics without granting any regression allowance.

`profile_baseline.py --baseline BASELINE --output NEW_PROFILE` prepares a separate instrumented source snapshot. Build it under the same guard, then qualify its observations with `baseline.py --qualify-profile PROFILE --baseline BASELINE --production-run RUN --output NEW_CHECK`. New qualification records share one attempt structure and identify production or diagnostic mode. Existing v1 records remain readable without rewriting them. The production run is explicit; no particular pilot-directory name is required. Qualification compares shader bytes, inputs and independently verified exports. It does not replace complete-walk equivalence or production timing.

`benchmarks/workflow/performance.py INPUT --output NEW_RECEIPT` evaluates retained paired observations for existing production paths. Its input binds approved regression budgets, the mandatory endpoint roster, both planned looks, matched-work identities and a separate precision assessment. Use `status: "unapproved"`, `approval: null` and `margin: null` when no margin has been selected. Unapproved budgets and insufficient evidence cannot pass. A hard resource violation remains a failure even when a practical margin is still unapproved. A slowdown inside an approved tolerance must still be reported as a slowdown.

`baseline.py --host build/metal/scheme_tool --repetitions 6 --output NEW_RUN` measures verified reporting, CPU-text round trips and bounded selection from generated inventories. It retains fixtures, build settings, hardware, the ordered workload roster and every attempt. Complete-workflow time includes independent output verification; native process time, verification time and peak native RSS are reported separately. Input generation occurs before timing. These repeated measurements establish a usability and scaling baseline, without absolute latency budgets or statistical acceptance verdicts. Execution safeguards, exact verification and source/build checks remain required. Incomplete attempts are retained and reported as failures. This host-only mode does not initialize Metal or establish GPU performance.

`baseline.py --native-config RUN.json --native-binary build/metal/flip_graph --output NEW_RUN`
measures one configured native workflow. It runs one trial by default; request
repetitions explicitly. Each trial has separate outputs and history. Resume
measurements copy history before execution. The existing supervisor, resource
measurements and independent verifiers are reused. Records include actual
operations, captures, committed discoveries, native phase clocks, process time
and workflow time including verification. Circuit artifacts and read-only journal
exports are independently checked. The result is `NOT EVALUATED` for performance
and grants no regression budget. Controlled and legacy iteration counts cannot
be compared as equal work.

## FGM-1 short-run effectiveness baseline

The checked-in panel at `benchmarks/workflow/fixtures/fgm1/panel.json` contains
five signed rank-23 3×3 factor starts and three supplied reference circuits.
To reproduce panel preparation from a retained corpus, use a new directory. This
preparation verifies source hashes, exact factors and reference circuits
without launching Metal:

```sh
python3 benchmarks/workflow/baseline.py \
  --prepare-effectiveness-panel RETAINED_CORPUS --output NEW_PANEL_DIR
```

After building the native executables and compiled signed Metal library, run
one measurement into a new directory. A later run can reuse calibration only
when its retained pilot evidence, build inputs, fixture bytes and hardware
bindings still match:

```sh
make metal scheme-tool
python3 benchmarks/workflow/baseline.py \
  --effectiveness-panel benchmarks/workflow/fixtures/fgm1/panel.json \
  --binary-dir build/metal --output NEW_RUN
python3 benchmarks/workflow/baseline.py \
  --effectiveness-panel benchmarks/workflow/fixtures/fgm1/panel.json \
  --binary-dir build/metal --calibration FIRST_RUN/calibration.json \
  --output NEW_REPEAT
python3 benchmarks/workflow/baseline.py \
  --summarize-effectiveness NEW_RUN/measurement.json --output NEW_SUMMARY.json
```

The timed protocol has a 15-minute envelope after supplied-reference
verification. Calibration uses the five starts and halves initial rounds or
workers until the one-second reduction and five-second generation pilot limits
hold. Three seeds run fixed reduction and cost-blind generation for each start,
giving 30 paired 20-second arms. The 10- and 20-second endpoints use the time
when independent verification finishes. Every child retains the existing
45-second process-group and sampled 3 GiB wired-memory guard; no child launches
inside the final 50-second cleanup reserve. An incomplete arm stays incomplete.
Late verified circuits and captures remain in the evidence but earn no earlier
endpoint credit. A committed native discovery whose observation export could
not finish is recorded as unexported, not scored as an evaluated circuit.

Protocol versions 2 and 3 wait for 1216 MiB of system wired-memory headroom before
each GPU child, polling every 25 ms within the pilot or arm deadline. Waiting
is charged and retained as `headroom_wait_seconds`. This admission heuristic
was selected from version 1 observations; a later version 2 attempt showed a
larger transient, so it is not a worst-case memory bound and does not replace
the hard 3 GiB guard. Version 3 uses 128 reducers in both arms, halving the
planned reducer buffers from approximately 365 MiB to 182 MiB. Earlier versions
used 256. Host verification and journal export do not use this GPU reservation.
Each version requires separate calibration and must not be pooled with the
others as repetitions. A step canceled before child launch remains unscored
bookkeeping.

The harness is Python; production search, reduction and journal export are
native C++/Metal. Python independently verifies tensors, circuit factors,
identities and addition counts. Supplied reference circuits are checked during
preparation and reported separately. Their costs do not count as rediscovered
circuits. `measurement.json`, `calibration.json`, per-step receipts and logs,
`summary.json` and `report.md` retain complete and incomplete evidence. An
offline summary checks calibration and frozen artifacts, deterministic step
configurations, sequential timelines, the 900-second completion limit, and
retained circuit bindings. It uses the current `build/metal/scheme_tool` to
replay each search receipt's acknowledged journal prefix and compare its
ordered captures with the retained export. Run `make scheme-tool` before
replaying older bundles. Calibration reuse retains pilot journals as well as
step evidence. Replay is read-only and does not launch Metal.
See `docs/performance.md` for measured findings.

## Complete-walk comparison

`build.py` compiles selected Metal source as the C++ reference and builds the selected GPU implementation. This reference is distinct from the separate `ternary_flip_graph` CPU application. Set both source directories explicitly: omitting them can select the same source for both sides. The general kernel is the build default, so packed validation must select `randomWalkCompactKernel` explicitly.

From the checkout root, with `REFERENCE_METAL_SOURCE` set to a reviewed compatible frozen source directory and `ATTEMPT` set to a fresh absolute output directory:

```sh
python3 benchmarks/metal/guard.py --output "$ATTEMPT/build-guard" -- \
  python3 benchmarks/metal/build.py \
    --source "$REFERENCE_METAL_SOURCE" --gpu-source "$PWD/src/metal" \
    --gpu-kernel randomWalkCompactKernel --rank-capacity 350 --gpu-library-mode metallib \
    --output "$ATTEMPT/profile"

python3 benchmarks/metal/screen.py \
  --builds "$ATTEMPT/profile" --output "$ATTEMPT/naive" \
  --expected-kernel randomWalkCompactKernel --expected-library-mode metallib \
  --populations 33 512 --seeds 7 19 --repeats 1 --mode matched

python3 benchmarks/metal/screen.py \
  --builds "$ATTEMPT/profile" --output "$ATTEMPT/rank23" \
  --expected-kernel randomWalkCompactKernel --expected-library-mode metallib \
  --populations 33 512 --seeds 7 19 --repeats 1 --mode matched \
  --fixture "$PWD/tests/metal/fixtures/rank23_3x3.txt"
```

Verify reference hashes before building. The manifest must identify reference source, GPU source, runtime, kernel and rank capacity. `--expected-kernel` is required by the screen and checks the manifest and every walk dispatch. Select `--gpu-library-mode metallib` and `--expected-library-mode metallib` to validate the production library path. The reference diagnostics continue to use their independently frozen source. Library digests and executable bindings are rechecked before every case. Output directories must be new.

Matched mode runs six rounds of 1000 iterations. Acceptance requires all rounds, exact current/best state, ordered candidates, RNG and counters, successful tensor checks, independently verified exports and positive finite GPU timings. A successful general-kernel comparison cannot stand in for the packed path. Frozen sources must use compatible layouts; the tool does not translate arbitrary revisions.

## Application snapshots and comparisons

Freeze a coherent project root, including its parser dependencies:

```sh
python3 benchmarks/metal/freeze_application.py \
  --project-root "$PWD" --output "$ATTEMPT/application"
```

The snapshot resolves the parser only within the selected project root, supporting the original or current layout and rejecting ambiguous dependencies. It records source identities, build arguments and executable hashes. Current project snapshots compile a matching library and bind its digest into the executable. Keep its sibling `shaders/` directory with it. The generated header and frozen build helper remain part of the verification evidence. Older projects retain source mode and their absolute shader-directory requirement; `--library-mode source` explicitly selects that mode for experiments. Shader and host compilation each retain guarded logs and an incomplete manifest on failure.

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
    --gpu-kernel randomWalkKernel --rank-capacity 350 --gpu-library-mode metallib \
    --output "$ATTEMPT/general-profile"

for fixture in naive_4x4 strassen_4x4; do
  python3 benchmarks/metal/screen.py \
    --builds "$ATTEMPT/general-profile" --output "$ATTEMPT/check-$fixture" \
    --expected-kernel randomWalkKernel --expected-library-mode metallib \
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

## FGM-2 construction comparison

Build and qualify the native programs first (`make test-workflow`,
`make test-metal`, and the existing relocation/packaging checks). The comparison
uses the five checked-in FGM-1 starts plus one local private factor-only input.
Reference circuits are audited separately and never passed to synthesis.

```sh
python3 benchmarks/workflow/baseline.py --fgm2 \
  --fgm2-private-input /absolute/path/to/factors.json \
  --fgm2-private-reference /absolute/path/to/reference-circuit.json \
  --binary-dir build/metal --output build/fgm2/comparison-01
```

The optional reference must be FGM circuit JSON bound to the submitted factors.
Private inputs, references, and outputs remain uncommitted. Public tests have no
private dependency. Keep the retained reference-family audit with the attempt;
a supplied direct-W circuit alone does not establish Wᵀ family coverage.

The frozen [protocol](../workflow/fixtures/fgm2/protocol.json) has 72 trials:
six inputs, seeds 7/19/41, and four strategies. Every invocation gets the same
128-reducer, 16-round direct baseline quantum. Transposition adds another
128-reducer, 16-round pair quantum with a separate deterministic seed;
restricted construction enumerates its finite family without RNG. Strategy
order rotates deterministically. The comparison measures fixed work plus
extensions, not matched-time algorithm effectiveness.

Each trial includes configuration, input checks, resource waiting, supervision,
publication, and independent verification. Preparation and headroom waiting
consume the overall budget. After those finish, the entire native invocation
receives a 45-second supervision allowance. The independent Python verifier
receives a separate 45-second allowance, with its reference derived independently
from retained input factors. Neither allowance resets within a child process.
Early completion advances the schedule and earns no extra reduction work.

The overall envelope remains 900 seconds with a 50-second finalization reserve.
Before launching native work, at least 150 seconds must remain: 45 for native
execution, 45 for verification, five for bookkeeping before and after
verification, and 50 for finalization. Verifier admission requires 100 seconds:
45 for its execution, five for final binding and bookkeeping, and 50 for
finalization. These are conservative admission reserves, not mandatory waits.
Headroom waiting stops when native admission is no longer possible, and the
harness rechecks the budget immediately before each supervised call.

The 3-GiB wired-memory cutoff and existing launch-headroom reserve are unchanged.
A forced GPU-child termination or failed cleanup stops further GPU trials;
killing a process does not prove cancellation of submitted Metal work. Resource
or budget exhaustion before launch leaves the trial unrun. A published circuit
without completed independent verification remains awaiting verification.
Late results remain evidence but receive no completed-result credit. An overrun
does not establish that no circuit exists, and the internal baseline is not
published early or borrowed from another trial.

Reports distinguish preparation, headroom waiting, supervised execution,
independent verification, binding checks, persistence, and finalization. Existing
GPU and native phase timings describe work inside the supervised invocation;
they are not additional elapsed costs. Intermediate persistence overlaps the
active phase. Per-trial timing ends before its final record write, which remains
charged to overall elapsed time. All 72 independently verified results and
finalization within 900 seconds are required for a complete comparison. The
harness does not retry or launch a second campaign.

The report and measurement use the same finalization timing snapshot, taken
before closing summary, report, measurement, and checksum writes. The overall
budget is checked after those writes. A closing-write overrun records the check's
elapsed time and changes the retained result to `budget-exceeded`.

The frozen protocol records its revision and hash. The original comparison's
eight-/ten-second deadlines remain documented with its retained evidence; its
results are not pooled with measurements using the revised admission policy.

Native construction, materialization, and verification require no Python.
Python is used for this experiment's coordination and independent checking;
Objective-C++ continues to own runtime dispatch and GPU timing.

## FGM-3 retained candidates and additive comparison

The retained-candidate pass uses the existing fixed-factor reduction interface,
`combined`, seed 7 and a 128-reducer/16-round quantum. Preparation and segmented
launch glue stay with the local evidence under `build/fgm3/retained-rescore/`.
The frozen roster has 262 first canonical discoveries admitted by FGM-1's
20-second endpoints. Each invocation runs one 15-minute segment. Continuation
checks source, build, settings and completed circuit bindings; it skips completed
evaluations and records failures separately. Original FGM-1 results are unchanged.

The native workflow runs directly with the existing CLI:

```sh
build/metal/flip_graph --run-config /absolute/path/to/additive-search.json
```

Its configuration extends the existing search config with `workflow:
"additive-search"`, `evaluation`, `pool.elite_capacity`, a positive
`execution.max_batches` and optional `circuit_target`. See the
[native interface](../../docs/development.md#native-additive-search) and the
[frozen comparison settings](../workflow/fixtures/fgm3/protocol.json).
Production execution needs no Python.

After correctness qualification and a reviewed resource envelope, the existing
Python harness can run one comparison from a locally prepared, factor-only
16-parent population:

```sh
python3 benchmarks/workflow/baseline.py --fgm3 \
  --fgm3-population build/fgm3/population \
  --binary-dir build/metal --output build/fgm3/comparison-01
```

Population preparation deduplicates the six calibration inputs canonically,
then fills ten positions by existing seeded collection selection at seed 7.
Selection is independent of retrospective scores. Keep its source bindings,
selection command and hashes with `population.json`. Private input factors and
all generated histories remain local and uncommitted.

Qualification exercises initialization and resume for both selectors at two
workers and four 32-step batches. The fixed reducer quantum does not change.
If complete work exceeds 30 seconds, qualification tries one worker once. A
resource guard failure stops admission. For each qualified chunk, work duration
is `elapsed_seconds - headroom_wait_seconds`. The admission allowance is
`ceil(1.25 * maximum work duration) + 2`. Only explicit prelaunch headroom waiting
is excluded from calibration; configuration, startup, replay, native work,
exports, independent verification and persistence remain included. The measurement
retains total elapsed, headroom waiting and the calibration basis. Invalid
timing observations are rejected rather than clamped.

Headroom waiting still consumes the full arm and global budgets. Admission is
rechecked after waiting, and the memory thresholds, child guards, cleanup
reserves and endpoint credit rules do not change. The allowance is an estimate;
later history growth can increase replay and export costs.

Preparation and qualification precede the measured 900-second envelope. The
comparison has seeds 7/19/41, alternating paired arm order, fresh histories and
90 seconds per arm, with cumulative endpoints at 30/60/90 seconds. Both arms
freshly evaluate the same starting factors and continue bounded work while its
allowance fits. Scores and circuits are never carried between arms.

An endpoint does not shorten the 45-second process guard. Native work, journal
exports, independent verification, headroom waits, startup, replay and evidence
writes all consume the arm budget. Native admission reserves 200 seconds globally
for its child, two bounded read-only exports, bookkeeping and finalization.
The two exports recheck remaining reserves of 150 and 105 seconds respectively. Only complete, independently verified,
durably published results receive endpoint credit. Late or missing verification
is reported separately. Finalization retains 50 seconds globally. Forced
termination or uncertain cleanup stops subsequent GPU admission; an unfinished
matrix or insufficient parent feedback remains incomplete or inconclusive.
Failed chunk attempts retain their elapsed time and available guard evidence.
Arm summaries preserve earlier independently verified endpoint results when a
later chunk fails and omit endpoints not reached before the stop.

An explicit protocol `limits.scan_bytes` applies to native admission and both
read-only exports. It bounds cumulative read work, not allocated memory. Without
it, native admission retains its default and exports use `history.storage_bytes`.
Repeated journal replay can consume much more read work than the journal's size;
the read budget and the 45-second child guard remain independent limits.

Small packaged initialization and resume checks passed for both selectors.
After a full-population resume hit the 3-GiB guard, a separately authorized
qualification passed with a 4-GiB per-call limit and the original prelaunch
ceiling. `guard.run(..., wired_limit_bytes=...)` supports that explicit override;
the default and checked-in comparison protocol remain at 3 GiB. The frozen
qualification is retained under `build/fgm3/qualification-4g-01/`. The first
measured attempt stopped during uniform resume at the
native scan budget. Its earlier verified results and failed attempt are retained
under `build/fgm3/comparison-02/`. A fresh attempt with a 1-GiB read budget,
retained under `build/fgm3/comparison-03/`, completed five arms with 180 verified
evaluations and best cost 55. One arm could not launch within the headroom
admission window, leaving the matrix incomplete. The qualification wait also
increased the frozen chunk allowance to 59 seconds, leaving that comparison
inconclusive.

With headroom waiting excluded from calibration, `build/fgm3/comparison-04/`
completed all six arms in 540.125 seconds, after 75.108 seconds of preparation
and qualification. Its frozen allowance was 23 seconds; all 307 measured
evaluations were independently verified by their endpoints. Best cost remained
55. This completed measurement is separate from the earlier incomplete attempts.
Retained-candidate and qualification results are reported
in [performance notes](../../docs/performance.md#fgm-3-retained-candidates-and-native-search).
