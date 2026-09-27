# Performance

## Signed 3×3 rank 23

### What was measured

A fresh build of FlipGraphMetal at revision `4159df7` was measured on September 22, 2026. The five-fixture study compared signed 3×3 rank-23 application throughput on an 8 GB Apple M1 with seven GPU cores, running macOS 27.0 (26A428). The CPU comparator was the unchanged eight-thread [ternary_flip_graph revision b942f005](https://github.com/dronperminov/ternary_flip_graph/tree/b942f005c61882f678fafb65ba4f9f348f9ef8df). "Previous Metal" was the production compact implementation before packed terms; "packed Metal" is the implementation carried into this project.

The comparison contains 450 complete processes: 30 per backend for each of five fixtures. Each used 2048 walks, 1000 attempted iterations per walk per round, 33 reports, five seeds and six repetitions per backend. CPU-bracketed ABBA/BAAB panels counterbalanced the order of previous and packed Metal. The measured interval, report 33 minus report 1, contains 65,536,000 attempted iterations.

Timing uses cumulative source elapsed clocks, not stdout arrival times. Printed-clock rounding is bounded conservatively by ±0.011 seconds for CPU and ±0.002 seconds for Metal. GPU command time, application throughput and total process time are different measurements.

All 450 confirmation processes completed without failures, restarts or exclusions. Thirty short pilot processes are excluded from the results, and no earlier measurements are pooled into this comparison. The desktop environment was not an isolated-hardware experiment.

### Results

Median paired application-throughput gains over the pinned eight-thread CPU ranged from **12.5% to 26.6%** across the five tested rank-23 fixtures under the recorded conditions.

The CPU ratio is the median of paired process ratios; the gain column expresses that ratio as a percentage increase. Confidence intervals use paired ratios adjusted downward for conservative printed-clock bounds, so they describe a conservative estimate rather than the unadjusted median. The previous-Metal ratio is the geometric mean over counterbalanced panels.

| Fixture | Packed/CPU | Median gain over CPU | 95% interval for conservative lower ratios | Packed/previous Metal |
|---|---:|---:|---:|---:|
| Original | 1.231 | 23.1% | [1.205, 1.232] | 1.604 |
| Laderman | 1.266 | 26.6% | [1.211, 1.342] | 1.544 |
| Smirnov | 1.133 | 13.3% | [1.114, 1.137] | 1.527 |
| Sun | 1.125 | 12.5% | [1.109, 1.122] | 1.528 |
| CN122 | 1.202 | 20.2% | [1.184, 1.203] | 1.497 |

Each fixture's conservative 95% ratio interval lies above parity with the pinned CPU. Confidence intervals apply per fixture, not simultaneously to all five. Improvement over previous production Metal was 49.7–60.4% across the five fixtures.

### Interpretation and limits

These are bounded application-throughput findings. CPU lazily samples candidates using MT19937 streams; Metal fully shuffles candidates and uses per-walk xorshift32. Their attempted iterations therefore perform different candidate work. The comparison does not establish equal search quality, equal-budget success rates or a universal rank-23 advantage.

All five starting tensors passed the 729 exact integer equations. Original is a locally generated signed search fixture; the other labels identify Laderman, Smirnov, Sun and CN122 source records. Labels and their published addition counts do not establish addition counts for later search outputs. The five tensors are not a representative sample of all rank-23 schemes.

The timed searches exported no rank improvements. Separate complete-walk checks used 33 schemes, six rounds and seed 7 per fixture, matching the frozen reference for RNG state, candidate order, current and best state, and outputs. All 165 correctness exports passed independent exact-integer verification. The 300 timed Metal processes recorded 9,900 positive packed-kernel GPU timings on the Apple M1. Throughput results alone are not correctness evidence.

The packed path also accepts eligible signed 3×3 inputs at other ranks. Earlier bounded rank-26 and rank-27 measurements do not establish a general CPU advantage. Other dimensions use the general Metal kernels. CUDA parity, performance on other Apple GPUs and the cause of historical timing variation remain unestablished.

## Signed 4×4

A separate September 22, 2026 comparison used the same Apple M1 and pinned eight-thread CPU, with a fresh FlipGraphMetal build from the same production revision. It starts from two generated signed schemes: schoolbook multiplication at rank 64 and the tensor square of Strassen's seven-product algorithm at rank 49. The fixtures, generator and provenance are included in this repository.

The comparison contains 120 complete processes: 30 per backend per fixture, using 2048 walks, 1000 attempted iterations per walk per round, five seeds and six repetitions. CPU/Metal order alternates across repetitions. Timing uses report 17 minus report 1, or 32,768,000 attempted iterations, with the same source-clock rounding bounds described above. Six-report pilots led to the shorter measurement window to preserve timeout headroom; all 16 pilot processes are excluded. The confirmation protocol was fixed before its measurements, and all 120 processes completed without restarts or replacements.

The pinned CPU had higher application throughput on both fixtures. Rates below are medians in millions of attempted iterations per second. Ratios are medians of paired Metal/CPU rates, so they need not equal the quotient of the two rate medians.

| Starting fixture | CPU Miterations/s | Metal Miterations/s | Metal/CPU | 95% interval for conservative lower ratios |
|---|---:|---:|---:|---:|
| Naive, rank 64 | 21.629 | 3.722 | 0.173 | [0.163, 0.190] |
| Strassen tensor square, rank 49 | 26.006 | 1.461 | 0.057 | [0.054, 0.058] |

All 60 Metal processes used `randomWalkKernel`, with 1020 positive GPU command timings on the Apple M1. The packed signed 3×3 path does not apply to 4×4. These results establish a general-path throughput baseline for the two fixtures, not a performance claim for other dimensions, starting schemes or Apple GPUs.

Separate complete-walk checks covered populations 33 and 512 with two seeds and six rounds for each fixture. Current and best states, RNG, ordered candidates, counters and outputs matched the frozen reference; all 2180 exports passed independent reconstruction of the 4096 exact-integer tensor equations. Every round showed changed states from both starting fixtures. The 60 exports from the timed confirmation processes also passed independent tensor verification.

Ranks 64 and 49 describe starting schemes. Failed-flip recovery can expand a scheme even with optional expansion disabled, and ranks can change during search. CPU and Metal use different candidate-selection and RNG policies, so attempted iterations are not equal candidate work or equal search quality. CPU termination is asynchronous; exported improvements can include work beyond the measured interval. The shorter window and evolving ranks also prevent treating these rates as interchangeable with the 33-report 3×3 study. Confidence intervals apply per fixture.

## Maintenance revision comparison

A September 23, 2026 regression check compared fresh builds of `4159df7` and `45042b7` on the same Apple M1, using matching compiler settings. The newer revision adds persistent candidate-capacity checks and corrects input and build handling. This comparison measures Metal against Metal; it does not update the CPU comparisons above.

The check contains 56 complete measured processes: two seeds (7 and 19), two repetitions and both revisions for each fixture. Each seed runs previous/current followed by current/previous. All runs use 2048 walks and 1000 attempted iterations per walk per round. Fourteen six-report pilot processes are excluded. The predeclared time-headroom rule selected 33 reports for rank 23, 17 for naive 4×4 and six for Strassen 4×4. Both revisions use the same window within each fixture, measured from cumulative source report 1 to the final report.

Changes below are geometric means of four paired current/previous throughput ratios. The range shows individual paired changes, not a confidence interval.

| Fixture | Throughput change | Individual paired changes |
|---|---:|---:|
| Original, 3×3 rank 23 | +0.2% | -1.4% to +1.6% |
| Laderman, 3×3 rank 23 | +3.7% | +0.5% to +7.1% |
| Smirnov, 3×3 rank 23 | +1.2% | -0.3% to +3.1% |
| Sun, 3×3 rank 23 | +3.7% | +1.9% to +5.5% |
| CN122, 3×3 rank 23 | +2.4% | +1.7% to +3.1% |
| Naive, 4×4 rank 64 | +2.6% | -10.9% to +17.5% |
| Strassen tensor square, 4×4 rank 49 | +0.5% | +0.1% to +1.1% |

No consistent throughput loss appeared in this screen. Two seeds and four pairs per fixture do not establish performance equivalence or a general speedup. Naive 4×4 showed substantial variation between pairs; its small aggregate increase should not be treated as a demonstrated improvement. Strassen 4×4 covers a shorter search window than the earlier CPU comparison.

All 1504 expected search dispatches recorded positive GPU timings on Apple M1. Eight exports from the measured runs passed independent exact-integer tensor verification. Every process remained within the 45-second and sampled 3 GiB wired-memory limits.

## Native scheme utilities

On September 24, 2026, `scheme_tool` at revision `3b29651` was measured on an 8 GB Apple M1 running macOS 27.0, using the default C++17 `-O2` build. Six repetitions of each workload completed, comprising 30 workflows and 84 native processes. Every returned presentation passed independent domain-correct tensor and identity checks. No attempts failed or were excluded.

Complete-workflow time includes process supervision and independent verification. Inputs are generated and hashed before timing. The native-time column is the mean of the child-process elapsed counters, which are rounded to 0.01 seconds; several short interchange processes are below that resolution. Peak RSS is the maximum observed native-process value, excluding the benchmark controller and independent verifier.

| Workflow | Approximate native time | Mean complete workflow | Observed workflow range | Peak native RSS |
|---|---:|---:|---:|---:|
| Verify/report 100 signed/F2 presentations | 0.09 s | 1.069 s | 1.001 to 1.226 s | 4.81 MiB |
| Five CPU-text round trips | Resolution-limited | 0.281 s | 0.234 to 0.455 s | 3.42 MiB |
| Select 3 from 1,000 records | 0.02 s | 0.054 s | 0.051 to 0.059 s | 2.41 MiB |
| Select 3 from 10,000 records | 0.16 s | 0.209 s | 0.196 to 0.258 s | 3.88 MiB |
| Select 3 from 100,000 records | 1.63 s | 1.670 s | 1.600 to 1.825 s | 17.69 MiB |

Reporting and interchange use the public signed rank-23 3×3, signed Strassen-derived 4×4 and F2 3×3 fixtures, plus generated signed/F2 rectangular 2×3×4 schemes. The report repeats those five presentations 20 times. Each selection inventory contains distinct presentation IDs referring to one scalar source; seed 7 selects three records. This measures manifest handling with small selected inputs, not verification of an entire library or the cost of richer metadata and larger scheme records.

These workloads completed quickly enough for interactive verification and short preprocessing on this host. Independent verification accounted for about 0.94 seconds of the 1.07-second reporting workflow; it is benchmark work outside the native executable. Inventory selection time and memory increased with inventory size. Six repetitions on a normal desktop establish descriptive baselines, not statistical performance verdicts, cold-cache guarantees or results for other hardware. No Metal kernels ran. Existing production-path regression requirements remain separate.

Reproduce these host workloads with `make scheme-tool` and the native-host measurement command in [the benchmark instructions](../benchmarks/metal/README.md). They require only repository fixtures and generated inventories, without private inputs or the external CPU project.

## FGM-1 effectiveness baseline

The [FGM-1 protocol and commands](../benchmarks/metal/README.md#fgm-1-short-run-effectiveness-baseline)
completed one baseline on the 8 GB Apple M1 MacBook Air under macOS 27.0.
Protocol version 3 used 128 reducers, 16 calibrated rounds and two search
workers. All 30 arms completed in 756.706 seconds, including 45.392 seconds
of final verification and artifact hashing. Supplied-circuit verification
preceded the timed envelope. No second successful run was performed.

The retained evidence is in `build/fgm1/baseline-04/`: frozen sources, binaries,
panel, calibration, histories, configurations, native receipts, circuits,
logs, `measurement.json`, `summary.json`, `report.md` and `artifacts.json`.
Independent replay reproduced the summary exactly and checked all 11,301
artifact hashes. The calibration also validates for reuse against its retained
bindings. Build, input, hardware or protocol changes require a separate run.

| Start | Naive factor count | Fixed reduction at 20 s | Generation + reduction at 20 s | Verified supplied reference |
|---|---:|---:|---:|---|
| Original | 113 | 69 | 66 | unavailable in this panel |
| Laderman | 98 | 62 | 62 | unavailable in this panel |
| Smirnov | 84 | 68 | 68 | unavailable in this panel |
| Sun | 120 | 58 | 58 | 56 |
| CN122 | 122 | 58 | 58 | 55 and 58 |

Each reported best was reached in all three seeds and was already present at
10 seconds. No completed arm improved its best between 10 and 20 seconds.
The verified Sun-56 and CN122-55 supplied circuits were not rediscovered.
CN122's 58-addition reference cost was reconstructed from factors alone.
Reference verification is separate from these reconstruction outcomes.

Generation retained 262 distinct alternative canonical identities across the
15 arms by their 20-second endpoints; 246 received an evaluation by a
20-second endpoint. Summing each arm's distinct discoveries instead gives 304, up from
178 at 10 seconds. There were 300 evaluated candidates counted per arm,
including the 15 starting inputs, and 19 pending discoveries at 20 seconds.
The complete per-start, per-seed histograms and U/V/W counts are in the summary.
Canonical identity removes term-order and sign-gauge aliases only; it does not
establish general mathematical inequivalence.

The observed constraints differ by stage:

- **Generation:** Original consistently benefited. Laderman retained no new
  canonical rank-23 candidate under this policy. Smirnov, Sun and CN122 generated
  alternatives without improving the starting factors' best reduced cost.
- **Retention:** completed generation-arm execution, including late work,
  recorded 749 optional captures, 443 optional duplicates (59.1%), 335 optional
  seed rediscoveries, 380 mandatory/optional redundant events and 6,080 dropped
  encounters. FIFO replay implies 143 active-pool evictions; journal history
  preserves those identities. Drops do not identify lost novel schemes.
- **Circuit quality:** Sun's reconstructed U/V/W costs were 14/14/30, versus
  13/13/30 in its verified 56-addition reference. CN122 reconstructed 14/15/29,
  versus 13/14/28 in its 55-addition reference. These locate observed gaps in
  this reducer quantum without claiming that longer reduction cannot close them.
- **Execution overhead:** the generation arms took 305.106 seconds including
  late completions, with 125.954 seconds of GPU time. Search blocks consumed
  128.729 seconds and reduction blocks 175.844 seconds. Independent verification
  took 31.214 seconds, observation export 27.322 seconds, and headroom waiting
  3.342 seconds. These narrower timings overlap the block totals. Fixed arms
  took 303.161 seconds with 202.512 seconds of GPU time. Repeated admission,
  history replay, startup and verification are material under 20-second windows.

Twenty-five circuit evaluations completed after their arm's 20-second endpoint
and received no endpoint credit. The maximum sampled wired memory during
scored-arm execution was 2.624 GiB; no guard failed. Samples are not a continuous
peak-memory bound.

Three earlier attempts remain incomplete and are excluded from these findings:
`baseline-01` and `baseline-02` used version 1 with 256 reducers; `baseline-03`
used version 2 with 256 reducers and charged headroom waiting. They stopped at
the unchanged 3 GiB guard after zero, 25 and five completed arms respectively.
Version 3 reduced the population to 128, retained the
hard guards and recalibrated. These versions are not repeatability samples.
Longer-run effectiveness and variation between complete repeat runs remain
unmeasured.

## FGM-2 calibration coverage

FGM-2 compares a fixed direct-reduction quantum with that same work plus GPU
construction extensions. The retained reference audit replayed supplied gates;
it did not run a constructor to fill evidence gaps. Public references were
checked against their submitted ordered factors and again after the existing
admission sign normalization. W orientation conversion is separate from any
algebraic basis change.

| Reference | U coverage | V coverage | Wᵀ coverage before construction |
|---|---|---|---|
| Local private 55 | One auxiliary, 14 gates | One auxiliary, 13 gates | Retained 14-gate witness; emitted W verified at 28 |
| CN122-55 | One auxiliary, 13 gates | One auxiliary, 14 gates | Not established by its direct-W witness |
| Sun-56 | One auxiliary, 13 gates | Supplied 13-gate witness uses two auxiliaries; alternative unresolved | Not established |
| CN122-58 | Same factors as CN122-55, whose witness covers U13 | Same factors as CN122-55, whose witness covers V14 | Not established |
| Original, Laderman, Smirnov | No optimized reference established | No optimized reference established | No optimized reference established |

The supplied CN122-58 U14/V15 witnesses each use two auxiliaries. A supplied
witness outside the family does not exclude another one-auxiliary realization.
Reference costs are achieved costs, not assumed stagewise minima. Complete
restricted-family failure on a covered component requires investigation;
unknown reference-family membership remains an unresolved capability question.
Private witnesses and their derived factors stay local and uncommitted.

## FGM-2 first bounded comparison

The single attempt at implementation revision `2d3d002` is **incomplete**:
21 timely independently verified results, one terminated invocation, and 50
unrun trials. The recorded envelope ended after 27.920 seconds. Evidence is
retained locally under `build/fgm2/comparison-01/`; no automatic retry ran.
The host was an 8-GB Apple M1 MacBook Air on macOS 27.0.

| Input | Completed paired seeds | Baseline | Transpose | Cancellation | Combined |
|---|---|---:|---:|---:|---:|
| Original | 7, 19, 41 | 69 | 69 | 67 | 66 |
| Laderman | 7, 19 | 62 | 62 | 62 | 62 |

Original's baseline U/V/W costs were 20/16/33; combined achieved 19/15/32.
Laderman stayed at 16/16/30. Laderman seed 41 also has a verified combined
result at 62, but its standalone baseline pair is missing. Smirnov, Sun,
CN122, and the private input have no measured cells in this attempt.
All 21 credited circuits passed a further exact replay against independently
normalized frozen factors; shared baseline stage counts match within every
observed input/seed group.

Across these incomplete observations, median incremental extension time was
50.9 ms for transpose, 12.5 ms for cancellation, and 79.8 ms for combined.
Median recorded trial times were 656, 698, 649, and 730 ms for baseline,
transpose, cancellation, and combined respectively. These are descriptive
observations of fixed work plus extensions, not a matched-time comparison or
a complete-panel performance conclusion. Peak sampled system wired memory was
2,698,952,704 bytes, below the unchanged 3-GiB cutoff.

The stopped Laderman seed-41 baseline trial consumed 8.019 seconds, but its
native supervisor lasted only 0.098 seconds. Its child log establishes no GPU
dispatch. About 7.9 seconds elapsed before supervision, which includes hashing,
configuration/evidence writes, and headroom waiting. The exact wait contribution
was not separately recorded. The inherited launch policy reserved 1,275,068,416
bytes below the hard cutoff, yielding a 1,946,157,056-byte launch threshold.
This is evidence of a prelaunch-budget limitation, not an eight-second GPU
reduction or a hard-memory-limit failure. Forced termination stopped later GPU
admission, and the failed invocation published no circuit.

Separate fixed-factor correctness qualification reconstructed 55 from both
public CN122 and the private calibration input:

| Qualification input | Reducers / rounds / seed | Baseline U/V/W | Combined U/V/W |
|---|---|---|---|
| Public CN122 | 32 / 2 / 7 | 14/15/29 = 58 | 13/14/28 = **55** |
| Local private 55 | 128 / 16 / 7 | 15/14/29 = 58 | 14/13/28 = **55** |

Both passed native and independent Python verification against the unchanged
input factors. CN122 ran through the relocated package with Python absent from
the production process's PATH. Its combined result establishes an observed
one-auxiliary Wᵀ construction and verified 28-addition W, extending the
pre-construction coverage audit above. These are factors-only reconstruction
results, separate from verification of supplied circuits. Their different
qualification settings are not pooled, and neither fills an unrun comparison
cell or establishes repeatability.

The retained qualification checks include 18 GPU/oracle cases with 1,460
candidate witnesses, the 33-direction/24-gate boundary, Python-free packaged
operation, and native signed/F2 regressions. The host workflow suite passed
220 tests.

These results establish useful cancellation construction on Original and covered
55 reconstruction during qualification. This attempt did not establish full-panel
results.

## FGM-2 completed fixed-work comparison

The next single attempt, at revision `6a0af27`, completed all **72 independently
verified trials in 60.884 seconds**. Evidence is retained locally under
`build/fgm2/comparison-02/`. The native executable and signed Metal library were
byte-identical to the previously qualified build. The six inputs, seeds 7/19/41,
strategy order, and 128-reducer, 16-round baseline work were unchanged.

Every seed produced the following verified totals:

| Input | Baseline | Transpose | Cancellation | Combined |
|---|---:|---:|---:|---:|
| Original | 69 | 69 | 67 | 66 |
| Laderman | 62 | 62 | 62 | 62 |
| Smirnov | 68 | 68 | 68 | 68 |
| Sun | 58 | 58 | 57 | 57 |
| CN122 | 58 | 58 | 56 | 55 |
| Local private 55 | 58 | 58 | 56 | 55 |

Combined construction improved four of the six inputs. CN122 reached U13/V14/W28
and the private input reached U14/V13/W28. Original reached U19/V15/W32; Sun
reached U13/V14/W30 and did not attain its supplied 56-addition reference.
Transpose alone did not improve any total. The combined route improved W by
one addition on Original, CN122, and the private input through transposed
restricted construction.

The restricted constructor exhausted its family on all three Laderman maps,
Smirnov V/Wᵀ, and Sun V. These are results within the declared family, not general
circuit lower bounds. All known covered calibration components were reconstructed.
A further independent replay checked all 72 circuits against frozen input
factors and confirmed matching baseline stage counts in all 18 input/seed groups.

Headroom waiting totaled 7.478 seconds; the longest wait was 3.080 seconds.
No process was terminated, and no trial missed verification. The maximum
supervised native and verifier times were 1.066 and 0.201 seconds. Median
incremental extension times were 51.8 ms for transpose, 12.1 ms for cancellation,
and 70.2 ms for combined. Median recorded trial times were 686, 815, 683, and
822 ms respectively for baseline, transpose, cancellation, and combined, with
18 observations each. These are fixed-work observations, not matched-time
effectiveness results. Peak sampled wired memory was 2,778,103,808 bytes, below
the unchanged 3-GiB cutoff.

The longest trial was 3.951 seconds, so this attempt did not encounter the long
wait that stopped the earlier one. Host conditions differed; completion alone
does not isolate the effect of the timing change. Injected-clock tests cover
waits and valid work beyond the old deadlines. The two protocols remain separate
measurements. No 54-addition circuit was found, and variation between repeated
complete campaigns remains unmeasured. No further campaign was launched.

## Reproduction and new measurements

Raw measurements, frozen binaries, source snapshots, corpus inputs and verification records for the CPU/GPU studies and earlier comparisons are retained privately. This repository provides reusable developer tools, the rank-23 test fixture and both generated 4×4 fixtures, not the complete measurement bundles. Reproducing the recorded comparisons requires the frozen inputs and binaries, the pinned external CPU source and build environment, including OpenMP support. A fresh checkout alone is insufficient.

Use [the benchmark tools](../benchmarks/metal/README.md) for new, separately labeled measurements. Keep source identities, fixtures, commands, raw logs, incomplete runs and timing protocols. Serialize GPU work with the documented process-group and memory supervision. Do not pool incompatible timing methods or replace slow observations selectively.

New execution paths, compiler settings, search policies, hardware or broader performance claims warrant additional measurements after correctness checks. Each comparison above applies to its pinned revisions and recorded workload.

## FGM-3 retained candidates and native search

The retrospective pass completed **262/262** first retained canonical FGM-1
alternatives in one 214.645-second segment. The roster came from the completed
`baseline-04` discovery records at their 20-second endpoints, with 196 relevant
receipt-bound journal prefixes replayed natively. It was not selected by circuit
cost or by enumerating the candidate directory. Original endpoint results were
preserved.

Each factor-only presentation received one `combined` evaluation at seed 7,
128 reducers, 16 rounds, `no_improvements=16`, one scheme, no flips and reducer
target zero. All circuits passed native and independent exact factor/count
verification.

| Starting family | Alternatives | Best achieved additions |
|---|---:|---:|
| Original | 58 | 62 |
| Laderman | 0 | unavailable |
| Smirnov | 55 | 66 |
| Sun | 73 | 59 |
| CN122 | 76 | 56 |

Against the 246 candidates with timely verified FGM-1 costs, 209 improved,
37 tied and none worsened. Sixteen had no timely old cost. Against the direct
baseline phase of these new invocations, 223 improved and 39 tied. Final totals
ranged from 56 to 73; two reached 56. No 54 was found. These are reconstruction
results for a fixed retained collection, not evidence that cost-guided selection
improves search.

The private reproducibility bundle is `build/fgm3/retained-rescore/`, with frozen
inputs and builds, source/presentation bindings, circuits, logs, `audit.json`
and a report. An initial sandbox launch failed before Metal dispatch; its
explicit infrastructure retry and original failure remain recorded. No completed
evaluation was repeated. Peak sampled wired memory in the successful segment
was 2,841,739,264 bytes.

The first native additive-search GPU smoke used public CN122, two packed
workers, four bounded batches and the same 128-reducer/16-round evaluation
quantum. Its test stagnation limit was 10, to exercise restart feedback. The
3-GiB system wired-memory guard terminated it after 7.903 seconds: memory rose
from 1,833,369,600 to a sampled 3,240,722,432 bytes. Process cleanup completed;
this does not prove cancellation of all submitted Metal work. Further GPU
admission stopped and the shared-population qualification/comparison was not
launched.

Eight evaluations had already committed. Read-only native replay and independent
exact verification checked all eight, including CN122 at U13/V14/W28 = 55.
The invocation published no final circuit artifact or complete receipt. These
committed circuits are partial correctness evidence, with no endpoint or
comparison credit. The source/build snapshot, journal, guard log and partial
verification are retained under `build/fgm3/gpu-smoke/`.

The final host suite passed 255 tests with no failures or skips; the independent
verifier's four focused tests also passed. Native programs and the package built,
and relocated host commands plus additive preflight passed with Python absent
from PATH. Host tests cover score binding, population selection, resume,
transaction failures and malformed histories. Full GPU qualification, resumed GPU search,
packaged additive execution and matched-time selection benefit remain
**unverified**. No conclusion about uniform versus cost-diverse selection follows
from this smoke or from the retrospective reconstruction pass.
