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

## Reproduction and new measurements

Raw measurements, frozen binaries, source snapshots, corpus inputs and verification records for these studies and earlier comparisons are retained privately. This repository provides reusable developer tools, the rank-23 test fixture and both generated 4×4 fixtures, not the complete measurement bundles. Reproducing the recorded comparisons requires the frozen inputs and binaries, the pinned external CPU source and build environment, including OpenMP support. A fresh checkout alone is insufficient.

Use [the benchmark tools](../benchmarks/metal/README.md) for new, separately labeled measurements. Keep source identities, fixtures, commands, raw logs, incomplete runs and timing protocols. Serialize GPU work with the documented process-group and memory supervision. Do not pool incompatible timing methods or replace slow observations selectively.

New execution paths, compiler settings, search policies, hardware or broader performance claims warrant additional measurements after correctness checks. The rates above describe the pinned measured revision; subsequent maintenance changes have not been rebenchmarked.
