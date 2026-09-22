# Performance

## What was measured

A completed five-fixture study compared signed 3×3 rank-23 application throughput on an 8 GB Apple M1 with seven GPU cores, running macOS 27.0 (26A428). The CPU comparator was the unchanged eight-thread [ternary_flip_graph revision b942f005](https://github.com/dronperminov/ternary_flip_graph/tree/b942f005c61882f678fafb65ba4f9f348f9ef8df). Previous Metal was the production compact implementation before packed terms; packed Metal is the implementation carried into this project.

The selected comparison contains 450 complete processes: 30 per backend for each of five fixtures. Each used 2048 walks, 1000 attempted iterations per walk per round, 33 reports, five seeds and six repetitions per backend. CPU-bracketed ABBA/BAAB panels counterbalanced the order of previous and packed Metal. The measured interval, report 33 minus report 1, contains 65,536,000 attempted iterations.

Timing uses cumulative source elapsed clocks, not stdout arrival times. Printed-clock rounding is bounded conservatively by ±0.011 seconds for CPU and ±0.002 seconds for Metal. GPU command time, application throughput and total process time are different measurements.

An interrupted fixture block was excluded in full and repeated. No timings from that partial block were selected into the reported comparison. Complete fixture blocks span two measurement epochs under the same recorded OS; comparisons remain within each fixture's panels. The desktop environment was not an isolated-hardware experiment.

## Results

The CPU ratio is the median of paired process ratios. Its 95% interval includes conservative printed-clock bounds. The previous-Metal ratio is the geometric mean over counterbalanced panels.

| Fixture | Packed/CPU | Conservative 95% interval | Packed/previous Metal |
|---|---:|---:|---:|
| Original | 1.625 | [1.409, 1.845] | 1.574 |
| Laderman | 2.098 | [2.040, 2.126] | 1.597 |
| Smirnov | 1.125 | [1.102, 1.145] | 1.490 |
| Sun | 1.099 | [1.083, 1.104] | 1.500 |
| CN122 | 1.198 | [1.134, 1.207] | 1.483 |

Packed Metal exceeded the pinned CPU's application throughput on all five tested fixtures under these conditions. A conservative 20% advantage was established only for Original and Laderman. Improvement over previous production Metal was 48.3–59.7% across the five fixtures. Confidence intervals apply per fixture, not simultaneously to all five.

## Interpretation and limits

These are bounded application-throughput findings. CPU lazily samples candidates using MT19937 streams; Metal fully shuffles candidates and uses per-walk xorshift32. Their attempted iterations therefore perform different candidate work. The comparison does not establish equal search quality, equal-budget success rates or a universal rank-23 advantage.

All five starting tensors passed the 729 exact integer equations. Original is a locally generated signed search fixture; the other labels identify Laderman, Smirnov, Sun and CN122 source records. Labels and their published addition counts do not establish addition counts for later search outputs. The five tensors are not a representative sample of all rank-23 schemes.

The selected runs exported no rank improvements. Complete-walk correctness was checked separately, including RNG state, candidate order and best state, with exact-integer verification of exports. Throughput results alone are not correctness evidence.

The packed path also accepts eligible signed 3×3 inputs at other ranks. Earlier bounded rank-26 and rank-27 measurements do not establish a general CPU advantage. Other dimensions use the general Metal kernels. CUDA parity, performance on other Apple GPUs and the cause of historical timing variation remain unestablished.

## Reproduction and new measurements

Original raw measurements, frozen binaries, source snapshots, corpus inputs and verification records are retained privately. This repository provides reusable developer tools and the rank-23 test fixture, not the complete historical measurement bundle. Reproducing the historical comparison requires those frozen inputs, the pinned external CPU source and build environment, including OpenMP support. A fresh checkout alone is insufficient.

Use [the benchmark tools](../benchmarks/metal/README.md) for new, separately labeled measurements. Keep source identities, fixtures, commands, raw logs, incomplete runs and timing protocols. Serialize GPU work with the documented process-group and memory supervision. Do not pool incompatible timing methods or replace slow observations selectively.

Repository organization and documentation changes do not require repeating the 450-process study. New execution paths, compiler settings, search policies, hardware or broader performance claims warrant additional measurements after correctness checks.
