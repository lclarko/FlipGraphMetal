# Development

## Architecture and build

`src/metal` contains the C++ controllers, shared arithmetic and transformation code, Metal kernels and Objective-C++ runtime. Controllers manage input, population updates, reporting and export. Search, transformations and additions reduction execute on the Apple GPU through Metal compute pipelines and shared buffers.

`src/common` contains the shared C++ argument parser. Its former CUDA suffixes did not indicate a CUDA dependency. The root build includes parser changes in its dependencies. `reference/cuda` preserves upstream source and its build recipe for reference; CUDA is neither a supported backend nor covered by the Metal validation results.

Normal builds assemble the shared headers and kernels, then compile Metal 3.2 libraries for macOS 15 or later. `METAL_COMPILER` and `METAL_SHADER_FLAGS` select the offline compiler and its options. Shader compilation uses safe math and precise floating-point functions; host compilation disables floating-point contraction. Production signed/F2 libraries and their testing variants are distinct. The signed library contains both packed and general kernels.

Each executable embeds its library's relative filename and SHA-256 digest. The runtime resolves resources from the executable's actual location, including symlink resolution, verifies the bytes and loads them with `newLibraryWithData`. Missing or mismatched libraries fail explicitly. GPU-specific pipeline creation still occurs at runtime.

Build receipts track compiler settings, source location and source contents; unchanged inputs do not trigger recompilation. `make package-metal PACKAGE_DIR=NEW_DIRECTORY` copies the five Metal executables, host-only `scheme_tool`, two libraries, the README's attribution and rights section, and a portable hash manifest. It excludes test programs, source snapshots and benchmark evidence. Move that directory intact; no rebuild is needed merely to run the relocated installation.

For shader experiments without the offline compiler, use `make METAL_LIBRARY_MODE=source`. This explicitly selects runtime compilation through `newLibraryWithSource`, embeds the source checkout's absolute path and requires that source to remain available. Run `make METAL_LIBRARY_MODE=source` after moving that checkout. Running ordinary `make` switches back to compiled libraries and rebuilds the affected programs. There is no automatic fallback between modes.

## Scheme interchange and analysis

Build the host-only tool with `make scheme-tool`. It needs no Metal device. From the checkout root, verify a signed fixture and export it in the CPU coefficient format:

```sh
build/metal/scheme_tool verify \
  --input tests/metal/fixtures/strassen_3x3.txt --format cpu-text --domain ZT \
  --output build/metal/strassen-verified.jsonl
build/metal/scheme_tool export \
  --input build/metal/strassen-verified.jsonl --format jsonl \
  --output-format cpu-text --output build/metal/strassen-export.txt
```

Every output must be a new file. `import`, `verify` and `analyze` emit verified scheme records; `export` selects an interchange format. Input formats are `json`, `jsonl`, a streaming `json-array`, `circuit-json`, `cpu-text`, count-prefixed `metal-search-text` and `metal-minimizer-text`. Text requires an explicit `--domain ZT` or `--domain F2`. JSON may declare the domain or use the existing Boolean `z2` field. Contradictory declarations and unsupported coefficient domains fail. Signed coefficients must be -1, 0 or 1 and satisfy the integer tensor equations; F2 coefficients must be 0 or 1 and satisfy the equations modulo two.

The `fgm-scheme-v1` JSON representation contains `dimensions`, `rank`, `domain`, `orientation`, and dense `u`, `v`, `w` arrays. Its orientation is `cyclic-w`; an explicit `row-major-w` input is converted with a provenance record. `legacy-json` export provides the existing `n`, `m`, `z2`, `u`, `v`, `w` format for current JSON tools. CPU and Metal text exports preserve factor order and signed coefficients. Single-record export formats reject multiple records rather than silently selecting one.

Records distinguish submitted factors, canonical scheme identity and the effective positive-first U/V normalization used for execution eligibility. Identity follows [FGM-CONTRACT-v1](specifications/FGM-CONTRACT-v1.md): sign/order aliases share a canonical identity, while zero terms and multiplicity remain significant. Import does not reduce rank. Tensor validity, search eligibility and signed-reducer eligibility are separate findings; candidate capacities are assessed after the documented execution normalization.

Analysis reports naive addition cost, coefficient counts, zero-factor terms, equal-factor candidate pairs, factor-matrix ranks over Q or F2, and sign-normalization status. These descriptors do not establish scheme equivalence or an optimized addition circuit. Circuit inputs reconstruct their factors, verify the tensor in the declared domain and check the reported operation count. This verification interface does not add an F2 GPU additions reducer.

`factor_ranks` describes assembled rank-by-factor-width matrices. The CPU repository's type invariant instead uses reshaped factors of individual multiplication terms. Buds-based structural signatures and structural/type diversity selection are not implemented. Canonical identity removes term-order and sign-gauge aliases but does not test general equivalence.

Resource limits default to 1 MiB per record, 10 million accounted verification/analysis operations per record, 64 MiB of selection-content accounting and 256 MiB of cumulative input reads. Use `--record-bytes`, `--verification-work`, `--selection-memory` and `--scan-bytes` to change them. Input reads include hashing passes. Selection accounting is not a claim about process RSS. Exit code 2 means a resource limit prevented completion; it does not establish an invalid tensor. Exit code 1 reports malformed or invalid input; 0 reports completed verification and output.

### Read-only external selections

A collection is a JSONL manifest. Each row has schema `fgm-collection-v1`, a common namespace, source presentation ID, relative path, SHA-256, format and domain. Dimensions, rank, metadata and evidence references are optional. Paths must remain within the manifest directory. Multi-record sources require a locator: `{"line": 1}` for a one-based JSONL line, or `{"index": 0}` for a zero-based array/text record. Source IDs are case-sensitive; conflicting duplicate IDs fail.

For example, a row referring to a local scalar fixture has this form. Replace the hash with the SHA-256 of the actual source bytes:

```json
{"schema":"fgm-collection-v1","namespace":"example","id":"scalar-1","path":"scalar.json","sha256":"SOURCE_SHA256","format":"json","domain":"ZT","dimensions":[1,1,1],"rank":1}
```

Select without modifying the collection:

```sh
build/metal/scheme_tool select --input collection/manifest.jsonl \
  --count 8 --seed 7 --output build/metal/selection.jsonl
```

Seeded selection uses the contract's byte-level hash order, independent of manifest enumeration. `--ids ordered-ids.json` instead preserves an explicit JSON array of IDs and conflicts with `--seed`. Optional domain, dimension, rank and namespaced group filters apply before selection. Group filters use `--filter-group namespace:name=value` against a string array at `metadata.groups["namespace:name"]`.

Only selected schemes are admitted and exactly verified. Their source hashes, locators, presentation bindings and optional metadata accompany the output, including aliases with the same canonical identity. Reimport preserves earlier bindings and adds the new artifact binding. Supplied classifications, bounds and evidence are provenance claims; the adapter does not certify them or transfer them to mutated schemes. An external selection is not a Metal journal resume.

### Host checks

`make test-workflow` runs routine host checks for formats, domains, identities, selection, policy boundaries and performance verdicts. `make qualify-workflow` additionally checks inventories of 1,000, 10,000 and 100,000 presentations, builds an isolated scalar adapter from the pinned arithmetic revision, and qualifies the frozen-baseline instrumentation fixtures. These longer checks are required when accepting changes to external selection, the arithmetic reference or baseline profiling, and at the foundation/release gates. Both targets retain a new receipt and log directory under `build/parity/`. The controller reference and its trace fixtures are separate from production kernels; these host checks do not establish GPU or packed/general agreement.

The scalar adapter performs bounded arithmetic calls in C++, including proposal retries and per-proposal outcomes. Its Python bridge checks build identities, transports ordered state and independently replays RNG records. The separate policy reference owns scheduling, captures and restarts; it does not reuse a production controller. Reviewed trace fixtures cover their named cases and are not a claim of complete controlled-kernel validation.

The performance tools in `benchmarks/workflow` retain a pinned production baseline, qualify separate profiling builds and evaluate prospective paired comparisons for changed production paths. Repeatability estimates describe measurement precision, not permitted regression. Regression budgets must be approved independently of candidate observations; a measured slowdown within an approved tolerance remains a reported slowdown. New host utilities instead use repeatable timings, memory measurements and independently verified outputs to assess usability and scaling and establish a baseline. They have no preapproved absolute latency gate. Incomplete or incorrect runs remain failures.

## Search behavior

The general kernels support signed and F2 arithmetic, both complexity minimizers, transformations and resizing. Candidate-list overflow is a persistent per-walk error, even if a later removal or rebuild would fit. A dispatch containing an overflow fails before the host accepts or exports its results. The capacity remains 500 pairs per factor in both general and packed paths. The signed additions reducer supports all seven selection modes, including prefix reuse. Nonzero signed sandwiching is rejected because the inherited implementation is absent; an F2-only tensor is not admitted to the integer reducer.

The compact signed 3×3 path requires block size 32, 1000 fixed inner iterations, plus threshold 1000000000 and zero optional transformation probabilities. Every current and best state must satisfy its representation checks. Eligible inputs can have different ranks; production rank capacity remains 350. Other configurations select the general Metal path.

Packed device terms use nine value bits, nine sign bits and a validity bit in one 32-bit word. Local arithmetic uses 16-bit masks, and device data is interleaved across 32 walks. This specializes storage and execution without changing ordered candidates, RNG consumption, best-state handling or export formats.

Each worker uses xorshift32 with shifts 13, 17 and 5. Worker i starts from `uint32(seed) ^ (0x9e3779b9u * (i+1))`, replacing zero with one. Seed zero selects time-based initialization. These streams are not cuRAND subsequences, and matching seed values do not reproduce CUDA trajectories. The minimizer's inherited host `rand()` refresh is separate from `--seed`.

Tensor arithmetic uses integer coefficients. Metal probabilities and heuristic scores use 32-bit floats; decisions near ties can differ from CUDA. CUDA runtime parity has not been established.

## Input and output conventions

The plaintext formats are distinct:

| Program | Header |
|---|---|
| Flip graph | Scheme count, then a dimension/rank header for each scheme |
| Complexity minimizer | `n1 n2 n3 rank count` |
| Additions reducer | `n1 n2 n3 rank` for one scheme |

Coefficients follow in U, V, W order, with one flattened factor per rank term. W is transposed: output `(i,j)` uses index `j*n1+i`. JSON exports retain the upstream schema and explicit `z2` domain declaration. For file-loaded search populations, each scheme header determines its dimensions and initial rank bound; mixed dimensions remain supported. CLI dimensions describe naive initialization.

The independent Python verifier checks every tensor equation with exact integers, reducing modulo two only for declared F2 schemes. For addition circuits it expands fresh variables and outputs, checks references and operation counts, and rejects invalid tensors. Signs and fanout are free; each fresh pair costs one addition/subtraction, and a k-term output costs `max(k-1,0)`.

```sh
python3 tests/metal/verify.py path/to/export.json
python3 tests/metal/verify.py --reference path/to/input.json path/to/reduced.json
```

The second command also requires exact equality with the reference factors, appropriate to reduction without scheme flips.

## Native controlled workflows

`flip_graph`, `flip_graph_f2` and `additions_reducer` accept `--run-config FILE`.
Existing flags keep their legacy behavior. They cannot be mixed with a run
configuration. `--validate-only` admits inputs, verifies tensors and reports
planned allocations without initializing Metal or creating history. All paths
inside a configuration are relative to that file. Outputs must be new.

For example, this configuration runs a bounded signed search:

```json
{
  "schema": "fgm-run-v1",
  "operation": "search",
  "policy": {
    "schema": "fgm-controlled-config-v1", "policy": "controlled-v1",
    "mode": "alternatives", "domain": "ZT", "seed": 7,
    "dimensions": [3, 3, 3], "collection_rank": 23, "excursion": 2,
    "interval_min": 4, "interval_max": 8, "reduction_q": 0,
    "stagnation_limit": 100, "flip_budget": 100, "control_budget": 150,
    "optional_quota": 2, "proposal_limit": 64, "target_rank": null
  },
  "input": {"kind": "files", "files": [
    {"path": "scheme.txt", "format": "cpu-text", "domain": "ZT"}
  ]},
  "execution": {"workers": 1, "batch_steps": 10, "block_size": 32,
    "backend": "auto", "memory_bytes": 268435456},
  "output": "run.json"
}
```

Use `mode: "rank-reduction"` and `stage_rank` instead of `collection_rank`
for rank reduction. F2 uses the F2 executable and declared domain. Both modes
use fixed dimensions and the versioned [controlled policy](specifications/FGM-CONTRACT-v1.md).
`auto` selects packed execution for signed 3×3 with block size 32; other
configurations use general execution. Explicit `packed` rejects ineligible
configurations. Eligibility, tensor validity and storage capacity are separate.

Optional `pool` settings are `capacity_per_rank`, `reserve_per_rank`,
`memory_bytes`, `stage_threshold` and `selector`. Defaults are 16, 16, 1048576,
1 and `uniform`. `flips` uses the checked sum of effective candidate counts;
zero total weight falls back to uniform selection. Both consume the prescribed
host draw even for one parent. Duplicates do not refresh FIFO order. A full
pool evicts its oldest member. After deterministic input target preflight,
initialization and restarts select active stage-rank parents for rank reduction
or active requested-rank parents for alternatives. Off-rank imports remain
retained inputs. An empty eligible roster refills from bounded stage-entry
reserves without discovery credit. If it is still empty, the run ends with
`no_eligible_parent`. Rank reduction advances at a
completed batch boundary to the lowest lower rank meeting the threshold.
There is no smaller-population fallback. Stage changes and installations use
separate lifetime control credits.

The workflows draw on the CPU implementation, but `FGM-CONTRACT-v1` specifies
their timing, expansion and population policy. Matching CPU parameter values
do not imply matching search behavior. Reference, general and packed checks
establish conformance to the controlled contract; they do not compare search
effectiveness with the CPU implementation.

`batch_steps` and `optional_quota` affect retained output as well as runtime.
Each worker keeps the first eligible optional encounters up to the quota and
one mandatory lowest-rank encounter per batch. Captures can duplicate each
other, and the mandatory encounter can be outside the requested alternatives
rank. A quota of two therefore stores at most three captures, possibly with fewer
distinct qualifying results. Later optional encounters count as drops. Increasing
batch length does not increase the quota. Compare verified committed discoveries
and capture drops with total workflow time when assessing useful output.

`history` accepts `path`, `storage_bytes`, `transaction_bytes` and
`index_memory_bytes`. Defaults are `OUTPUT.journal`, 536870912, 1048576 and
1048576. Reservations cover captures, durable frame completion and bounded
index rebuilding. Resource exhaustion stops the run. `fgm-search-transaction-v1`
stores verified admissions, exact captured presentations, provenance, pool
snapshots, worker accounting and stage changes. The framed, checksummed journal
is authoritative. A single writer acknowledges only synchronized commits.
Recovery preserves incomplete tails and rejects corrupt committed history.
Required `committed-head.json` records the acknowledged sequence, byte offset
and frame hash. Recovery checks the commit head before repair. It rejects missing
history, including a journal truncated at a valid frame boundary. A missing
sorted identity index can be rebuilt. Restore the journal and its commit head
together from backups; a consistent rollback of both needs an independently
retained receipt to detect.

For resume, replace `input` with `{"kind":"resume","journal":"run.json.journal"}`
and choose a new output. Resume re-verifies history, restores pools and stages,
and records new seeded RNG streams. It does not continue an interrupted walker.
`discovery_target` is an optional cumulative alternatives target at the requested
rank. Imports, aliases, reserve refills, rediscoveries and new run IDs add no
discovery credit. Receipts separate current-run and historical counts.

Read-only external selection uses `input.kind: "selection"`, with
`manifest`, `count`, either `seed` or `ids`, and optional `filters` matching
`scheme_tool select`. Search deduplicates canonical seeds while preserving
presentation bindings. Reduction keeps presentation-level work items.

Direct signed reduction sets `operation: "reduce"` and replaces `policy` with
`reduction`: `domain: "ZT"`, `seed`, `rounds`, `reducers`, `schemes`,
`max_flips`, `no_improvements` and `target_additions`. All work is finite;
`execution.workers` equals `schemes`. Zero `max_flips` preserves effective input
factors exactly. Positive `max_flips` enables bounded mutations and reports
attempted and applied flips. Zero `target_additions` disables that stopping
target. The best circuit is verified against its own reconstructed factors and
rank. It is written to `OUTPUT.circuits.jsonl`; the receipt binds its hash and
record indices. Supplied bounds, naive additions and verified circuit costs stay
distinct. Parent classifications are not transferred to changed factors.

Each mutation round starts nonzero lanes from the effective input and applies
bounded flips. This does not implement the CPU optimizer's ongoing walk,
naive-cost or maximize-flips objective, expansion, or sandwiching. The pool's
`flips` selector weights parent draws; it is not a maximize-flips optimizer.

Packaged native analysis and verification need no Python:

```sh
scheme_tool analyze --input run.json.journal --format journal --summary --output corpus.json
scheme_tool analyze --input run.json.journal --format journal --observations \
  --record-bytes 8388608 --output captures.jsonl
scheme_tool verify --input reduction.json.circuits.jsonl --format jsonl --output verified.jsonl
```

Journal input provides read-only corpus access without resuming the run. Summary
groups use the existing descriptors and bounded memory. `--observations` is
available only with `analyze --format journal` and conflicts with `--summary`.
It exports acknowledged captures in transaction and encounter order, including
aliases and duplicates, with `fgm-journal-observation-v1` metadata, a transaction
hash and `fgm-scheme-v1` factors. Each exported capture is checked against its
tensor, scheme and factor identities, domain and workflow dimensions. Empty
capture history produces an empty JSONL file. Neither form repairs the journal.
Metadata is optional.
Run records bind configuration, input identities, executable/library digests,
backend, terminal reasons, work and discovery counters. The complete process
wall time includes verification and persistence. Internal phase clocks cover
narrower parts of the workflow. These records grant no performance regression
allowance.

Verified circuit reports and reduction results include
`verified_circuit_additions_by_stage` with `u`, `v` and `w` counts reconstructed
for each circuit stage; their sum equals the verified total. The Python verifier
returns `additions_by_stage`
for circuit inputs. Top-level run receipts add `admission_microseconds` for
configuration and input admission, `dispatch_microseconds` for host wall time
around dispatch call sites, `verification_microseconds` for search capture checks
or reducer best-circuit reconstruction, and `persistence_microseconds` for journal append
or circuit artifact publication. `setup_microseconds` is the remaining execution
time, including allocation and other host work. Reducer dispatch timing also
wraps `initialize()`, including its host summary calculations and stdout writes;
reducer and counter-buffer allocation occur outside that timer. These fields do
not partition host and GPU time; GPU command times are recorded separately.
They exclude final receipt publication and are narrower than process wall time.
Timers retain elapsed failed dispatch or verification work without changing
failure accounting.

The FGM-1 Python harness additionally reports charged GPU headroom waiting.
Its version 2 and 3 protocols reserve 1216 MiB below the unchanged sampled 3 GiB
system wired-memory cutoff before GPU launch, with deadline checks around
each memory sample. This is an empirical admission rule, not a memory guarantee.
Host verification/export uses the ordinary deadline and process guards.
Version 3 uses 128 reducers in both measured arms; versions 1 and 2 used 256.
Changing this population requires fresh calibration and a separate comparison.

Search receipts retain `fgm-search-accounting-v1` snapshots on failure. `counters`
and `workers` describe the latest fully verified dispatch and completed host
control operations. `committed_counters` records the corresponding totals at
the last acknowledged journal transaction. Discovery counts in both objects
include only acknowledged discoveries; resumed historical counts are retained
even if the new run cannot start. Complete frames beyond the prior durable head
enter those totals only after writable recovery synchronizes the journal and
publishes the recovered head. Work and captures can therefore exceed their
committed counts after a failed append, without earning discovery credit.
`accounting.work_snapshot_batch` and `committed_batches` identify those boundaries.
If dispatch or verification fails, `dispatch_unverified` is true and the receipt
keeps the preceding trusted snapshot instead of reading uncertain device state.

## Testing

Run from the checkout root, with no concurrent GPU workload:

```sh
make test-metal
make smoke-metal
```

These targets create fresh `build/metal/check-*` attempts. Correctness and smoke exports, commands and logs are retained separately for each attempt. A retry must use a new directory. Failed and incomplete runs remain useful evidence and must not be overwritten.

GPU processes are serialized and supervised with a 45-second process-group timeout, sampled 3 GiB system wired-memory cutoff, TERM and then KILL after two seconds. Sampling cannot guarantee a bound on transient memory peaks. The smoke child writes output directly to retained files so termination does not discard already-written logs.

Native correctness covers signed/F2 arithmetic and transformations, ordered candidates and RNG states, and reducer circuits. Smoke checks cover both searches, resizing, both minimizers, reducer prefix reuse and reducer scheme flips. CLI regressions cover malformed inputs and deliberate unsupported-case rejection. Independent tensor verification complements comparisons against C++ versions of the arithmetic.

GPU acceptance requires an identified Apple device, successful arithmetic/dispatch checks and positive finite GPU timings in correctness or search runs. Unavailable GPU access and skipped tests do not count as passes. A macOS CI runner may not expose usable Metal hardware.

For advanced paired smoke checks, select explicit `--binary-dir`, `--signed-fixtures`, `--f2-fixtures`, `--fixture-receipt` and a new `--output` directory. The fixture receipt contains the flat fixture-hash mapping. Baseline and candidate must use the same fixture bytes, checked before each case.

## Changes to execution

For implementation-preserving refactors, compare complete walks against an independently selected frozen reference. Bind the reference source, candidate source, rank capacity and expected kernel explicitly. Require all rounds to match RNG state, ordered candidates, current state, best state, counters and outputs, then independently verify exports. See [the profiling workflow](../benchmarks/metal/README.md).

For compiled-library changes, also select `--gpu-library-mode metallib` when building the matched profile and `--expected-library-mode metallib` when screening it. Source-only comparisons cannot validate the packaged loader or offline compiler. Use `tests/metal/packaging.py --help` for relocation checks with the same signed/F2 fixtures and receipt used by smoke tests. It checks all production programs from another working directory, symlink invocation and explicit failure for missing, corrupt or mismatched libraries.

A passing general-kernel comparison does not establish packed-kernel equivalence. Keep fixtures and source/build identities with the results. Changes to search policy require separate correctness and quality evaluation; throughput alone cannot justify adopting them.

Optional diagnostics retain lazy-selection, sign-aware-reduction, storage and SIMD experiments. Their presence does not change production policy. Earlier bounded experiments did not justify promoting the alternative policies or SIMD cooperation; reevaluate only with an explicit hypothesis, complete-walk checks where applicable and a suitable quality/performance comparison.

## Contributions

Keep behavioral changes separate from source organization and documentation changes. Preserve fixture attribution and required third-party notices. Add tests for changed behavior and update the document that explains it. Performance claims must state hardware, workload, measurement type and limitations.

Project licensing is unresolved as explained in the [README](../README.md#attribution-and-rights). Do not assign a license to inherited material without authority.
