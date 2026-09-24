# Development

## Architecture and build

`src/metal` contains the C++ controllers, shared arithmetic and transformation code, Metal kernels and Objective-C++ runtime. Controllers manage input, population updates, reporting and export. Search, transformations and additions reduction execute on the Apple GPU through Metal compute pipelines and shared buffers.

`src/common` contains the shared C++ argument parser. Its former CUDA suffixes did not indicate a CUDA dependency. The root build includes parser changes in its dependencies. `reference/cuda` preserves upstream source and its build recipe for reference; CUDA is neither a supported backend nor covered by the Metal validation results.

Normal builds assemble the shared headers and kernels, then compile Metal 3.2 libraries for macOS 15 or later. `METAL_COMPILER` and `METAL_SHADER_FLAGS` select the offline compiler and its options. Shader compilation uses safe math and precise floating-point functions; host compilation disables floating-point contraction. Production signed/F2 libraries and their testing variants are distinct. The signed library contains both packed and general kernels.

Each executable embeds its library's relative filename and SHA-256 digest. The runtime resolves resources from the executable's actual location, including symlink resolution, verifies the bytes and loads them with `newLibraryWithData`. Missing or mismatched libraries fail explicitly. GPU-specific pipeline creation still occurs at runtime.

Build receipts track compiler settings, source location and source contents; unchanged inputs do not trigger recompilation. `make package-metal PACKAGE_DIR=NEW_DIRECTORY` copies the five production executables, their two libraries, the README's attribution and rights section, and a portable hash manifest. It excludes test programs, source snapshots and benchmark evidence. Move that directory intact; no rebuild is needed merely to run the relocated installation.

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
