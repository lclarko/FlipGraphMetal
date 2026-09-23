# Development

## Architecture and build

`src/metal` contains the C++ controllers, shared arithmetic and transformation code, Metal kernels and Objective-C++ runtime. Controllers manage input, population updates, reporting and export. Search, transformations and additions reduction execute on the Apple GPU through Metal compute pipelines and shared buffers.

`src/common` contains the shared C++ argument parser. Its former CUDA suffixes did not indicate a CUDA dependency. The root build includes parser changes in its dependencies. `reference/cuda` preserves upstream source and its build recipe for reference; CUDA is neither a supported backend nor covered by the Metal validation results.

Normal builds assemble the shared headers and kernels, then compile Metal 3.2 libraries for macOS 15 or later. `METAL_COMPILER` and `METAL_SHADER_FLAGS` select the offline compiler and its options. Shader compilation uses safe math and precise floating-point functions; host compilation disables floating-point contraction. Production signed/F2 libraries and their testing variants are distinct. The signed library contains both packed and general kernels.

Each executable embeds its library's relative filename and SHA-256 digest. The runtime resolves resources from the executable's actual location, including symlink resolution, verifies the bytes and loads them with `newLibraryWithData`. Missing or mismatched libraries fail explicitly. GPU-specific pipeline creation still occurs at runtime.

Build receipts track compiler settings, source location and source contents; unchanged inputs do not trigger recompilation. `make package-metal PACKAGE_DIR=NEW_DIRECTORY` copies the five production executables, their two libraries, the README's attribution and rights section, and a portable hash manifest. It excludes test programs, source snapshots and benchmark evidence. Move that directory intact; no rebuild is needed merely to run the relocated installation.

For shader experiments without the offline compiler, use `make METAL_LIBRARY_MODE=source`. This explicitly selects runtime compilation through `newLibraryWithSource`, embeds the source checkout's absolute path and requires that source to remain available. Run `make METAL_LIBRARY_MODE=source` after moving that checkout. Running ordinary `make` switches back to compiled libraries and rebuilds the affected programs. There is no automatic fallback between modes.

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
