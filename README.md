# FlipGraphMetal

An Apple silicon Metal port of [Perminov's FlipGraphGPU](https://github.com/dronperminov/FlipGraphGPU), with performance tuning focused on signed 3×3 matrix-multiplication searches.

FlipGraphMetal retains the upstream search and transformation functionality while providing a Metal backend, independent correctness checks and a packed GPU execution path. It is maintained independently; CUDA is retained as unsupported reference source.

## Build and run

Requirements: Apple silicon, macOS 15 or later, and Xcode Command Line Tools. Python 3.9+ is needed for tests and developer tools, not for building or running the production programs. CUDA and third-party GPU libraries are not required.

From the checkout root:

```sh
make
# Equivalent explicit target:
make metal
```

The build produces:

| Executable | Function |
|---|---|
| `build/metal/flip_graph` | Signed search |
| `build/metal/flip_graph_f2` | F2 search |
| `build/metal/complexity_minimizer` | Signed complexity minimization |
| `build/metal/complexity_minimizer_f2` | F2 complexity minimization |
| `build/metal/additions_reducer` | Signed additions reduction |

A small bounded search:

```sh
./build/metal/flip_graph -n1 3 -n2 3 -n3 3 \
  --schemes 8 --block-size 4 --max-iterations 100 \
  --resize-probability 0 --rounds 3 --seed 7 --path build/metal/example
```

Use `--help` for each program's options. Positive `--rounds` bounds outer iterations; `--max-iterations` bounds inner search work. With `--rounds 0`, search is unbounded and the minimizer and reducer retain their own stopping conditions. Choose a new output path when retaining results.

Shaders compile at runtime from the source directory embedded during the build. Keep the checkout available and rebuild after moving it. Copying an executable alone does not produce a standalone installation. An unavailable Apple GPU is an error; there is no CPU fallback.

## Supported functionality

FlipGraphMetal ports FlipGraphGPU's signed and F2 search, scheme transformations, complexity minimizers and signed additions reducer. Projection, extension, direct sums, tensor products and randomized resizing are supported. F2 search also supports sandwiching; nonzero signed sandwiching is deliberately rejected. There is no F2 additions reducer.

Dimensions are limited to 1 through 16, factors to 64 elements and rank to 350. Naive initialization additionally requires the product of the dimensions to fit within rank 350. Malformed inputs and unsupported configurations fail explicitly.

Eligible signed 3×3 searches use a specialized packed path across ranks. Other supported dimensions and configurations use general Metal kernels. See [development](docs/development.md) for eligibility and numeric behavior, and [performance](docs/performance.md) for measured results and their limits.

## Tests

```sh
make test-metal
make smoke-metal
```

These targets require Python 3.9+ and access to an actual Apple GPU. They retain isolated outputs and bounded execution logs. See the [testing guidance](docs/development.md#testing) before running other GPU work concurrently.

## Scope and future work

Potential future work includes newer CPU pool, restart and meta-search strategies; Z3 support and lifting tools; and additional analysis tools and search metrics. These are separate from the existing projection, extension, tensor-product and resizing support. The initial project scope preserves current search behavior and focuses maintenance on Apple silicon Metal.

## Attribution and rights

The inherited implementation is by Perminov and the contributors to FlipGraphGPU. Upstream ancestry is retained from commit `ca8ac1c09715f77bdedd61459a8c3e013d7718b7`. The Metal backend, validation tools and GPU-specific optimization work were developed for this port by lclarko.

Perminov [stated that he had no objection to the fork](https://github.com/dronperminov/FlipGraphGPU/issues/1#issuecomment-5764325910), but did not specify a project license. Project licensing remains unresolved; this acknowledgment is not a license to relicense inherited code. The MIT notice accompanying the separate reduction test helper applies only to that material. Preserve its notice and provenance.
