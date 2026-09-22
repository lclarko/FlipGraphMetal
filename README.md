# FlipGraphMetal

An Apple silicon Metal port of [Perminov's FlipGraphGPU](https://github.com/dronperminov/FlipGraphGPU), maintained independently with performance tuning focused on signed 3x3 searches.

This import retains upstream ancestry and ports signed and F2 search, scheme transformations, both complexity minimizers and the signed additions reducer. Projection, extension, direct sums, tensor products and resizing are supported. CUDA source remains available as reference; CUDA runtime parity has not been tested.

Build with `make metal` on Apple silicon with macOS 15 or later and Xcode Command Line Tools. The five programs are built under `build/metal`; use their `--help` options. Python 3.9+ is needed for tests and developer tools. Shaders compile at runtime from the source directory embedded during the build; rebuild after moving the checkout.

The inherited implementation is by Perminov and the contributors to FlipGraphGPU. The Metal implementation, validation and optimization work were developed for this port by lclarko. Original internal development records are privately retained.

Perminov [stated that he had no objection to a fork](https://github.com/dronperminov/FlipGraphGPU/issues/1#issuecomment-5764325910), without specifying a project license. Project licensing remains unresolved. The separate reduction test helper retains its MIT license and provenance; that license does not cover the project as a whole.
