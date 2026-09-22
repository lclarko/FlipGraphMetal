.DEFAULT_GOAL := metal

METAL_FLAGS = -mmacosx-version-min=15.0 -std=c++17 -O2 -fobjc-arc -ffp-contract=off -framework Foundation -framework Metal -DMETAL_SOURCE_DIR='"$(CURDIR)/src/metal"'
METAL_SOURCES = $(wildcard src/metal/*.h src/metal/*.cpp src/metal/*.metal) src/metal/runtime.mm src/entities/arg_parser.cu src/entities/arg_parser.cuh
.PHONY: metal
metal: build/metal/flip_graph build/metal/complexity_minimizer build/metal/additions_reducer

build/metal/flip_graph: $(METAL_SOURCES)
	mkdir -p build/metal
	xcrun clang++ $(METAL_FLAGS) -DMETAL_PROGRAM=1 src/metal/main.cpp src/metal/runtime.mm -o $@

build/metal/complexity_minimizer: $(METAL_SOURCES)
	mkdir -p build/metal
	xcrun clang++ $(METAL_FLAGS) -DMETAL_PROGRAM=2 src/metal/main.cpp src/metal/runtime.mm -o $@

build/metal/additions_reducer: $(METAL_SOURCES)
	mkdir -p build/metal
	xcrun clang++ $(METAL_FLAGS) -DMETAL_PROGRAM=3 src/metal/main.cpp src/metal/runtime.mm -o $@

build/metal/correctness: $(METAL_SOURCES) tests/metal/correctness.cpp
	mkdir -p build/metal
	xcrun clang++ $(METAL_FLAGS) -DMETAL_TESTING tests/metal/correctness.cpp src/metal/runtime.mm -o $@


metal: build/metal/flip_graph_f2 build/metal/complexity_minimizer_f2

build/metal/flip_graph_f2: $(METAL_SOURCES)
	mkdir -p build/metal
	xcrun clang++ $(METAL_FLAGS) -DMETAL_PROGRAM=1 -DMETAL_F2 src/metal/main.cpp src/metal/runtime.mm -o $@

build/metal/complexity_minimizer_f2: $(METAL_SOURCES)
	mkdir -p build/metal
	xcrun clang++ $(METAL_FLAGS) -DMETAL_PROGRAM=2 -DMETAL_F2 src/metal/main.cpp src/metal/runtime.mm -o $@

build/metal/f2_correctness: $(METAL_SOURCES) tests/metal/f2_correctness.cpp
	mkdir -p build/metal
	xcrun clang++ $(METAL_FLAGS) -DMETAL_TESTING -DMETAL_F2 tests/metal/f2_correctness.cpp src/metal/runtime.mm -o $@

.PHONY: test-metal smoke-metal metal-probe clean

build/metal/probe: src/metal/probe.mm
	mkdir -p build/metal
	xcrun clang++ -std=c++17 -O2 -fobjc-arc -framework Foundation -framework Metal $< -o $@

test-metal: metal build/metal/probe build/metal/correctness build/metal/f2_correctness
	python3 tests/metal/run.py

smoke-metal: metal build/metal/probe build/metal/correctness build/metal/f2_correctness
	python3 tests/metal/run.py --smoke

metal-probe: build/metal/probe
	python3 tests/metal/run.py --probe

clean:
	rm -f build/metal/flip_graph build/metal/flip_graph_f2 build/metal/complexity_minimizer build/metal/complexity_minimizer_f2 build/metal/additions_reducer build/metal/probe build/metal/correctness build/metal/f2_correctness
