CXX = nvcc
FLAGS = -O3 -Xptxas -O3
OBJECTS = src/entities/arg_parser.o src/utils/utils.o src/entities/addition.o src/entities/flip_set.o src/schemes/scheme_integer.o src/schemes/scheme_z2.o
FLIP_GRAPH_OBJECTS = src/entities/flip_graph.o src/main_flip_graph.o
COMPLEXITY_MINIMIZER_OBJECTS = src/entities/complexity_minimizer.o src/main_complexity_minimizer.o
ADDITIONS_REDUCER_OBJECTS = src/entities/scheme_additions_reducer.o src/main_additions_reducer.o

all: flip-graph complexity-minimizer

flip-graph: $(OBJECTS) $(FLIP_GRAPH_OBJECTS)
	$(CXX) $(FLAGS) $(OBJECTS) $(FLIP_GRAPH_OBJECTS) -o flip_graph

complexity-minimizer: $(OBJECTS) $(COMPLEXITY_MINIMIZER_OBJECTS)
	$(CXX) $(FLAGS) $(OBJECTS) $(COMPLEXITY_MINIMIZER_OBJECTS) -o complexity_minimizer

additions-reducer: $(OBJECTS) $(ADDITIONS_REDUCER_OBJECTS)
	$(CXX) $(FLAGS) $(OBJECTS) $(ADDITIONS_REDUCER_OBJECTS) -o additions_reducer

%.o: %.cu
	$(CXX) $(FLAGS) -I. -dc $< -o $@

clean:
	rm -f flip_graph complexity_minimizer additions_reducer
	rm -f $(OBJECTS) $(FLIP_GRAPH_OBJECTS) $(COMPLEXITY_MINIMIZER_OBJECTS) $(ADDITIONS_REDUCER_OBJECTS)
	rm -f *.nsys-rep *.sqlite

.PHONY: metal-probe
metal-probe:
	mkdir -p build/metal
	xcrun clang++ -std=c++17 -O2 -fobjc-arc -framework Foundation -framework Metal src/metal/probe.mm -o build/metal/probe
	./build/metal/probe

METAL_FLAGS = -mmacosx-version-min=15.0 -std=c++17 -O2 -fobjc-arc -ffp-contract=off -framework Foundation -framework Metal -DMETAL_SOURCE_DIR='"$(CURDIR)/src/metal"'
METAL_SOURCES = $(wildcard src/metal/*.h src/metal/*.cpp src/metal/*.metal) src/metal/runtime.mm
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

.PHONY: test-metal
test-metal: metal build/metal/correctness build/metal/f2_correctness
	python3 -m unittest discover -s tests/metal -p 'test_*.py'
	./build/metal/correctness
	python3 tests/metal/verify.py build/metal/test-output/*.json
	./build/metal/f2_correctness
	python3 tests/metal/verify.py build/metal/f2-test-output/*.json

.PHONY: smoke-metal
smoke-metal: test-metal
	python3 tests/metal/smoke.py
