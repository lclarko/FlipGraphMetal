.DEFAULT_GOAL := metal

METAL_CXX = xcrun clang++
METAL_FLAGS = -mmacosx-version-min=15.0 -std=c++17 -O2 -fobjc-arc -ffp-contract=off -framework Foundation -framework Metal
PROBE_FLAGS = -std=c++17 -O2 -fobjc-arc -framework Foundation -framework Metal
# Quote one shell argument, including checkout paths containing spaces or apostrophes.
shell_quote = '$(subst ','"'"',$(1))'
METAL_SOURCE_FLAG = -DMETAL_SOURCE_DIR=$(call shell_quote,"$(CURDIR)/src/metal")
METAL_CONFIG = build/metal/config.json
PROBE_CONFIG = build/metal/probe-config.json

.PHONY: FORCE
FORCE:

$(METAL_CONFIG): FORCE scripts/build_config.py
	@python3 scripts/build_config.py stamp --output=$@ --compiler=$(call shell_quote,$(METAL_CXX)) --flags=$(call shell_quote,$(METAL_FLAGS) $(METAL_SOURCE_FLAG)) --source-dir=$(call shell_quote,$(CURDIR)/src/metal)

$(PROBE_CONFIG): FORCE scripts/build_config.py
	@python3 scripts/build_config.py stamp --output=$@ --compiler=$(call shell_quote,$(METAL_CXX)) --flags=$(call shell_quote,$(PROBE_FLAGS)) --source-dir=$(call shell_quote,$(CURDIR)/src/metal)

# Check content receipts even when make only compares whole-second timestamps.
run_build = python3 scripts/build_config.py build --output=$@ --config=$(1) $(foreach dependency,$(filter-out FORCE $(1),$^),--dependency=$(call shell_quote,$(dependency))) --

METAL_SOURCES = $(wildcard src/metal/*.h src/metal/*.cpp src/metal/*.metal) src/metal/runtime.mm src/common/arg_parser.cpp src/common/arg_parser.h
.PHONY: metal
metal: build/metal/flip_graph build/metal/complexity_minimizer build/metal/additions_reducer

build/metal/flip_graph: $(METAL_SOURCES) makefile scripts/build_config.py $(METAL_CONFIG) FORCE
	@mkdir -p build/metal
	@$(call run_build,$(METAL_CONFIG)) $(METAL_CXX) $(METAL_FLAGS) $(METAL_SOURCE_FLAG) -DMETAL_PROGRAM=1 src/metal/main.cpp src/metal/runtime.mm -o $@

build/metal/complexity_minimizer: $(METAL_SOURCES) makefile scripts/build_config.py $(METAL_CONFIG) FORCE
	@mkdir -p build/metal
	@$(call run_build,$(METAL_CONFIG)) $(METAL_CXX) $(METAL_FLAGS) $(METAL_SOURCE_FLAG) -DMETAL_PROGRAM=2 src/metal/main.cpp src/metal/runtime.mm -o $@

build/metal/additions_reducer: $(METAL_SOURCES) makefile scripts/build_config.py $(METAL_CONFIG) FORCE
	@mkdir -p build/metal
	@$(call run_build,$(METAL_CONFIG)) $(METAL_CXX) $(METAL_FLAGS) $(METAL_SOURCE_FLAG) -DMETAL_PROGRAM=3 src/metal/main.cpp src/metal/runtime.mm -o $@

build/metal/correctness: $(METAL_SOURCES) makefile scripts/build_config.py $(METAL_CONFIG) FORCE tests/metal/correctness.cpp tests/metal/candidate_capacity.h
	@mkdir -p build/metal
	@$(call run_build,$(METAL_CONFIG)) $(METAL_CXX) $(METAL_FLAGS) $(METAL_SOURCE_FLAG) -DMETAL_TESTING tests/metal/correctness.cpp src/metal/runtime.mm -o $@


metal: build/metal/flip_graph_f2 build/metal/complexity_minimizer_f2

build/metal/flip_graph_f2: $(METAL_SOURCES) makefile scripts/build_config.py $(METAL_CONFIG) FORCE
	@mkdir -p build/metal
	@$(call run_build,$(METAL_CONFIG)) $(METAL_CXX) $(METAL_FLAGS) $(METAL_SOURCE_FLAG) -DMETAL_PROGRAM=1 -DMETAL_F2 src/metal/main.cpp src/metal/runtime.mm -o $@

build/metal/complexity_minimizer_f2: $(METAL_SOURCES) makefile scripts/build_config.py $(METAL_CONFIG) FORCE
	@mkdir -p build/metal
	@$(call run_build,$(METAL_CONFIG)) $(METAL_CXX) $(METAL_FLAGS) $(METAL_SOURCE_FLAG) -DMETAL_PROGRAM=2 -DMETAL_F2 src/metal/main.cpp src/metal/runtime.mm -o $@

build/metal/f2_correctness: $(METAL_SOURCES) makefile scripts/build_config.py $(METAL_CONFIG) FORCE tests/metal/f2_correctness.cpp tests/metal/candidate_capacity.h
	@mkdir -p build/metal
	@$(call run_build,$(METAL_CONFIG)) $(METAL_CXX) $(METAL_FLAGS) $(METAL_SOURCE_FLAG) -DMETAL_TESTING -DMETAL_F2 tests/metal/f2_correctness.cpp src/metal/runtime.mm -o $@

.PHONY: test-metal smoke-metal metal-probe clean

build/metal/probe: src/metal/probe.mm makefile scripts/build_config.py $(PROBE_CONFIG) FORCE
	@mkdir -p build/metal
	@$(call run_build,$(PROBE_CONFIG)) $(METAL_CXX) $(PROBE_FLAGS) $< -o $@

test-metal: metal build/metal/probe build/metal/correctness build/metal/f2_correctness
	python3 tests/metal/run.py

smoke-metal: metal build/metal/probe build/metal/correctness build/metal/f2_correctness
	python3 tests/metal/run.py --smoke

metal-probe: build/metal/probe
	python3 tests/metal/run.py --probe

clean:
	rm -f build/metal/flip_graph build/metal/flip_graph_f2 build/metal/complexity_minimizer build/metal/complexity_minimizer_f2 build/metal/additions_reducer build/metal/probe build/metal/correctness build/metal/f2_correctness
