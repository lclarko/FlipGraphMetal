.DEFAULT_GOAL := metal

METAL_CXX = xcrun clang++
METAL_FLAGS = -mmacosx-version-min=15.0 -std=c++17 -O2 -fobjc-arc -ffp-contract=off -framework Foundation -framework Metal
PROBE_FLAGS = -std=c++17 -O2 -fobjc-arc -framework Foundation -framework Metal
# Quote one shell argument, including checkout paths containing spaces or apostrophes.
shell_quote = '$(subst ','"'"',$(1))'
METAL_LIBRARY_MODE ?= metallib
METAL_COMPILER ?= xcrun -sdk macosx metal
METAL_SHADER_FLAGS ?= -std=metal3.2 -mmacosx-version-min=15.0 -fmetal-math-mode=safe -fmetal-math-fp32-functions=precise
PACKAGE_DIR ?= build/metal/package

ifeq ($(METAL_LIBRARY_MODE),source)
metal_runtime_flags = -DMETAL_SOURCE_DIR=$(call shell_quote,"$(CURDIR)/src/metal")
metal_library_dependency =
else ifeq ($(METAL_LIBRARY_MODE),metallib)
metal_runtime_flags = -include $(call shell_quote,build/metal/shaders/$(1).h)
metal_library_dependency = build/metal/shaders/$(1).h
else
$(error METAL_LIBRARY_MODE must be metallib or source)
endif
METAL_CONFIG = build/metal/config.json
PROBE_CONFIG = build/metal/probe-config.json

.PHONY: FORCE
FORCE:

$(METAL_CONFIG): FORCE scripts/build_config.py
	@python3 scripts/build_config.py stamp --output=$@ --compiler=$(call shell_quote,$(METAL_CXX)) --flags=$(call shell_quote,$(METAL_FLAGS) $(METAL_LIBRARY_MODE) $(if $(filter source,$(METAL_LIBRARY_MODE)),$(call metal_runtime_flags,signed))) --source-dir=$(call shell_quote,$(CURDIR)/src/metal)

$(PROBE_CONFIG): FORCE scripts/build_config.py
	@python3 scripts/build_config.py stamp --output=$@ --compiler=$(call shell_quote,$(METAL_CXX)) --flags=$(call shell_quote,$(PROBE_FLAGS)) --source-dir=$(call shell_quote,$(CURDIR)/src/metal)

SHADER_SOURCES = $(wildcard src/metal/*.h src/metal/*.metal)
build/metal/shaders/%.h: $(SHADER_SOURCES) scripts/metal_library.py scripts/build_config.py makefile FORCE
	@python3 scripts/metal_library.py build --source-dir=src/metal --output=build/metal/shaders/$*.metallib --header=$@ --variant=$* --compiler=$(call shell_quote,$(METAL_COMPILER)) --flags=$(call shell_quote,$(METAL_SHADER_FLAGS))

# Check content receipts even when make only compares whole-second timestamps.
run_build = python3 scripts/build_config.py build --output=$@ --config=$(1) $(foreach dependency,$(filter-out FORCE $(1),$^),--dependency=$(call shell_quote,$(dependency))) --

METAL_SOURCES = $(wildcard src/metal/*.h src/metal/*.cpp src/metal/*.metal) src/metal/runtime.mm src/common/arg_parser.cpp src/common/arg_parser.h
NATIVE_WORKFLOW = src/workflow/scheme_io.cpp src/workflow/execution.cpp src/workflow/journal.cpp
NATIVE_WORKFLOW_HEADERS = $(wildcard src/workflow/*.h)
.PHONY: metal
metal: build/metal/flip_graph build/metal/complexity_minimizer build/metal/additions_reducer

build/metal/flip_graph: $(METAL_SOURCES) $(NATIVE_WORKFLOW) $(NATIVE_WORKFLOW_HEADERS) makefile scripts/build_config.py $(METAL_CONFIG) FORCE $(call metal_library_dependency,signed)
	@mkdir -p build/metal
	@$(call run_build,$(METAL_CONFIG)) $(METAL_CXX) $(METAL_FLAGS) $(call metal_runtime_flags,signed) -DMETAL_PROGRAM=1 src/metal/main.cpp src/metal/runtime.mm $(NATIVE_WORKFLOW) -o $@

build/metal/complexity_minimizer: $(METAL_SOURCES) makefile scripts/build_config.py $(METAL_CONFIG) FORCE $(call metal_library_dependency,signed)
	@mkdir -p build/metal
	@$(call run_build,$(METAL_CONFIG)) $(METAL_CXX) $(METAL_FLAGS) $(call metal_runtime_flags,signed) -DMETAL_PROGRAM=2 src/metal/main.cpp src/metal/runtime.mm -o $@

build/metal/additions_reducer: $(METAL_SOURCES) $(NATIVE_WORKFLOW) $(NATIVE_WORKFLOW_HEADERS) makefile scripts/build_config.py $(METAL_CONFIG) FORCE $(call metal_library_dependency,signed)
	@mkdir -p build/metal
	@$(call run_build,$(METAL_CONFIG)) $(METAL_CXX) $(METAL_FLAGS) $(call metal_runtime_flags,signed) -DMETAL_PROGRAM=3 src/metal/main.cpp src/metal/runtime.mm $(NATIVE_WORKFLOW) -o $@

build/metal/correctness: $(METAL_SOURCES) makefile scripts/build_config.py $(METAL_CONFIG) FORCE tests/metal/correctness.cpp tests/metal/candidate_capacity.h $(call metal_library_dependency,signed-testing)
	@mkdir -p build/metal
	@$(call run_build,$(METAL_CONFIG)) $(METAL_CXX) $(METAL_FLAGS) $(call metal_runtime_flags,signed-testing) -DMETAL_TESTING tests/metal/correctness.cpp src/metal/runtime.mm -o $@


metal: build/metal/flip_graph_f2 build/metal/complexity_minimizer_f2

build/metal/flip_graph_f2: $(METAL_SOURCES) $(NATIVE_WORKFLOW) $(NATIVE_WORKFLOW_HEADERS) makefile scripts/build_config.py $(METAL_CONFIG) FORCE $(call metal_library_dependency,f2)
	@mkdir -p build/metal
	@$(call run_build,$(METAL_CONFIG)) $(METAL_CXX) $(METAL_FLAGS) $(call metal_runtime_flags,f2) -DMETAL_PROGRAM=1 -DMETAL_F2 src/metal/main.cpp src/metal/runtime.mm $(NATIVE_WORKFLOW) -o $@

build/metal/complexity_minimizer_f2: $(METAL_SOURCES) makefile scripts/build_config.py $(METAL_CONFIG) FORCE $(call metal_library_dependency,f2)
	@mkdir -p build/metal
	@$(call run_build,$(METAL_CONFIG)) $(METAL_CXX) $(METAL_FLAGS) $(call metal_runtime_flags,f2) -DMETAL_PROGRAM=2 -DMETAL_F2 src/metal/main.cpp src/metal/runtime.mm -o $@

build/metal/f2_correctness: $(METAL_SOURCES) makefile scripts/build_config.py $(METAL_CONFIG) FORCE tests/metal/f2_correctness.cpp tests/metal/candidate_capacity.h $(call metal_library_dependency,f2-testing)
	@mkdir -p build/metal
	@$(call run_build,$(METAL_CONFIG)) $(METAL_CXX) $(METAL_FLAGS) $(call metal_runtime_flags,f2-testing) -DMETAL_TESTING -DMETAL_F2 tests/metal/f2_correctness.cpp src/metal/runtime.mm -o $@

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
	rm -f build/metal/flip_graph build/metal/flip_graph_f2 build/metal/complexity_minimizer build/metal/complexity_minimizer_f2 build/metal/additions_reducer build/metal/probe build/metal/correctness build/metal/f2_correctness build/metal/scheme_tool

# Produce a new, relocatable directory without copying build evidence or sources.
.PHONY: package-metal
package-metal: metal scheme-tool
	python3 scripts/metal_library.py package --binary-dir=build/metal --output=$(call shell_quote,$(PACKAGE_DIR))

WORKFLOW_CXX ?= xcrun clang++
WORKFLOW_FLAGS ?= -mmacosx-version-min=15.0 -std=c++17 -O2
WORKFLOW_CONFIG = build/workflow/config.json
WORKFLOW_SOURCES = $(wildcard src/workflow/*.h src/workflow/*.cpp)

$(WORKFLOW_CONFIG): FORCE scripts/build_config.py
	@python3 scripts/build_config.py stamp --output=$@ --compiler=$(call shell_quote,$(WORKFLOW_CXX)) --flags=$(call shell_quote,$(WORKFLOW_FLAGS)) --source-dir=$(call shell_quote,$(CURDIR)/src/workflow)

.PHONY: scheme-tool test-workflow qualify-workflow
scheme-tool: build/metal/scheme_tool

build/metal/scheme_tool: $(WORKFLOW_SOURCES) makefile scripts/build_config.py $(WORKFLOW_CONFIG) FORCE
	@mkdir -p build/metal
	@$(call run_build,$(WORKFLOW_CONFIG)) $(WORKFLOW_CXX) $(WORKFLOW_FLAGS) src/workflow/scheme_tool.cpp $(NATIVE_WORKFLOW) -o $@

WORKFLOW_TEST_DRIVERS = build/workflow/test_run_config build/workflow/test_host_rng build/workflow/test_execution build/workflow/test_journal build/workflow/test_pool build/workflow/test_reduction_result

build/workflow/test_reduction_result: tests/workflow/reduction_result.cpp $(WORKFLOW_SOURCES) $(SHADER_SOURCES) makefile scripts/build_config.py $(WORKFLOW_CONFIG) FORCE
	@$(call run_build,$(WORKFLOW_CONFIG)) $(WORKFLOW_CXX) $(WORKFLOW_FLAGS) -Isrc/workflow -Isrc/metal tests/workflow/reduction_result.cpp src/workflow/scheme_io.cpp src/workflow/journal.cpp -o $@

build/workflow/test_controlled: tests/workflow/controlled.cpp src/workflow/run_config.h src/workflow/json.h $(SHADER_SOURCES) makefile scripts/build_config.py $(WORKFLOW_CONFIG) FORCE
	@$(call run_build,$(WORKFLOW_CONFIG)) $(WORKFLOW_CXX) $(WORKFLOW_FLAGS) -Isrc/workflow -Isrc/metal tests/workflow/controlled.cpp -o $@

build/metal/controlled_signed: tests/workflow/controlled.cpp $(METAL_SOURCES) src/workflow/run_config.h src/workflow/json.h makefile scripts/build_config.py $(METAL_CONFIG) FORCE $(call metal_library_dependency,signed-testing)
	@$(call run_build,$(METAL_CONFIG)) $(METAL_CXX) $(METAL_FLAGS) $(call metal_runtime_flags,signed-testing) -DMETAL_TESTING -DFGM_CONTROLLED_GPU -Isrc/workflow -Isrc/metal tests/workflow/controlled.cpp src/metal/runtime.mm -o $@

build/metal/controlled_f2: tests/workflow/controlled.cpp $(METAL_SOURCES) src/workflow/run_config.h src/workflow/json.h makefile scripts/build_config.py $(METAL_CONFIG) FORCE $(call metal_library_dependency,f2-testing)
	@$(call run_build,$(METAL_CONFIG)) $(METAL_CXX) $(METAL_FLAGS) $(call metal_runtime_flags,f2-testing) -DMETAL_TESTING -DMETAL_F2 -DFGM_CONTROLLED_GPU -Isrc/workflow -Isrc/metal tests/workflow/controlled.cpp src/metal/runtime.mm -o $@

build/workflow/test_run_config: tests/workflow/run_config.cpp src/workflow/run_config.h src/workflow/json.h makefile scripts/build_config.py $(WORKFLOW_CONFIG) FORCE
	@$(call run_build,$(WORKFLOW_CONFIG)) $(WORKFLOW_CXX) $(WORKFLOW_FLAGS) -Isrc/workflow tests/workflow/run_config.cpp -o $@

build/workflow/test_host_rng: tests/workflow/host_rng.cpp makefile scripts/build_config.py $(WORKFLOW_CONFIG) FORCE
	@$(call run_build,$(WORKFLOW_CONFIG)) $(WORKFLOW_CXX) $(WORKFLOW_FLAGS) tests/workflow/host_rng.cpp -o $@

build/workflow/test_execution: tests/workflow/execution.cpp $(WORKFLOW_SOURCES) $(SHADER_SOURCES) src/metal/host.h makefile scripts/build_config.py $(WORKFLOW_CONFIG) FORCE
	@$(call run_build,$(WORKFLOW_CONFIG)) $(WORKFLOW_CXX) $(WORKFLOW_FLAGS) -DFGM_JOURNAL_TESTING -DFGM_SEARCH_TESTING -Isrc/workflow tests/workflow/execution.cpp $(NATIVE_WORKFLOW) -o $@

build/workflow/test_pool: src/workflow/journal.cpp src/workflow/journal.h tests/workflow/pool.cpp src/workflow/pool.h src/workflow/run_config.h src/workflow/scheme_io.cpp src/workflow/scheme_io.h src/workflow/json.h makefile scripts/build_config.py $(WORKFLOW_CONFIG) FORCE
	@$(call run_build,$(WORKFLOW_CONFIG)) $(WORKFLOW_CXX) $(WORKFLOW_FLAGS) -Isrc/workflow tests/workflow/pool.cpp src/workflow/scheme_io.cpp src/workflow/journal.cpp -o $@

build/workflow/test_journal: tests/workflow/journal.cpp src/workflow/journal.cpp src/workflow/journal.h src/workflow/scheme_io.cpp src/workflow/scheme_io.h src/workflow/json.h makefile scripts/build_config.py $(WORKFLOW_CONFIG) FORCE
	@$(call run_build,$(WORKFLOW_CONFIG)) $(WORKFLOW_CXX) $(WORKFLOW_FLAGS) -DFGM_JOURNAL_TESTING -Isrc/workflow tests/workflow/journal.cpp src/workflow/journal.cpp src/workflow/scheme_io.cpp -o $@

test-workflow: scheme-tool $(WORKFLOW_TEST_DRIVERS)
	python3 tests/workflow/run.py

qualify-workflow: scheme-tool $(WORKFLOW_TEST_DRIVERS) build/workflow/test_controlled
	python3 tests/workflow/run.py --qualification
