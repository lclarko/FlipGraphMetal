#pragma once

#include "run_config.h"
#include "scheme_io.h"
#include "pool.h"
#include <filesystem>
#include <optional>

namespace fgm {

struct InputFile {
    std::filesystem::path path;
    std::string format, domain;
};

struct RunInput {
    enum class Kind { Files, Selection, Resume };
    Kind kind;
    std::vector<InputFile> files;
    std::map<std::string, std::string> selection;
    std::filesystem::path journal;
};

struct ExecutionSettings {
    uint64_t workers, batchSteps, blockSize, memoryBytes;
    std::string backend;
    uint64_t maxBatches=0;
};

struct ReductionSettings {
    uint32_t seed;
    uint64_t rounds, reducers, schemes, maxFlips, noImprovements, targetAdditions;
    std::string strategy="baseline";
};

struct HistorySettings {
    std::filesystem::path path;
    uint64_t storageBytes=512*1024*1024, transactionBytes=1024*1024, indexMemoryBytes=1024*1024;
};

struct RunConfig {
    std::string operation;
    bool additive=false;
    std::optional<ControlledConfig> policy;
    std::optional<ReductionSettings> reduction;
    std::optional<ReductionSettings> evaluation;
    RunInput input;
    ExecutionSettings execution;
    AdmissionLimits limits;
    PoolSettings pool;
    HistorySettings history;
    std::optional<uint64_t> discoveryTarget;
    std::optional<uint64_t> circuitTarget;
    std::filesystem::path output;
    Json resolved;
};

struct PreparedRun {
    RunConfig config;
    std::vector<AdmittedScheme> inputs;
    Json receipt;
    Json recoveredPools;
    Json recoveredAdditive;
    std::string recoveredHead;
    uint64_t historicalDiscoveries=0;
};

ReductionSettings parseReductionSettings(const Json &);
Json additiveContract(const PreparedRun &);

struct ExecutionLayout {
    uint64_t signedSchemeBytes, f2SchemeBytes, reducerLaneBytes, rngBytes, controlBytes;
};

RunConfig parseRunConfig(const Json &, const std::filesystem::path &base,
                         const std::string &expectedOperation = "",
                         const std::string &expectedDomain = "");
PreparedRun prepareRun(const std::filesystem::path &configuration, const ExecutionLayout &layout,
                       const std::string &expectedOperation = "",
                       const std::string &expectedDomain = "");
int runConfigured(int argc, char **argv, const std::string &operation,
                  const std::string &domain, const ExecutionLayout &layout,
                  void (*execute)(PreparedRun &) = nullptr);

} // namespace fgm
