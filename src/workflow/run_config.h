#pragma once

#include "json.h"
#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <set>

namespace fgm {

inline uint64_t configCheckedAdd(uint64_t left, uint64_t right) {
    if (right > std::numeric_limits<uint64_t>::max() - left) {
        throw std::runtime_error("configuration addition overflow");
    }
    return left + right;
}

inline uint64_t configCheckedMultiply(uint64_t left, uint64_t right) {
    if (right && left > std::numeric_limits<uint64_t>::max() / right) {
        throw std::runtime_error("configuration multiplication overflow");
    }
    return left * right;
}

// Policy settings only. This type neither loads parents nor dispatches kernels.
// Configuration or execution errors do not establish tensor invalidity.
// JSON integers are limited to INT64_MAX; arithmetic uses checked uint64_t.
struct ControlledConfig {
    enum class Mode { RankReduction, Alternatives };
    enum class Domain { Signed, F2 };
    Mode mode;
    Domain domain;
    uint32_t seed;
    std::array<uint64_t, 3> dimensions;
    uint64_t anchor;
    uint64_t excursion;
    uint64_t intervalMin;
    uint64_t intervalMax;
    uint32_t reductionQ;
    uint64_t stagnationLimit;
    uint64_t flipBudget;
    uint64_t controlBudget;
    uint64_t optionalQuota;
    uint64_t proposalLimit;
    std::optional<uint64_t> targetRank;
    static constexpr uint64_t rankCapacity = 350;

    uint64_t intervalSpan() const {
        if (intervalMin == 0 || intervalMax < intervalMin) {
            throw std::runtime_error("invalid expansion interval");
        }
        return configCheckedAdd(intervalMax - intervalMin, 1);
    }

    uint64_t ceiling() const {
        const auto excursionCeiling = configCheckedAdd(anchor, excursion);
        const auto naive = configCheckedMultiply(
            configCheckedMultiply(dimensions[0], dimensions[1]), dimensions[2]);
        return std::min(std::min(excursionCeiling, naive), rankCapacity);
    }

    bool expansionAllowed(uint64_t currentRank) const {
        return configCheckedAdd(currentRank, 1) <= ceiling();
    }

    // Execution eligibility is not a tensor-validity verdict. Candidate capacity
    // needs actual factors and is assessed separately by scheme admission.
    std::vector<std::string> representationIneligibility(uint64_t currentRank) const {
        std::vector<std::string> reasons;
        if (currentRank == 0 || currentRank > rankCapacity) {
            reasons.push_back("rank exceeds execution representation");
        }
        for (size_t i = 0; i < dimensions.size(); ++i) {
            if (dimensions[i] == 0 || dimensions[i] > 16) {
                reasons.push_back("dimension exceeds execution representation");
            }
            if (configCheckedMultiply(dimensions[i], dimensions[(i + 1) % 3]) > 64) {
                reasons.push_back("factor exceeds execution representation");
            }
        }
        return reasons;
    }

    Json resolved() const {
        Json result = Json::dict();
        result.object["schema"] = Json("fgm-controlled-config-v1");
        result.object["policy"] = Json("controlled-v1");
        result.object["mode"] = Json(mode == Mode::RankReduction ? "rank-reduction" : "alternatives");
        result.object["domain"] = Json(domain == Domain::Signed ? "ZT" : "F2");
        auto put = [&](const char *name, uint64_t value) {
            if (value > uint64_t(std::numeric_limits<int64_t>::max())) {
                throw std::runtime_error("configuration integer exceeds JSON representation");
            }
            result.object[name] = Json(int64_t(value));
        };
        put("seed", seed);
        put(mode == Mode::RankReduction ? "stage_rank" : "collection_rank", anchor);
        put("excursion", excursion);
        put("interval_min", intervalMin);
        put("interval_max", intervalMax);
        put("reduction_q", reductionQ);
        put("stagnation_limit", stagnationLimit);
        put("flip_budget", flipBudget);
        put("control_budget", controlBudget);
        put("optional_quota", optionalQuota);
        put("proposal_limit", proposalLimit);
        if (targetRank) {
            put("target_rank", *targetRank);
        } else {
            result.object["target_rank"] = Json();
        }
        Json dims = Json::list();
        for (auto dimension : dimensions) {
            if (dimension > uint64_t(std::numeric_limits<int64_t>::max())) {
                throw std::runtime_error("dimension exceeds JSON representation");
            }
            dims.array.emplace_back(int64_t(dimension));
        }
        result.object["dimensions"] = dims;
        return result;
    }
};

inline ControlledConfig parseControlledConfig(const Json &value, const std::string &expectedDomain = "") {
    if (value.kind != Json::Object) {
        throw std::runtime_error("configuration must be an object");
    }
    const std::set<std::string> fields = {
        "schema", "policy", "mode", "domain", "seed", "dimensions",
        "stage_rank", "collection_rank", "excursion", "interval_min", "interval_max",
        "reduction_q", "stagnation_limit", "flip_budget", "control_budget",
        "optional_quota", "proposal_limit", "target_rank"
    };
    for (const auto &entry : value.object) {
        if (!fields.count(entry.first)) {
            throw std::runtime_error("unknown configuration field: " + entry.first);
        }
    }
    if (value.at("schema").str() != "fgm-controlled-config-v1" ||
        value.at("policy").str() != "controlled-v1") {
        throw std::runtime_error("unsupported configuration schema or policy");
    }
    ControlledConfig result{};
    const auto mode = value.at("mode").str();
    if (mode != "rank-reduction" && mode != "alternatives") {
        throw std::runtime_error("unsupported search mode");
    }
    result.mode = mode == "rank-reduction" ? ControlledConfig::Mode::RankReduction : ControlledConfig::Mode::Alternatives;
    const auto domain = value.at("domain").str();
    if ((domain != "ZT" && domain != "F2") || (!expectedDomain.empty() && domain != expectedDomain)) {
        throw std::runtime_error("unsupported or conflicting coefficient domain");
    }
    result.domain = domain == "ZT" ? ControlledConfig::Domain::Signed : ControlledConfig::Domain::F2;
    auto number = [](const Json &item, bool positive = false) {
        const auto parsed = item.num();
        if (parsed < 0 || (positive && parsed == 0)) {
            throw std::runtime_error("invalid nonnegative configuration integer");
        }
        return uint64_t(parsed);
    };
    auto integer = [&](const char *name, bool positive = false) {
        return number(value.at(name), positive);
    };
    const auto seed = integer("seed"), q = integer("reduction_q");
    if (seed > UINT32_MAX || q > UINT32_MAX) {
        throw std::runtime_error("seed and reduction threshold must fit uint32");
    }
    result.seed = uint32_t(seed);
    result.reductionQ = uint32_t(q);
    const auto &dims = value.at("dimensions");
    if (dims.kind != Json::Array || dims.array.size() != 3) {
        throw std::runtime_error("three dimensions required");
    }
    for (size_t i = 0; i < 3; ++i) {
        result.dimensions[i] = number(dims.array[i], true);
    }
    const bool reduction = result.mode == ControlledConfig::Mode::RankReduction;
    if (value.has(reduction ? "collection_rank" : "stage_rank")) {
        throw std::runtime_error("anchor conflicts with search mode");
    }
    result.anchor = integer(reduction ? "stage_rank" : "collection_rank", true);
    result.excursion = integer("excursion");
    result.intervalMin = integer("interval_min", true);
    result.intervalMax = integer("interval_max", true);
    result.stagnationLimit = integer("stagnation_limit", true);
    result.flipBudget = integer("flip_budget");
    result.controlBudget = integer("control_budget");
    result.optionalQuota = integer("optional_quota");
    result.proposalLimit = value.has("proposal_limit") ? integer("proposal_limit", true) : 64;
    if (result.proposalLimit > UINT32_MAX) {
        throw std::runtime_error("proposal limit must fit uint32");
    }
    const auto &target = value.at("target_rank");
    if (target.kind != Json::Null) {
        result.targetRank = number(target);
    }
    result.intervalSpan();
    result.ceiling();
    return result;
}

} // namespace fgm
