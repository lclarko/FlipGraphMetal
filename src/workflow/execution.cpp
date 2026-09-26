#include "execution.h"
#include "journal.h"
#include <fstream>
#include <set>
#include <sstream>
#include <iostream>
#include <chrono>
#include <mach-o/dyld.h>

namespace fgm {
namespace {
void fields(const Json &value, std::initializer_list<const char *> allowed) {
    if (value.kind != Json::Object) throw std::runtime_error("expected configuration object");
    std::set<std::string> names;
    for (auto name : allowed) names.insert(name);
    for (const auto &item : value.object)
        if (!names.count(item.first)) throw std::runtime_error("unknown configuration field: " + item.first);
}
uint64_t natural(const Json &value, bool positive = true) {
    const auto number = value.num();
    if (number < 0 || (positive && number == 0)) throw std::runtime_error("invalid configuration integer");
    return uint64_t(number);
}
Json integer(uint64_t value) {
    if (value > INT64_MAX) throw std::runtime_error("record integer overflow");
    return Json(int64_t(value));
}
std::filesystem::path pathAt(const Json &value, const std::filesystem::path &base) {
    const auto name = value.str();
    if (name.empty() || name.find('\0') != std::string::npos) throw std::runtime_error("invalid configuration path");
    return std::filesystem::absolute(base / name).lexically_normal();
}
std::string readBounded(const std::filesystem::path &path, uint64_t limit) {
    if (!std::filesystem::is_regular_file(path)) throw std::runtime_error("configuration must be a regular file");
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read configuration");
    std::string result;
    char buffer[4096];
    while (input) {
        input.read(buffer, sizeof buffer);
        const auto count = uint64_t(input.gcount());
        if (count > limit - result.size()) throw Resource("configuration byte limit exceeded");
        result.append(buffer, size_t(count));
    }
    if (!input.eof()) throw std::runtime_error("configuration read failed");
    return result;
}
std::string domainOf(const RunConfig &config) {
    return config.policy && config.policy->domain == ControlledConfig::Domain::F2 ? "F2" : "ZT";
}
}

RunConfig parseRunConfig(const Json &value, const std::filesystem::path &base,
                         const std::string &expectedOperation, const std::string &expectedDomain) {
    fields(value, {"schema", "operation", "policy", "reduction", "input", "execution", "limits", "output", "pool", "history", "discovery_target"});
    if (value.at("schema").str() != "fgm-run-v1") throw std::runtime_error("unsupported run schema");
    RunConfig result{};
    result.resolved = value;
    result.operation = value.at("operation").str();
    if (result.operation != "search" && result.operation != "reduce") throw std::runtime_error("unsupported run operation");
    if (!expectedOperation.empty() && result.operation != expectedOperation) throw std::runtime_error("operation conflicts with executable");
    if (result.operation == "search") {
        if (value.has("reduction")) throw std::runtime_error("reduction settings conflict with search");
        result.policy = parseControlledConfig(value.at("policy"), expectedDomain);
        result.resolved.object["policy"] = result.policy->resolved();
    } else {
        if (value.has("policy") || (!expectedDomain.empty() && expectedDomain != "ZT"))
            throw std::runtime_error("signed reduction requires ZT and reduction settings");
        const auto &settings = value.at("reduction");
        fields(settings, {"domain", "seed", "rounds", "reducers", "schemes", "max_flips", "no_improvements", "target_additions"});
        if (settings.at("domain").str() != "ZT") throw std::runtime_error("only signed addition reduction is supported");
        auto seed = natural(settings.at("seed"), false);
        if (seed > UINT32_MAX) throw std::runtime_error("seed exceeds uint32");
        result.reduction = ReductionSettings{uint32_t(seed), natural(settings.at("rounds")),
            natural(settings.at("reducers")), natural(settings.at("schemes")),
            natural(settings.at("max_flips"), false), natural(settings.at("no_improvements")),
            natural(settings.at("target_additions"), false)};
        const auto &r = *result.reduction;
        if (r.schemes > r.reducers || r.reducers > 1048576 || r.maxFlips >= INT32_MAX ||
            r.rounds > INT32_MAX || r.noImprovements > INT32_MAX || r.targetAdditions > INT32_MAX)
            throw std::runtime_error("reduction setting exceeds execution capacity");
        if(configCheckedMultiply(configCheckedMultiply(r.rounds,r.maxFlips),r.schemes-1)>INT64_MAX)
            throw std::runtime_error("reducer applied-work counters exceed record capacity");
    }
    const auto &execution = value.at("execution");
    fields(execution, {"workers", "batch_steps", "block_size", "backend", "memory_bytes"});
    result.execution = {natural(execution.at("workers")), natural(execution.at("batch_steps")),
        natural(execution.at("block_size")), natural(execution.at("memory_bytes")), execution.at("backend").str()};
    const auto &e = result.execution;
    if (e.workers > 1048576 || e.blockSize > 1024 || e.batchSteps > UINT32_MAX)
        throw std::runtime_error("execution setting exceeds capacity");
    if (e.backend != "auto" && e.backend != "general" && e.backend != "packed") throw std::runtime_error("unsupported execution backend");
    if (result.reduction && (e.backend == "packed" || e.workers != result.reduction->schemes))
        throw std::runtime_error("reducer requires general execution and workers equal to schemes");
    if (value.has("limits")) {
        const auto &limits = value.at("limits");
        fields(limits, {"record_bytes", "verification_work", "selection_memory", "scan_bytes"});
        if (limits.has("record_bytes")) result.limits.record = natural(limits.at("record_bytes"));
        if (limits.has("verification_work")) result.limits.work = natural(limits.at("verification_work"));
        if (limits.has("selection_memory")) result.limits.selection = natural(limits.at("selection_memory"));
        if (limits.has("scan_bytes")) result.limits.scan = natural(limits.at("scan_bytes"));
    }
    Json resolvedLimits = Json::dict();
    resolvedLimits.object["record_bytes"] = integer(result.limits.record);
    resolvedLimits.object["verification_work"] = integer(result.limits.work);
    resolvedLimits.object["selection_memory"] = integer(result.limits.selection);
    resolvedLimits.object["scan_bytes"] = integer(result.limits.scan);
    result.resolved.object["limits"] = resolvedLimits;
    result.output = pathAt(value.at("output"), base);
    result.resolved.object["output"] = Json(result.output.string());
    if(value.has("pool")) {
        if(!result.policy)throw std::runtime_error("retained pool settings require search");
        const auto &p=value.at("pool");fields(p,{"capacity_per_rank","reserve_per_rank","memory_bytes","stage_threshold","selector"});
        result.pool={natural(p.at("capacity_per_rank")),natural(p.at("reserve_per_rank")),natural(p.at("memory_bytes")),natural(p.at("stage_threshold")),p.at("selector").str()};
    }
    RankPools checkPool(result.pool);
    if(value.has("discovery_target") && value.at("discovery_target").kind!=Json::Null) {
        if(!result.policy||result.policy->mode!=ControlledConfig::Mode::Alternatives)throw std::runtime_error("discovery target requires alternatives");
        result.discoveryTarget=natural(value.at("discovery_target"),false);
    }
    result.resolved.object["discovery_target"]=result.discoveryTarget?integer(*result.discoveryTarget):Json();
    Json p=Json::dict();p.object["capacity_per_rank"]=integer(result.pool.capacity);p.object["reserve_per_rank"]=integer(result.pool.reserve);
    p.object["memory_bytes"]=integer(result.pool.memoryBytes);p.object["stage_threshold"]=integer(result.pool.threshold);p.object["selector"]=Json(result.pool.selector);
    if(result.policy)result.resolved.object["pool"]=p;
    result.history.path=std::filesystem::path(result.output.string()+".journal");
    if(value.has("history")) {
        const auto &h=value.at("history");fields(h,{"path","storage_bytes","transaction_bytes","index_memory_bytes"});
        result.history.path=pathAt(h.at("path"),base);result.history.storageBytes=natural(h.at("storage_bytes"));
        result.history.transactionBytes=natural(h.at("transaction_bytes"));result.history.indexMemoryBytes=natural(h.at("index_memory_bytes"));
    }
    if(result.history.transactionBytes>result.history.storageBytes||result.history.indexMemoryBytes<4096)
        throw std::runtime_error("invalid history resource limits");
    const auto &input = value.at("input");
    const auto kind = input.at("kind").str();
    Json resolvedInput = input;
    if (kind == "files") {
        fields(input, {"kind", "files"});
        result.input.kind = RunInput::Kind::Files;
        const auto &files = input.at("files");
        if (files.kind != Json::Array || files.array.empty()) throw std::runtime_error("input files must be a nonempty array");
        for (size_t i = 0; i < files.array.size(); ++i) {
            const auto &item = files.array[i];
            fields(item, {"path", "format", "domain"});
            const auto domain = item.has("domain") ? item.at("domain").str() : "";
            if (!domain.empty() && domain != domainOf(result)) throw std::runtime_error("input domain conflicts with run");
            result.input.files.push_back({pathAt(item.at("path"), base), item.at("format").str(), domain});
            resolvedInput.object["files"].array[i].object["path"] = Json(result.input.files.back().path.string());
        }
    } else if (kind == "selection") {
        fields(input, {"kind", "manifest", "count", "seed", "ids", "filters"});
        result.input.kind = RunInput::Kind::Selection;
        const auto manifest = pathAt(input.at("manifest"), base);
        result.input.selection["--input"] = manifest.string();
        resolvedInput.object["manifest"] = Json(manifest.string());
        result.input.selection["--count"] = std::to_string(natural(input.at("count")));
        if (input.has("seed") == input.has("ids")) throw std::runtime_error("selection requires exactly one of seed or ids");
        if (input.has("seed")) result.input.selection["--seed"] = std::to_string(natural(input.at("seed"), false));
        else {
            const auto ids = pathAt(input.at("ids"), base);
            result.input.selection["--ids"] = ids.string();
            resolvedInput.object["ids"] = Json(ids.string());
        }
        if (input.has("filters")) {
            const auto &filters = input.at("filters");
            fields(filters, {"domain", "dimensions", "rank", "group"});
            for (const auto &entry : filters.object) {
                std::string text;
                if (entry.first == "rank") text = std::to_string(natural(entry.second));
                else if (entry.first == "dimensions") {
                    if (entry.second.kind != Json::Array || entry.second.array.size() != 3) throw std::runtime_error("three filter dimensions required");
                    for (const auto &dimension : entry.second.array) {
                        if (!text.empty()) text += ',';
                        text += std::to_string(natural(dimension));
                    }
                } else text = entry.second.str();
                result.input.selection["--filter-" + entry.first] = text;
            }
        }
    } else if (kind == "resume") {
        fields(input, {"kind", "journal"});
        result.input.kind = RunInput::Kind::Resume;
        result.input.journal = pathAt(input.at("journal"), base);
        if(value.has("history")&&result.history.path!=result.input.journal)throw std::runtime_error("resume and history paths differ");
        result.history.path=result.input.journal;
        resolvedInput.object["journal"] = Json(result.input.journal.string());
    } else throw std::runtime_error("unsupported run input route");
    result.resolved.object["input"] = resolvedInput;
    Json h=Json::dict();h.object["path"]=Json(result.history.path.string());h.object["storage_bytes"]=integer(result.history.storageBytes);
    h.object["transaction_bytes"]=integer(result.history.transactionBytes);h.object["index_memory_bytes"]=integer(result.history.indexMemoryBytes);result.resolved.object["history"]=h;
    return result;
}

PreparedRun prepareRun(const std::filesystem::path &configuration, const ExecutionLayout &layout,
                       const std::string &expectedOperation, const std::string &expectedDomain) {
    const auto configPath = std::filesystem::absolute(configuration);
    const auto bytes = readBounded(configPath, 1048576);
    PreparedRun run;
    run.config = parseRunConfig(Parser(bytes).parse(), configPath.parent_path(), expectedOperation, expectedDomain);
    const auto &config = run.config;
    if (std::filesystem::exists(config.output) || std::filesystem::is_symlink(config.output))
        throw std::runtime_error("run output already exists");
    if (!std::filesystem::is_directory(config.output.parent_path()))
        throw std::runtime_error("run output parent must exist");
    if(config.reduction&&(std::filesystem::exists(config.output.string()+".circuits.jsonl")||std::filesystem::is_symlink(config.output.string()+".circuits.jsonl")))
        throw std::runtime_error("circuit artifact output already exists");
    AdmissionContext context(config.limits);
    Json presentations = Json::list();
    std::set<std::string> seen;
    uint64_t admittedBytes = configCheckedMultiply(jsonMemoryBytes(config.resolved),2), duplicates = 0;
    uint64_t acknowledgedSequence=0,acknowledgedDiscoveries=0;
    std::string acknowledgedHead(64,'0');
    auto admit = [&](const AdmittedScheme &item) {
        const auto &source = item.source;
        if ((source.f2 ? "F2" : "ZT") != domainOf(config)) throw std::runtime_error("input domain conflicts with run");
        const auto &eligibility = item.report.at("eligibility");
        if (config.policy) {
            for (size_t i = 0; i < 3; ++i)
                if (source.n[i] != config.policy->dimensions[i]) throw std::runtime_error("input dimensions conflict with policy");
            if (!eligibility.at("search_eligible").boolean)
                throw std::runtime_error("mathematically valid input is not search-representation eligible");
        } else if (!eligibility.at("signed_reducer_eligible").boolean) {
            throw std::runtime_error("mathematically valid input is not signed-reducer eligible");
        }
        // Charge both retained report copies and both factor matrices before
        // retention. Allocator bookkeeping and process RSS remain separate.
        const auto charge = configCheckedAdd(admittedMemoryBytes(item), jsonMemoryBytes(item.report));
        admittedBytes = configCheckedAdd(admittedBytes, charge);
        if (admittedBytes > config.execution.memoryBytes) throw Resource("admission content exceeds run memory budget");
        presentations.array.push_back(item.report);
        const auto id = item.report.at("scheme_id").str();
        if (config.policy && !seen.insert(id).second) { ++duplicates; return; }
        run.inputs.push_back(item);
    };
    if (config.input.kind == RunInput::Kind::Files) {
        for (const auto &file : config.input.files)
            context.readRecords(file.path, file.format, file.domain, admit);
    } else if(config.input.kind==RunInput::Kind::Selection) context.readSelection(config.input.selection, admit);
    else {
        if(!config.policy)throw std::runtime_error("journal resume requires search");
        uint64_t stage=config.policy->anchor;
        auto recovered=Journal::replayReadOnly(config.input.journal,
            {config.history.storageBytes,config.history.transactionBytes,config.history.indexMemoryBytes},
            [&](const Json &transaction,const CommitReceipt &commit) {
                if(transaction.at("schema").str()!="fgm-search-transaction-v1")throw std::runtime_error("not a search workflow journal");
                const auto &binding=transaction.at("workflow");
                if(binding.at("domain").str()!=domainOf(config)||binding.at("mode").str()!=config.policy->resolved().at("mode").str()||
                    dump(binding.at("dimensions"))!=dump(config.policy->resolved().at("dimensions")))throw std::runtime_error("resume workflow mismatch");
                if(config.policy->mode==ControlledConfig::Mode::Alternatives&&uint64_t(binding.at("collection_rank").num())!=config.policy->anchor)
                    throw std::runtime_error("resume collection rank mismatch");
                std::set<std::string> credited(commit.creditedIds.begin(),commit.creditedIds.end());
                for(const auto &entry:transaction.at("admissions").array) {
                    AdmissionContext verifier(config.limits);auto scheme=verifier.fromJson(entry.at("scheme"));
                    if(!verifier.verify(scheme)||verifier.identity(scheme,true)!=entry.at("scheme_id").str()||
                        scheme.rank!=entry.at("rank").num()||(scheme.f2?"F2":"ZT")!=entry.at("domain").str()||
                        entry.at("domain").str()!=domainOf(config))throw std::runtime_error("invalid historical admission binding");
                    for(size_t p=0;p<3;++p)if(scheme.n[p]!=config.policy->dimensions[p])throw std::runtime_error("historical dimensions mismatch");
                    if(credited.count(entry.at("scheme_id").str())&&(config.policy->mode==ControlledConfig::Mode::RankReduction||scheme.rank==config.policy->anchor)) {
                        run.historicalDiscoveries=configCheckedAdd(run.historicalDiscoveries,1);
                        if(commit.acknowledged)acknowledgedDiscoveries=configCheckedAdd(acknowledgedDiscoveries,1);
                    }
                }
                if(transaction.has("observations"))for(const auto &entry:transaction.at("observations").array) {
                    AdmissionContext verifier(config.limits);auto scheme=verifier.fromJson(entry.at("scheme"));
                    if(!verifier.verify(scheme)||verifier.identity(scheme,true)!=entry.at("scheme_id").str()||verifier.identity(scheme,false)!=entry.at("factors_id").str())
                        throw std::runtime_error("invalid historical capture binding");
                }
                run.recoveredPools=transaction.at("pools");stage=uint64_t(transaction.at("stage").num());
                if(commit.acknowledged){acknowledgedSequence=commit.sequence;acknowledgedHead=commit.hash;}
            },[&](uint64_t bytes){context.accountRead(bytes);});
        if(!recovered.sequence||run.recoveredPools.kind==Json::Null)throw std::runtime_error("journal has no committed workflow history");
        run.recoveredHead=recovered.hash;
        if(config.policy->mode==ControlledConfig::Mode::RankReduction) {
            if(!stage||stage>350)throw std::runtime_error("invalid recovered stage");
            run.config.policy->anchor=stage;run.config.resolved.object["policy"]=run.config.policy->resolved();
        }
        RankPools checked(config.pool);checked.restore(run.recoveredPools,context);
        std::set<std::string> unbound;
        for(const auto *kind:{"active","reserves"})for(const auto &rank:run.recoveredPools.at(kind).array)
            for(const auto &member:rank.at("members").array) {
                AdmissionContext verifier(config.limits);const auto scheme=verifier.fromJson(member.at("scheme"));
                if((scheme.f2?"F2":"ZT")!=domainOf(config))throw std::runtime_error("pool history domain mismatch");
                for(size_t p=0;p<3;++p)if(scheme.n[p]!=config.policy->dimensions[p])throw std::runtime_error("pool history dimensions mismatch");
                unbound.insert(member.at("scheme_id").str());
            }
        const auto boundHead=Journal::replayReadOnly(config.input.journal,
            {config.history.storageBytes,config.history.transactionBytes,config.history.indexMemoryBytes},
            [&](const Json &transaction,const CommitReceipt &){for(const auto &entry:transaction.at("admissions").array)unbound.erase(entry.at("scheme_id").str());},
            [&](uint64_t bytes){context.accountRead(bytes);});
        if(boundHead.hash!=run.recoveredHead||!unbound.empty())throw std::runtime_error("pool refers to missing committed history");
        checked.refill(uint32_t(config.policy->anchor));run.recoveredPools=checked.snapshot(context);
        for(const auto &rank:run.recoveredPools.at("active").array) {
            if(uint64_t(rank.at("rank").num())!=config.policy->anchor)continue;
            for(const auto &member:rank.at("members").array) {
                AdmissionContext recordContext(config.limits);auto effective=recordContext.fromJson(member.at("scheme"));
                Json report=Json::dict();report.object["scheme_id"]=Json(recordContext.identity(effective,true));
                report.object["effective_factors_id"]=Json(recordContext.identity(effective,false));
                report.object["factors_id"]=report.at("effective_factors_id");report.object["eligibility"]=recordContext.analyze(effective);
                report.object["verification"]=Json(effective.f2?"exact-F2":"exact-Z");report.object["source"]=Json("journal resume");
                admit({effective,effective,report,0});
            }
        }
    }
    if (run.inputs.empty()&&config.input.kind!=RunInput::Kind::Resume) throw std::runtime_error("no admitted run inputs");
    if(config.policy&&!config.policy->representationIneligibility(config.policy->anchor).empty())
        throw std::runtime_error("policy anchor or dimensions exceed execution representation");
    bool packedEligible = config.policy && domainOf(config) == "ZT" && config.execution.blockSize == 32;
    if (config.policy) for (auto dimension : config.policy->dimensions) packedEligible &= dimension == 3;
    if (config.execution.backend == "packed" && !packedEligible)
        throw std::runtime_error("configuration is not eligible for signed 3x3 packed execution");
    uint64_t allocation = 0;
    if (config.policy) {
        const auto schemeSize = domainOf(config) == "ZT" ? layout.signedSchemeBytes : layout.f2SchemeBytes;
        // Current, best, mandatory and optional capture states; local device
        // scratch/driver residency are reported separately from buffer bytes.
        allocation = configCheckedMultiply(config.execution.workers,
            configCheckedAdd(configCheckedMultiply(configCheckedAdd(3, config.policy->optionalQuota), schemeSize),
                configCheckedAdd(layout.controlBytes,configCheckedMultiply(configCheckedAdd(1,config.policy->optionalQuota),16))));
        if(packedEligible&&config.execution.backend!="general")allocation=configCheckedAdd(allocation,
            configCheckedMultiply((config.execution.workers+31)/32*32,26400));
    } else {
        allocation = configCheckedAdd(configCheckedMultiply(config.reduction->reducers + 1,
            layout.reducerLaneBytes), configCheckedMultiply(config.reduction->reducers, layout.rngBytes));
        if(config.reduction->maxFlips)allocation=configCheckedAdd(allocation,configCheckedMultiply(config.reduction->schemes,16));
        allocation=configCheckedAdd(allocation,configCheckedMultiply(config.reduction->reducers,3*sizeof(int)));
    }
    uint64_t reservedHost=0;
    if(config.policy)reservedHost=configCheckedAdd(configCheckedMultiply(config.pool.memoryBytes,2),
        configCheckedAdd(configCheckedMultiply(config.history.transactionBytes,4),config.history.indexMemoryBytes));
    else reservedHost=configCheckedMultiply(config.limits.record,128);
    if (configCheckedAdd(configCheckedAdd(allocation, admittedBytes),reservedHost) > config.execution.memoryBytes)
        throw Resource("planned buffers and admission exceed run memory budget");
    Json receipt = Json::dict();
    receipt.object["schema"] = Json("fgm-run-record-v1");
    receipt.object["status"] = Json("validated");
    receipt.object["execution_started"] = Json(false);
    receipt.object["configuration_sha256"] = Json(sha256Bytes(bytes));
    receipt.object["configuration"] = config.resolved;
    receipt.object["specification"] = Json("FGM-CONTRACT-v1");
    receipt.object["specification_sha256"] = Json("b0ecaf3ae1e6da9eb0cfb36bee4454262be5f414eec15e3251f51af5bda54d34");
    receipt.object["arithmetic_reference"] = Json("9ea5bfc144b528184c32c178cae0b49bc60e8e3d");
    receipt.object["requested_backend"] = Json(config.execution.backend);
    receipt.object["actual_backend"] = Json();
    receipt.object["packed_eligible"] = Json(packedEligible);
    receipt.object["planned_buffer_bytes"] = integer(allocation);
    receipt.object["admission_content_bytes"] = integer(admittedBytes);
    receipt.object["reserved_host_bytes"]=integer(reservedHost);
    receipt.object["memory_accounting"] = Json("planned shared buffers and accounted owned input allocations; excludes allocator bookkeeping, driver residency and device scratch; not process RSS");
    receipt.object["input_read_bytes"] = integer(context.scannedBytes());
    receipt.object["presentations"] = std::move(presentations);
    receipt.object["admitted_inputs"] = integer(run.inputs.size());
    receipt.object["seed_duplicates"] = integer(duplicates);
    if(config.input.kind==RunInput::Kind::Resume) {
        receipt.object["journal_sequence"]=integer(acknowledgedSequence);
        receipt.object["journal_head_sha256"]=Json(acknowledgedHead);
    }
    Json counters = Json::dict();
    for (auto key : {"flip_attempts", "flips_applied", "control_steps", "reduction_attempts", "terms_removed",
                    "expansion_attempts", "expansions_applied", "tuple_rejections", "coefficient_rejections",
                    "proposal_exhaustions", "rank_blocked", "mandatory_captures", "optional_captures", "capture_drops",
                    "discoveries_current_run", "discoveries_historical"}) counters.object[key] = integer(0);
    counters.object["discoveries_historical"]=integer(acknowledgedDiscoveries);
    receipt.object["counters"] = counters;
    run.receipt = std::move(receipt);
    if (readBounded(configPath, 1048576) != bytes) throw std::runtime_error("run configuration changed during admission");
    return run;
}

int runConfigured(int argc, char **argv, const std::string &operation, const std::string &domain, const ExecutionLayout &layout,void (*execute)(PreparedRun &)) {
    const auto started=std::chrono::steady_clock::now();
    std::filesystem::path configuration;
    bool validateOnly = false;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--run-config" && configuration.empty() && i + 1 < argc) configuration = argv[++i];
        else if (option == "--validate-only" && !validateOnly) validateOnly = true;
        else throw std::runtime_error("run configuration conflicts with option: " + option);
    }
    if (configuration.empty()) throw std::runtime_error("--run-config requires a file");
    auto run = prepareRun(configuration, layout, operation, domain);
    const auto admissionFinished=std::chrono::steady_clock::now();
    run.receipt.object["admission_microseconds"]=integer(uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(admissionFinished-started).count()));
    for(const auto *phase:{"setup_microseconds","dispatch_microseconds","verification_microseconds","persistence_microseconds"})
        run.receipt.object[phase]=integer(0);
    if (!validateOnly && !execute) throw std::runtime_error("this executable only supports configuration validation");
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> executable(size);
    if (!size || _NSGetExecutablePath(executable.data(), &size)) throw std::runtime_error("cannot resolve native executable");
    AdmissionContext identity;
    run.receipt.object["executable_sha256"] = Json(identity.sha256File(std::filesystem::canonical(executable.data())));
#ifdef METAL_LIBRARY_SHA256
    run.receipt.object["library_sha256"] = Json(METAL_LIBRARY_SHA256);
    run.receipt.object["library_mode"] = Json("metallib");
#elif defined(METAL_SOURCE_DIR)
    run.receipt.object["library_mode"] = Json("source");
#else
    run.receipt.object["library_mode"] = Json("host-only");
#endif
    if(!validateOnly) {
        const auto executionStarted=std::chrono::steady_clock::now();
        auto finishExecutionTiming=[&] {
            const auto elapsed=uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-executionStarted).count());
            uint64_t measured=0;
            for(const auto *phase:{"dispatch_microseconds","verification_microseconds","persistence_microseconds"})
                measured+=uint64_t(run.receipt.at(phase).num());
            run.receipt.object.at("setup_microseconds")=integer(elapsed>measured?elapsed-measured:0);
        };
        try {execute(run);finishExecutionTiming();}
        catch(const std::exception &error) {
            finishExecutionTiming();
            run.receipt.object["status"]=Json("failed");run.receipt.object["error"]=Json(error.what());
            publishNew(run.config.output,dump(run.receipt)+"\n");
            throw;
        }
    }
    run.receipt.object["through_final_journal_commit_microseconds"]=integer(uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-started).count()));
    publishNew(run.config.output, dump(run.receipt) + "\n");
    std::cout << (validateOnly?"Validated native run: ":"Completed native run: ") << run.config.output << '\n';
    return 0;
}

} // namespace fgm
