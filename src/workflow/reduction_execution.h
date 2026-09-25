#pragma once
// Included after native reducer declarations; no Python or child processes.
#include "execution.h"

namespace fgm {
// This stream checks the record bound before every append. ostream badbit
// exceptions preserve Resource from overflow/xsputn instead of swallowing it.
class ReductionRecordBuffer : public std::streambuf {
    std::string bytes;
    uint64_t limit;
protected:
    std::streamsize xsputn(const char *data,std::streamsize count) override {
        if(count<0 || uint64_t(count)>limit-bytes.size()) throw Resource("best circuit exceeds record budget");
        bytes.append(data,size_t(count)); return count;
    }
    int_type overflow(int_type value) override {
        if(traits_type::eq_int_type(value,traits_type::eof())) return traits_type::not_eof(value);
        if(bytes.size()>=limit) throw Resource("best circuit exceeds record budget");
        bytes.push_back(traits_type::to_char_type(value)); return value;
    }
public:
    explicit ReductionRecordBuffer(uint64_t bound):limit(bound) {}
    const std::string &str() const { return bytes; }
};
inline uint64_t reductionWorkspaceBytes(uint64_t recordBytes) {
    // Numeric circuit JSON uses multiple map/vector nodes per encoded term.
    // 128 bytes per input byte conservatively reserves simultaneous stream
    // growth, parsed JSON, reconstruction matrices, identities and output JSON.
    return configCheckedMultiply(recordBytes,128);
}
inline SchemeRecord verifyReductionCircuit(const Json &circuit,const SchemeRecord &effective,
                                          const AdmissionLimits &limits,bool fixed) {
    AdmissionContext verifier(limits);
    auto result=verifier.fromJson(circuit,"ZT");
    if(!result.circuit || !verifier.verify(result)) throw std::runtime_error("best circuit fails exact signed reconstruction/tensor verification");
    if(result.n!=effective.n || result.f2) throw std::runtime_error("best circuit domain or dimensions changed");
    if(fixed && verifier.identity(result,false)!=verifier.identity(effective,false))
        throw std::runtime_error("fixed reduction circuit changed effective input factors");
    return result;
}
inline void executeReduction(PreparedRun &run) {
    if(run.config.operation!="reduce" || !run.config.reduction) throw std::runtime_error("reduction configuration required");
    const auto &settings=*run.config.reduction;
    run.receipt.object["results"]=Json::list();
    run.receipt.object["execution_started"]=Json(true);
    run.receipt.object["actual_backend"]=Json("general");
    uint64_t attempted=0,applied=0,rounds=0,resultBytes=0,presentation=0;
    std::string circuits;
    const std::filesystem::path circuitPath=run.config.output.string()+".circuits.jsonl";
    const uint64_t fixedBytes=configCheckedAdd(uint64_t(run.receipt.at("planned_buffer_bytes").num()),uint64_t(run.receipt.at("admission_content_bytes").num()));
    const auto workspace=reductionWorkspaceBytes(run.config.limits.record);
    for(const auto &input:run.inputs) {
        const auto retained=configCheckedAdd(resultBytes,circuits.capacity());
        if(configCheckedAdd(configCheckedAdd(fixedBytes,retained),workspace)>run.config.execution.memoryBytes)
            throw Resource("reduction workspace exceeds run memory budget");
        SchemeAdditionsReducer reducer(int(settings.reducers),int(settings.schemes),int(settings.maxFlips),
            int(settings.seed),int(run.config.execution.blockSize),"",1);
        if(!reducer.read(input.effective)) throw std::runtime_error("effective input exceeds signed reducer capacity");
        auto result=reducer.reduceBounded(settings.rounds,settings.noImprovements,settings.targetAdditions,run.config.limits,input.effective);
        result.object["submitted_factors_id"]=input.report.at("factors_id");
        result.object["effective_input_factors_id"]=input.report.at("effective_factors_id");
        result.object["input_presentation_index"]=Json(int64_t(presentation));
        result.object["circuit_record_index"]=Json(int64_t(presentation));
        result.object["source_sha256"]=input.report.at("source_sha256");
        result.object["mode"]=Json(settings.maxFlips?"flip-enabled":"fixed");
        attempted=configCheckedAdd(attempted,uint64_t(result.at("flip_attempts").num()));
        applied=configCheckedAdd(applied,uint64_t(result.at("flips_applied").num()));
        rounds=configCheckedAdd(rounds,uint64_t(result.at("rounds_completed").num()));
        auto binding=Json::dict();
        for(const auto *key:{"input_presentation_index","submitted_factors_id","effective_input_factors_id","source_sha256"})
            binding.object[key]=result.at(key);
        result.object["circuit"].object["source_binding"]=std::move(binding);
        std::string line=dump(result.at("circuit"))+"\n";
        if(line.size()>run.config.limits.record) throw Resource("circuit JSONL record exceeds record budget");
        const uint64_t required=configCheckedAdd(circuits.size(),line.size());
        // Include live JSON and both strings, plus a conservative allowance for
        // string growth while replacing the old allocation.
        uint64_t transient=configCheckedAdd(workspace,circuits.capacity());
        transient=configCheckedAdd(transient,configCheckedAdd(configCheckedMultiply(required,2),256));
        if(configCheckedAdd(configCheckedAdd(fixedBytes,resultBytes),transient)>run.config.execution.memoryBytes)
            throw Resource("circuit serialization exceeds run memory budget");
        circuits.reserve(size_t(required));
        circuits+=line;
        result.object.erase("circuit");
        resultBytes=configCheckedAdd(resultBytes,jsonMemoryBytes(result));
        if(configCheckedAdd(configCheckedAdd(fixedBytes,resultBytes),circuits.capacity())>run.config.execution.memoryBytes)
            throw Resource("retained reduction results exceed memory budget");
        ++presentation;
        run.receipt.object["results"].array.push_back(std::move(result));
    }
    if(attempted>INT64_MAX || applied>INT64_MAX || rounds>INT64_MAX) throw Resource("reduction run counters exceed JSON integer capacity");
    publishNew(circuitPath,circuits);
    auto artifact=Json::dict();
    artifact.object["path"]=Json(circuitPath.string());
    artifact.object["sha256"]=Json(sha256Bytes(circuits));
    artifact.object["format"]=Json("jsonl");
    artifact.object["records"]=Json(int64_t(presentation));
    run.receipt.object["circuit_artifact"]=std::move(artifact);
    run.receipt.object["status"]=Json("complete");
    run.receipt.object["terminal_reason"]=Json("finite_selection_complete");
    auto &metrics=run.receipt.object["counters"];
    if(metrics.kind!=Json::Object) metrics=Json::dict();
    metrics.object["flip_attempts"]=Json(int64_t(attempted)); metrics.object["flips_applied"]=Json(int64_t(applied));
    metrics.object["reduction_rounds"]=Json(int64_t(rounds));
    run.receipt.object["verification"]=Json("native exact-Z best-circuit reconstruction and factor binding");
}
}
