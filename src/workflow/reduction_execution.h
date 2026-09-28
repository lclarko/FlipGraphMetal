#pragma once
// Included after native reducer declarations; no Python or child processes.
#include "execution.h"

namespace fgm {
// The reducer is also called by legacy entry points. Controlled execution
// installs receipt fields only for the duration of its bounded call.
inline thread_local Json *reductionDispatchField=nullptr;
inline thread_local Json *reductionVerificationField=nullptr;
struct ReductionTimingBinding {
    ReductionTimingBinding(Json &dispatch,Json &verification) {
        reductionDispatchField=&dispatch;reductionVerificationField=&verification;
    }
    ~ReductionTimingBinding() { reductionDispatchField=nullptr;reductionVerificationField=nullptr; }
};
struct ReductionPhaseTimer {
    uint64_t &local;
    Json *receipt;
    std::chrono::steady_clock::time_point started=std::chrono::steady_clock::now();
    ReductionPhaseTimer(uint64_t &value,Json *field):local(value),receipt(field) {}
    ~ReductionPhaseTimer() {
        auto elapsed=uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now()-started).count());
        local+=elapsed;
        if(receipt)receipt->integer+=int64_t(elapsed);
    }
};
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
// Evaluator phases release their shared allocations before the next phase.
template<class T> struct ReductionBuffer {
    T *data=nullptr;
    explicit ReductionBuffer(size_t count) {
        if(count>SIZE_MAX/sizeof(T)) throw Resource("reduction allocation size overflow");
        metalAllocate(&data,count*sizeof(T));
    }
    ~ReductionBuffer() { if(data) metalFree(data); }
    ReductionBuffer(const ReductionBuffer&)=delete;
    ReductionBuffer& operator=(const ReductionBuffer&)=delete;
};
struct CircuitStage { Json outputs=Json::list(), fresh=Json::list(); };
inline Json circuitTerm(int wire) {
    if(!wire) throw std::runtime_error("zero wire reference");
    auto term=Json::dict(); term.object["index"]=Json(int64_t(std::abs(wire)-1));
    term.object["value"]=Json(int64_t(wire<0?-1:1)); return term;
}
inline Json circuitExpression(const std::vector<int> &wires) {
    auto result=Json::list(); for(int wire:wires) result.array.push_back(circuitTerm(wire)); return result;
}
inline uint64_t verifyStage(const CircuitStage &stage,const Matrix &target,int inputs,const AdmissionLimits &limits) {
    AdmissionContext verifier(limits); uint64_t count=0;
    if(verifier.reconstructStage(stage.outputs,stage.fresh,inputs,count)!=target)
        throw std::runtime_error("constructed stage changed exact ordered factors");
    return count;
}
inline CircuitStage constructorWitness(const fgm_constructor::Problem &p,const fgm_constructor::Witness &w,int candidate) {
    using namespace fgm_constructor;
    if(w.status<0 || w.status>2 || w.count<0 || w.count>MaxGates)
        throw std::runtime_error("invalid GPU constructor witness bounds");
    uint64_t available=(uint64_t(1)<<p.inputs)-1;
    int wires[MaxDirections]{}; Matrix vectors(MaxDirections,std::vector<int64_t>(p.inputs));
    for(int i=0;i<p.directions;++i) for(int c=0;c<p.inputs;++c) vectors[i][c]=p.vectors[i][c];
    for(int i=0;i<p.inputs;++i) wires[i]=i+1;
    if(candidate>=0 && p.candidates[candidate].out>=0) {
        auto g=p.candidates[candidate];
        for(int c=0;c<p.inputs;++c) vectors[p.directions][c]=g.leftSign*vectors[g.left][c]+g.rightSign*vectors[g.right][c];
    }
    CircuitStage stage;
    for(int i=0;i<w.count;++i) {
        auto g=w.gates[i];
        if(g.out<p.inputs || g.out>p.directions || g.left<0 || g.right<0 ||
           g.left>p.directions || g.right>p.directions || g.left==g.right ||
           std::abs(g.leftSign)!=1 || std::abs(g.rightSign)!=1 || (available&(uint64_t(1)<<g.out)) ||
           !(available&(uint64_t(1)<<g.left)) || !(available&(uint64_t(1)<<g.right)))
            throw std::runtime_error("unreachable GPU constructor dependency");
        if(g.out==p.directions) {
            if(candidate<0) throw std::runtime_error("target-only witness created auxiliary");
            auto expected=p.candidates[candidate];
            if(g.out!=expected.out || g.left!=expected.left || g.right!=expected.right ||
               g.leftSign!=expected.leftSign || g.rightSign!=expected.rightSign)
                throw std::runtime_error("auxiliary prerequisites changed");
        }
        for(int c=0;c<p.inputs;++c)
            if(vectors[g.out][c]!=g.leftSign*vectors[g.left][c]+g.rightSign*vectors[g.right][c])
                throw std::runtime_error("incorrect GPU constructor gate");
        stage.fresh.array.push_back(circuitExpression({g.leftSign*wires[g.left],g.rightSign*wires[g.right]}));
        wires[g.out]=p.inputs+i+1; available|=uint64_t(1)<<g.out;
    }
    const bool complete=(available&((uint64_t(1)<<p.directions)-1))==((uint64_t(1)<<p.directions)-1);
    if(available!=w.available || (w.status!=2 && (w.status==1)!=complete) ||
       (w.status==2 && (candidate<0 || p.candidates[candidate].out>=0 || w.count)))
        throw std::runtime_error("GPU constructor completion mismatch");
    if(w.status==1) for(int i=0;i<p.outputs;++i) {
        const int direction=p.outputDirection[i];
        stage.outputs.array.push_back(circuitExpression(direction<0?std::vector<int>{}:std::vector<int>{wires[direction]*p.outputSign[i]}));
    }
    return stage;
}
inline std::optional<CircuitStage> constructStageGPU(const Matrix &target,uint64_t population,int block,
                                                    Json &report,uint64_t &dispatch) {
    using namespace fgm_constructor;
    ReductionBuffer<Problem> problem(1); *problem.data=prepare(target,9);
    const auto &p=*problem.data;
    const int batch=int(std::min<uint64_t>(population,MaxCandidates));
    ReductionBuffer<Witness> witnesses(batch);
    report=Json::dict(); report.object["target_directions"]=Json(int64_t(p.directions-p.inputs));
    report.object["candidate_slots"]=Json(int64_t(p.candidateCount));
    report.object["candidates_evaluated"]=Json(int64_t(0));
    report.object["constructed_count"]=Json(); report.object["witness_candidate"]=Json();
    auto launch=[&](int first,int count) {
        ReductionPhaseTimer timer(dispatch,reductionDispatchField);
        metalDispatch("constructorClosureKernel",size_t((count+block-1)/block)*block,block,problem.data,witnesses.data,first,count);
    };
    launch(-1,1);
    auto stage=constructorWitness(p,witnesses.data[0],-1);
    if(witnesses.data[0].status==1) {
        report.object["status"]=Json("target-only"); report.object["auxiliary_search"]=Json("skipped");
        report.object["constructed_count"]=Json(int64_t(witnesses.data[0].count)); return stage;
    }
    for(int first=0;first<p.candidateCount;first+=batch) {
        int count=std::min(batch,p.candidateCount-first); launch(first,count);
        report.object["candidates_evaluated"].integer+=count;
        std::optional<CircuitStage> best; int selected=-1;
        for(int i=0;i<count;++i) {
            auto current=constructorWitness(p,witnesses.data[i],first+i);
            if(witnesses.data[i].status==1 && !best) { best=std::move(current); selected=i; }
        }
        if(best) {
            report.object["status"]=Json("one-auxiliary"); report.object["auxiliary_search"]=Json("witness-found");
            report.object["constructed_count"]=Json(int64_t(witnesses.data[selected].count));
            report.object["witness_candidate"]=Json(int64_t(first+selected)); return best;
        }
    }
    report.object["status"]=Json("family-exhausted"); report.object["auxiliary_search"]=Json("complete"); return {};
}
// Reverse a binary linear DAG. No cost identity is used: the returned circuit
// is counted by the same exact reconstruction used for all native circuits.
inline CircuitStage transposeStage(const CircuitStage &source,int inputs) {
    std::vector<std::array<int,2>> gates;
    auto wire=[](const Json &term) { return int(term.at("value").num())*(int(term.at("index").num())+1); };
    for(const auto &gate:source.fresh.array) gates.push_back({wire(gate.array.at(0)),wire(gate.array.at(1))});
    std::vector<int> outputs;
    for(const auto &output:source.outputs.array) {
        int current=0;
        for(const auto &term:output.array) {
            int next=wire(term);
            if(!current) current=next;
            else { gates.push_back({current,next}); current=inputs+int(gates.size()); }
        }
        outputs.push_back(current);
    }
    std::vector<std::vector<int>> contributions(inputs+gates.size());
    for(size_t i=0;i<outputs.size();++i) if(outputs[i])
        contributions[std::abs(outputs[i])-1].push_back((outputs[i]<0?-1:1)*int(i+1));
    CircuitStage result;
    auto combine=[&](const std::vector<int> &terms) {
        int current=0;
        for(int next:terms) {
            if(!current) current=next;
            else { result.fresh.array.push_back(circuitExpression({current,next})); current=int(outputs.size()+result.fresh.array.size()); }
        }
        return current;
    };
    for(int node=int(contributions.size())-1;node>=inputs;--node) {
        int current=combine(contributions[node]);
        if(current) for(int operand:gates[node-inputs]) contributions[std::abs(operand)-1].push_back((operand<0?-1:1)*current);
    }
    for(int i=0;i<inputs;++i) {
        int current=combine(contributions[i]); result.outputs.array.push_back(circuitExpression(current?std::vector<int>{current}:std::vector<int>{}));
    }
    return result;
}
} // namespace fgm
#include "two_aux_execution.h"
namespace fgm {
inline uint32_t transposeSeed(uint32_t seed) {
    uint32_t value=seed^0xa511e9b3u; value^=value>>16; value*=0x7feb352du;
    value^=value>>15; value*=0x846ca68bu; value^=value>>16; return value?value:1;
}
inline CircuitStage pairTransposeGPU(const Matrix &target,const ReductionSettings &settings,int block,
                                     uint64_t &dispatch) {
    using PairReducer=fgm_constructor::PairReducer;
    const int count=int(settings.reducers); const uint32_t seed=transposeSeed(settings.seed);
    ReductionBuffer<PairReducer> reducers(count+1); ReductionBuffer<RandomState> states(count);
    ReductionBuffer<int> factors(23*9);
    for(int r=0;r<23;++r) for(int c=0;c<9;++c) factors.data[r*9+c]=int(target[r][c]);
    int best=INT32_MAX,bestFresh=INT32_MAX;
    for(uint64_t round=0;round<settings.rounds;++round) {
        { ReductionPhaseTimer timer(dispatch,reductionDispatchField);
          metalDispatch("transposePairKernel",size_t((count+block-1)/block)*block,block,reducers.data,factors.data,states.data,count,seed,round==0); }
        for(int i=0;i<count;++i) {
            if(!reducers.data[i].isValid()) throw std::runtime_error("invalid transposed pair reducer");
            int cost=reducers.data[i].getAdditions(), fresh=reducers.data[i].getFreshVars();
            if(cost<best || (cost==best && fresh<bestFresh)) { best=cost; bestFresh=fresh; reducers.data[count].copyFrom(reducers.data[i]); }
        }
    }
    std::ostringstream encoded; encoded<<'{'; reducers.data[count].write(encoded,"x",""); encoded<<'}';
    auto parsed=Parser(encoded.str()).parse(); return {parsed.at("x"),parsed.at("x_fresh")};
}
inline void extendReduction(Json &result,const SchemeRecord &effective,const ReductionSettings &settings,
                            int block,const AdmissionLimits &limits,Json &receipt) {
    using PairReducer=fgm_constructor::PairReducer;
    using U=AdditionsReducer<350,250,32,2016>;
    using W=AdditionsReducer<64,500,175,61075>;
    static_assert(sizeof(fgm_constructor::Problem)+sizeof(fgm_constructor::Witness)+sizeof(int)*1024<sizeof(U)+sizeof(U)+sizeof(W));
    static_assert(sizeof(PairReducer)+23*9*sizeof(int)+sizeof(int)*1024<sizeof(U)+sizeof(U)+sizeof(W));
    result.object["strategy"]=Json(settings.strategy);
    result.object["baseline_terminal_reason"]=result.at("terminal_reason");
    result.object["baseline_additions"]=result.at("verified_circuit_additions");
    result.object["baseline_additions_by_stage"]=result.at("verified_circuit_additions_by_stage");
    auto &sources=result.object["stage_sources"]; sources=Json::dict();
    auto &reports=result.object["construction"]; reports=Json::dict();
    for(const auto *key:{"u","v","w"}) sources.object[key]=Json("baseline");
    for(const auto *key:{"u","v","wt"}) { auto unused=Json::dict(); unused.object["status"]=Json("not-requested"); reports.object[key]=unused; }
    if(settings.strategy=="baseline") return;
    auto &phases=result.object.at("phase_microseconds");
    uint64_t dispatch=0,verification=0;
    Matrix directW(9,std::vector<int64_t>(23));
    for(int i=0;i<9;++i) for(int r=0;r<23;++r) directW[i][r]=effective.f[2][r][i];
    auto select=[&](const CircuitStage &stage,int p,const char *source) {
        ReductionPhaseTimer timer(verification,reductionVerificationField);
        std::string key(1,"uvw"[p]); uint64_t cost=verifyStage(stage,p==2?directW:effective.f[p],p==2?23:9,limits);
        if(cost<uint64_t(result.at("verified_circuit_additions_by_stage").at(key).num())) {
            result.object["verified_circuit_additions_by_stage"].object[key]=Json(int64_t(cost));
            result.object["circuit"].object[key]=stage.outputs; result.object["circuit"].object[key+"_fresh"]=stage.fresh;
            sources.object[key]=Json(source);
        }
    };
    auto transpose=[&](const CircuitStage &stage,const char *source) {
        { ReductionPhaseTimer timer(verification,reductionVerificationField); verifyStage(stage,effective.f[2],9,limits); }
        CircuitStage output;
        { ReceiptPhaseTimer timer(phases.object.at("transposition")); output=transposeStage(stage,9); }
        select(output,2,source);
    };
    if(settings.strategy=="transpose" || settings.strategy=="combined") {
        CircuitStage stage;
        { ReceiptPhaseTimer timer(phases.object.at("transpose_pair")); stage=pairTransposeGPU(effective.f[2],settings,block,dispatch); }
        result.object["transpose_seed"]=Json(int64_t(transposeSeed(settings.seed)));
        result.object["transpose_rounds"]=Json(int64_t(settings.rounds)); transpose(stage,"transpose-pair");
    }
    if(settings.strategy=="cancellation" || settings.strategy=="combined") {
        for(int p=0;p<(settings.strategy=="combined"?3:2);++p) {
            const std::string key=p==0?"u":p==1?"v":"wt";
            std::optional<CircuitStage> stage;
            { ReceiptPhaseTimer timer(phases.object.at("construction_"+key));
              stage=constructStageGPU(effective.f[p],settings.reducers,block,reports.object.at(key),dispatch); }
            if(stage) { if(p==2) transpose(*stage,"transpose-cancellation"); else select(*stage,p,"cancellation"); }
        }
    }
    if(settings.constructor) {
        result.object["pre_two_aux_additions_by_stage"]=result.at("verified_circuit_additions_by_stage");
        auto &active=receipt.object["two_auxiliary"]; active=Json::dict();
        AdmissionContext identity(limits);
        active.object["effective_factors_id"]=Json(identity.identity(effective,false));
        active.object["stages"]=Json::dict();
        for(int p=0;p<3;++p) {
            const std::string key=p==0?"u":p==1?"v":"wt", outputKey(1,"uvw"[p]);
            auto &report=active.object["stages"].object[key];
            report=twoAuxReport(*settings.constructor,key);
            report.object["baseline_cost"]=result.at("baseline_additions_by_stage").at(outputKey);
            const uint64_t incumbent=uint64_t(result.at("verified_circuit_additions_by_stage").at(outputKey).num());
            report.object["incumbent_cost"]=Json(int64_t(incumbent));
            report.object["final_cost"]=Json(int64_t(incumbent));
            auto &phase=phases.object["two_aux_"+key]; phase=Json(int64_t(0));
            ReceiptPhaseTimer phaseTimer(phase);
            try {
                auto prepared=prepareTwoAuxStage(effective.f[p],9,p==2,report);
                if(reports.at(key).at("status").str()!="family-exhausted") {
                    report.object["stop_reason"]=Json("smaller-family-succeeded");
                    continue;
                }
                if(incumbent<=uint64_t(report.at("improvement_floor").num())) {
                    report.object["stop_reason"]=Json("proved-no-improvement");
                    continue;
                }
                auto stage=constructTwoAuxStageGPU(prepared,effective.f[p],p==2,incumbent,
                    *settings.constructor,block,limits,report,dispatch,verification);
                if(stage) select(*stage,p,p==2?"transpose-two-auxiliary":"two-auxiliary");
                report.object["final_cost"]=result.at("verified_circuit_additions_by_stage").at(outputKey);
            } catch(const std::exception &) {
                report.object["stop_reason"]=Json("invocation-failed");
                throw;
            }
        }
        result.object["two_auxiliary"]=active;
    }
    { ReductionPhaseTimer timer(verification,reductionVerificationField);
      int64_t total=0; for(const auto &stage:result.at("verified_circuit_additions_by_stage").object) total+=stage.second.num();
      result.object["verified_circuit_additions"]=Json(total);
      result.object["circuit"].object["complexity"].object["reduced"]=Json(total);
      auto verified=verifyReductionCircuit(result.at("circuit"),effective,limits,true);
      if(verified.operations!=uint64_t(total)) throw std::runtime_error("extension count mismatch"); }
    phases.object["verification"]=Json(int64_t(verification));
    result.object["dispatch_microseconds"].integer+=int64_t(dispatch);
    result.object["verification_microseconds"].integer+=int64_t(verification);
}
inline Json evaluateReduction(const SchemeRecord &effective,const ReductionSettings &settings,uint64_t blockSize,
                              const AdmissionLimits &limits,Json &receipt) {
    Json result, phases=Json::dict();
    for(const auto *key:{"baseline","transpose_pair","construction_u","construction_v","construction_wt","transposition","verification"})
        phases.object[key]=Json(int64_t(0));
    ReductionTimingBinding timing(receipt.object.at("dispatch_microseconds"),receipt.object.at("verification_microseconds"));
    { ReceiptPhaseTimer timer(phases.object.at("baseline"));
      SchemeAdditionsReducer reducer(int(settings.reducers),int(settings.schemes),int(settings.maxFlips),
          int(settings.seed),int(blockSize),"",1);
      if(!reducer.read(effective)) throw std::runtime_error("effective input exceeds signed reducer capacity");
      result=reducer.reduceBounded(settings.rounds,settings.noImprovements,settings.targetAdditions,limits,effective); }
    result.object["phase_microseconds"]=std::move(phases);
    extendReduction(result,effective,settings,int(blockSize),limits,receipt);
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
    const auto workspace=configCheckedAdd(reductionWorkspaceBytes(run.config.limits.record),settings.constructor?TwoAuxPreparationBytes:0);
    for(const auto &input:run.inputs) {
        const auto retained=configCheckedAdd(resultBytes,circuits.capacity());
        if(configCheckedAdd(configCheckedAdd(fixedBytes,retained),workspace)>run.config.execution.memoryBytes)
            throw Resource("reduction workspace exceeds run memory budget");
        Json result=evaluateReduction(input.effective,settings,run.config.execution.blockSize,run.config.limits,run.receipt);
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
    { ReceiptPhaseTimer timer(run.receipt.object.at("persistence_microseconds"));
      publishNew(circuitPath,circuits); }
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
