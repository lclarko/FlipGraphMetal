#pragma once
// Included after the shared stage verifier and literal transposer.
#include "two_aux_validation.h"

namespace fgm {
inline Json twoAuxReport(const TwoAuxSettings &settings,const std::string &stage) {
    auto report=Json::dict();
    report.object["schema"]=Json("fgm-two-aux-report-v1");
    report.object["family"]=Json(settings.family);
    report.object["implementation"]=Json("metal-two-aux-v1");
    report.object["enumeration_order"]=Json("lexicographic-pairs-plus-first-v1");
    report.object["stage"]=Json(stage);
    report.object["requested_auxiliaries"]=Json(int64_t(2));
    report.object["max_pair_slots"]=Json(int64_t(settings.maxPairSlots));
    report.object["stop_reason"]=Json("in-progress");
    report.object["coverage"]=Json("not-searched");
    report.object["selected_witness"]=Json();
    for(const auto *key:{"raw_scanned","first_invalid","second_invalid","symmetry_filtered","prepared",
        "dispatched","completed","validated","gpu_rule_checks","gpu_sweeps","negative_validation_rule_checks",
        "logical_prefix","batch_tail_candidates","used_auxiliaries"}) report.object[key]=Json(int64_t(0));
    auto times=Json::dict();
    for(const auto *key:{"allocation","preparation","dispatch","gpu","witness_validation","transposition","verification"})
        times.object[key]=Json(int64_t(0));
    report.object["microseconds"]=std::move(times);
    return report;
}
inline void twoAuxCount(Json &report,const char *key,uint64_t count) {
    const auto updated=configCheckedAdd(uint64_t(report.at(key).num()),count);
    if(updated>INT64_MAX) throw Resource("two-auxiliary report counter overflow");
    report.object.at(key)=Json(int64_t(updated));
}
inline fgm_constructor::TwoAuxProblem prepareTwoAuxStage(const Matrix &target,int inputs,bool transpose,Json &report) {
    ReceiptPhaseTimer timer(report.object.at("microseconds").object.at("preparation"));
    auto p=fgm_constructor::prepareTwoAux(target,inputs);
    const auto total=fgm_constructor::twoAuxRawTotal(p);
    int q=0; for(int i=0;i<p.outputs;++i) q+=p.outputDirection[i]>=0;
    const int floor=p.directions-p.inputs+2+(transpose?q-p.inputs:0);
    report.object["target_directions"]=Json(int64_t(p.directions-p.inputs));
    report.object["nonzero_output_occurrences"]=Json(int64_t(q));
    report.object["improvement_floor"]=Json(int64_t(std::max(0,floor)));
    report.object["raw_family_size"]=Json(int64_t(total));
    report.object["prefix_limit"]=Json(int64_t(std::min(total,uint64_t(report.at("max_pair_slots").num()))));
    report.object["unvisited_raw_slots"]=Json(int64_t(total));
    return p;
}
inline void twoAuxProgress(Json &report,const fgm_constructor::TwoAuxStream &stream) {
    const auto &stats=stream.stats();
    report.object["raw_scanned"]=Json(int64_t(stats.rawScanned));
    report.object["first_invalid"]=Json(int64_t(stats.firstInvalid));
    report.object["second_invalid"]=Json(int64_t(stats.secondInvalid));
    report.object["symmetry_filtered"]=Json(int64_t(stats.symmetryFiltered));
    report.object["prepared"]=Json(int64_t(stats.prepared));
    report.object["unvisited_raw_slots"]=Json(int64_t(stream.rawTotal()-stats.rawScanned));
    report.object["coverage"]=Json(stats.rawScanned==stream.rawTotal()?"full":"partial");
}
struct TwoAuxBatchSelection {
    int lane=-1,usedHelpers=0;
    uint64_t cost=0;
    CircuitStage stage;
};
// Validate the whole batch before returning any selection. This is also the
// corruption-test seam; there is no early return on a successful lane.
inline TwoAuxBatchSelection validateTwoAuxBatch(const fgm_constructor::TwoAuxProblem &p,
        const fgm_constructor::TwoAuxTask *tasks,const fgm_constructor::TwoAuxWitness *witnesses,
        int count,const Matrix &target,bool transpose,bool requireTwoLive,const AdmissionLimits &limits,
        Json &report,uint64_t &verification) {
    using namespace fgm_constructor;
    if(count<1 || count>int(TwoAuxTileSize)) throw std::runtime_error("two-auxiliary batch capacity exceeded");
    TwoAuxBatchSelection best;
    Matrix expected;
    if(transpose) {
        expected=Matrix(p.inputs,std::vector<int64_t>(target.size()));
        for(size_t r=0;r<target.size();++r) for(int c=0;c<p.inputs;++c) expected[c][r]=target[r][c];
    }
    auto &times=report.object.at("microseconds");
    for(int lane=0;lane<count;++lane) {
        TwoAuxValidation checked;
        { ReceiptPhaseTimer timer(times.object.at("witness_validation"));
          ReductionPhaseTimer counted(verification,reductionVerificationField);
          checked=validateTwoAuxWitness(p,tasks[lane],witnesses[lane]); }
        uint64_t cost=0;
        if(witnesses[lane].status==1) {
            { ReceiptPhaseTimer timer(times.object.at("verification"));
              ReductionPhaseTimer counted(verification,reductionVerificationField);
              cost=verifyStage(checked.stage,target,p.inputs,limits);
              if(cost!=uint64_t(checked.liveCount)) throw std::runtime_error("two-auxiliary forward count mismatch"); }
            if(requireTwoLive && checked.usedHelpers!=2)
                throw std::runtime_error("two-auxiliary success contradicts smaller-family exhaustion");
            if(transpose) {
                CircuitStage output;
                { ReceiptPhaseTimer timer(times.object.at("transposition")); output=transposeStage(checked.stage,p.inputs); }
                { ReceiptPhaseTimer timer(times.object.at("verification"));
                  ReductionPhaseTimer counted(verification,reductionVerificationField);
                  cost=verifyStage(output,expected,int(target.size()),limits); }
                checked.stage=std::move(output);
            }
            if(requireTwoLive && cost!=uint64_t(report.at("improvement_floor").num()))
                throw std::runtime_error("two-auxiliary emitted count contradicts production bound");
            if(best.lane<0 || cost<best.cost) {
                best.lane=lane; best.cost=cost; best.usedHelpers=checked.usedHelpers; best.stage=std::move(checked.stage);
            }
        }
        twoAuxCount(report,"validated",1);
        twoAuxCount(report,"gpu_rule_checks",witnesses[lane].ruleChecks);
        twoAuxCount(report,"gpu_sweeps",uint64_t(witnesses[lane].sweeps));
        twoAuxCount(report,"negative_validation_rule_checks",checked.negativeRuleChecks);
    }
    return best;
}
inline Json twoAuxSelectedWitness(const fgm_constructor::TwoAuxProblem &p,
        const fgm_constructor::TwoAuxTask &task,const fgm_constructor::TwoAuxWitness &w) {
    auto result=Json::dict(),helpers=Json::list(),creations=Json::list(),gates=Json::list();
    auto gate=[](const fgm_constructor::Gate &g) {
        auto value=Json::list(); for(int x:{g.out,g.left,g.right,g.leftSign,g.rightSign}) value.array.emplace_back(int64_t(x)); return value;
    };
    for(int h=0;h<2;++h) {
        auto vector=Json::list(); for(int c=0;c<p.inputs;++c) vector.array.emplace_back(int64_t(task.helpers[h][c]));
        helpers.array.push_back(vector); creations.array.push_back(gate(task.creations[h]));
    }
    for(int i=0;i<w.count;++i) gates.array.push_back(gate(w.gates[i]));
    result.object["raw_slot"]=Json(int64_t(task.rawSlot));
    result.object["first_route"]=Json(int64_t(task.rawSlot/uint64_t(p.directions*(p.directions+1))));
    result.object["second_route"]=Json(int64_t(task.rawSlot%uint64_t(p.directions*(p.directions+1))));
    result.object["helpers"]=std::move(helpers); result.object["creations"]=std::move(creations);
    result.object["gates"]=std::move(gates); result.object["count"]=Json(int64_t(w.count));
    result.object["available"]=Json(int64_t(w.available));
    return result;
}
inline std::optional<CircuitStage> constructTwoAuxStageGPU(const fgm_constructor::TwoAuxProblem &p,
        const Matrix &target,bool transpose,uint64_t incumbent,const TwoAuxSettings &settings,int block,
        const AdmissionLimits &limits,Json &report,uint64_t &dispatch,uint64_t &verification,
        int tileSize=int(TwoAuxTileSize),bool requireTwoLive=true) {
    using namespace fgm_constructor;
    if(tileSize<1 || tileSize>int(TwoAuxTileSize) || block<1 || block>1024)
        throw std::runtime_error("two-auxiliary dispatch layout out of range");
    auto &times=report.object.at("microseconds");
    // Shared tiles are separate from the fixed host-preparation reservation.
    const auto allocationStarted=std::chrono::steady_clock::now();
    ReductionBuffer<TwoAuxProblem> problem(1); *problem.data=p;
    ReductionBuffer<TwoAuxTask> tasks(tileSize);
    ReductionBuffer<TwoAuxWitness> witnesses(tileSize);
    times.object.at("allocation")=Json(int64_t(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now()-allocationStarted).count()));
    TwoAuxStream stream(p,settings.maxPairSlots);
    std::optional<CircuitStage> best;
    uint64_t bestCost=incumbent;
    bool reachable=false;
    for(;;) {
        int count=0;
        { ReceiptPhaseTimer timer(times.object.at("preparation"));
          while(count<tileSize && stream.next(tasks.data[count])) ++count; }
        twoAuxProgress(report,stream);
        if(!count) break;
        twoAuxCount(report,"dispatched",uint64_t(count));
        { ReceiptPhaseTimer timer(times.object.at("dispatch"));
          ReductionPhaseTimer counted(dispatch,reductionDispatchField);
          const auto threads=configCheckedMultiply((uint64_t(count)+uint64_t(block)-1)/uint64_t(block),uint64_t(block));
          metalDispatch("constructorTwoAuxClosureKernel",size_t(threads),size_t(block),problem.data,tasks.data,witnesses.data,count); }
        times.object.at("gpu").integer+=int64_t(metalLastGpuNanoseconds/1000);
        twoAuxCount(report,"completed",uint64_t(count));
        auto selected=validateTwoAuxBatch(p,tasks.data,witnesses.data,count,target,transpose,requireTwoLive,limits,report,verification);
        report.object["logical_prefix"]=Json(int64_t(stream.stats().rawScanned));
        reachable |= selected.lane>=0;
        if(selected.lane>=0 && selected.cost<bestCost) {
            bestCost=selected.cost; best=std::move(selected.stage);
            const int lane=selected.lane;
            report.object["selected_witness"]=twoAuxSelectedWitness(p,tasks.data[lane],witnesses.data[lane]);
            report.object["used_auxiliaries"]=Json(int64_t(selected.usedHelpers));
            if(requireTwoLive || bestCost==0) {
                report.object["logical_prefix"]=Json(int64_t(tasks.data[lane].rawSlot+1));
                report.object["batch_tail_candidates"]=Json(int64_t(count-lane-1));
                report.object["stop_reason"]=Json("bound-attained");
                return best;
            }
        }
    }
    report.object["logical_prefix"]=Json(int64_t(stream.stats().rawScanned));
    report.object["stop_reason"]=Json(stream.limit()==stream.rawTotal()?(reachable?"family-covered":"family-exhausted"):"budget-exhausted");
    return best;
}
} // namespace fgm
