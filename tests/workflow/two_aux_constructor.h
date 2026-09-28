#pragma once

#include "../../src/workflow/two_aux_validation.h"

// Test-only closure. Production construction stays on Metal; M1 does not add a
// kernel or connect the new family to evaluateReduction.
inline fgm_constructor::TwoAuxWitness testTwoAuxClosure(
    const fgm_constructor::TwoAuxProblem &p,const fgm_constructor::TwoAuxTask &task) {
    using namespace fgm_constructor;
    TwoAuxWitness w{}; w.available=(uint64_t(1)<<p.inputs)-1;
    const uint64_t required=(uint64_t(1)<<p.directions)-1;
    bool changed=true;
    auto apply=[&](const Gate &g) {
        ++w.ruleChecks;
        if((w.available&(uint64_t(1)<<g.out)) || !(w.available&(uint64_t(1)<<g.left)) ||
           !(w.available&(uint64_t(1)<<g.right))) return;
        if(w.count>=TwoAuxMaxGates) throw std::runtime_error("test closure capacity exceeded");
        w.gates[w.count++]=g; w.available|=uint64_t(1)<<g.out; changed=true;
    };
    while(changed && (w.available&required)!=required) {
        changed=false; ++w.sweeps;
        for(int i=0;i<p.operationCount;++i) apply(p.operations[i]);
        for(int i=0;i<task.operationCount;++i) apply(task.operations[i]);
        for(const auto &g:task.creations) apply(g);
    }
    w.status=(w.available&required)==required?1:0;
    return w;
}

inline fgm::Json twoAuxGateJson(const fgm_constructor::Gate &gate) {
    auto result=fgm::Json::list();
    for(int x:{gate.out,gate.left,gate.right,gate.leftSign,gate.rightSign}) result.array.emplace_back(int64_t(x));
    return result;
}

inline fgm::Json twoAuxConstructorTest(const fgm::Json &request,bool gpu=false) {
    using namespace fgm;
    using namespace fgm_constructor;
    auto integer=[](const Json &x) {
        const int64_t value=x.num();
        if(value<INT32_MIN || value>INT32_MAX) throw std::runtime_error("test integer out of range");
        return int(value);
    };
    auto natural=[](const Json &x) {
        if(x.num()<0) throw std::runtime_error("negative test count");
        return uint64_t(x.num());
    };
    Matrix targets;
    for(const auto &row:request.at("targets").array) {
        std::vector<int64_t> values;
        for(const auto &x:row.array) values.push_back(x.num());
        targets.push_back(values);
    }
    const auto p=prepareTwoAux(targets,integer(request.at("inputs")));
    const uint64_t budget=request.has("max_pair_slots")?natural(request.at("max_pair_slots")):65536;
#ifdef FGM_CONSTRUCTOR_GPU_TEST
    if(gpu && request.has("gpu_stage")) {
        const bool transpose=request.has("transpose_two_aux") && request.at("transpose_two_aux").boolean;
        const bool requireTwoLive=request.has("require_two_live") && request.at("require_two_live").boolean;
        const int tile=request.has("tile_size")?integer(request.at("tile_size")):128;
        const TwoAuxSettings settings{"signed-two-aux-distinct-v1",budget};
        auto report=twoAuxReport(settings,"test");
        const auto stageProblem=prepareTwoAuxStage(targets,p.inputs,transpose,report);
        uint64_t dispatch=0,verification=0;
        const auto stage=constructTwoAuxStageGPU(stageProblem,targets,transpose,1000000,settings,32,
            AdmissionLimits{},report,dispatch,verification,tile,requireTwoLive);
        auto result=Json::dict();
        result.object["report"]=report;
        result.object["status"]=Json(int64_t(stage.has_value()));
        if(stage) {
            Matrix expected=targets;
            if(transpose) {
                expected=Matrix(p.inputs,std::vector<int64_t>(targets.size()));
                for(size_t r=0;r<targets.size();++r)
                    for(int c=0;c<p.inputs;++c) expected[c][r]=targets[r][c];
            }
            result.object["count"]=Json(int64_t(verifyStage(*stage,expected,
                transpose?int(targets.size()):p.inputs,AdmissionLimits{})));
            result.object["fresh"]=stage->fresh; result.object["outputs"]=stage->outputs;
        }
        return result;
    }
#else
    (void)gpu;
#endif
    TwoAuxStream stream(p,budget);
    auto response=Json::dict(), results=Json::list();
    response.object["directions"]=Json(int64_t(p.directions));
    response.object["raw_pair_slots"]=Json(int64_t(stream.rawTotal()));
    response.object["limit"]=Json(int64_t(stream.limit()));
    auto sizes=Json::dict();
    sizes.object["problem"]=Json(int64_t(sizeof(TwoAuxProblem)));
    sizes.object["task"]=Json(int64_t(sizeof(TwoAuxTask)));
    sizes.object["witness"]=Json(int64_t(sizeof(TwoAuxWitness)));
    response.object["sizes"]=sizes;
    const bool summary=request.has("summary_only") && request.at("summary_only").boolean;
    const bool prepareOnly=request.has("prepare_only") && request.at("prepare_only").boolean;
    uint64_t successes=0;
    auto accept=[&](const TwoAuxTask &task,TwoAuxWitness w) {
        const auto checked=validateTwoAuxWitness(p,task,w);
        auto result=Json::dict();
        result.object["raw_slot"]=Json(int64_t(task.rawSlot));
        result.object["status"]=Json(int64_t(w.status));
        result.object["count"]=Json(int64_t(w.count));
        result.object["available"]=Json(int64_t(w.available));
        result.object["rule_checks"]=Json(int64_t(w.ruleChecks));
        result.object["sweeps"]=Json(int64_t(w.sweeps));
        result.object["negative_rule_checks"]=Json(int64_t(checked.negativeRuleChecks));
        result.object["used_helpers"]=Json(int64_t(checked.usedHelpers));
        result.object["live_count"]=Json(int64_t(checked.liveCount));
        auto gates=Json::list(), helpers=Json::list(), creations=Json::list();
        for(int i=0;i<w.count;++i) gates.array.push_back(twoAuxGateJson(w.gates[i]));
        for(int h=0;h<2;++h) {
            auto vector=Json::list();
            for(int c=0;c<p.inputs;++c) vector.array.emplace_back(int64_t(task.helpers[h][c]));
            helpers.array.push_back(vector); creations.array.push_back(twoAuxGateJson(task.creations[h]));
        }
        result.object["gates"]=gates; result.object["helpers"]=helpers; result.object["creations"]=creations;
        if(w.status==1) {
            const auto cost=verifyStage(checked.stage,targets,p.inputs,AdmissionLimits{});
            if(cost!=uint64_t(checked.liveCount)) throw std::runtime_error("test emitted count mismatch");
            result.object["fresh"]=checked.stage.fresh; result.object["outputs"]=checked.stage.outputs;
            if(request.has("transpose_two_aux") && request.at("transpose_two_aux").boolean) {
                auto transposed=transposeStage(checked.stage,p.inputs);
                Matrix expected(p.inputs,std::vector<int64_t>(targets.size()));
                for(size_t r=0;r<targets.size();++r)
                    for(int c=0;c<p.inputs;++c) expected[c][r]=targets[r][c];
                const auto count=verifyStage(transposed,expected,int(targets.size()),AdmissionLimits{});
                result.object["transpose_count"]=Json(int64_t(count));
                result.object["transpose_fresh"]=transposed.fresh;
                result.object["transpose_outputs"]=transposed.outputs;
            }
            if(!successes++) response.object["first_success"]=result;
        }
        if(!summary) results.array.push_back(result);
    };
    if(request.has("witnesses")) {
        if(request.at("witnesses").array.size()>128) throw std::runtime_error("test batch exceeds tile capacity");
#ifdef FGM_CONSTRUCTOR_GPU_TEST
        std::vector<TwoAuxTask> providedTasks;
        std::vector<TwoAuxWitness> providedWitnesses;
#endif
        for(const auto &item:request.at("witnesses").array) {
            TwoAuxTask task{}; TwoAuxFilter filter{};
            if(!prepareTwoAuxTask(p,natural(item.at("raw_slot")),task,filter))
                throw std::runtime_error("test witness selected a filtered route pair");
            TwoAuxWitness w{};
            w.status=integer(item.at("status")); w.available=natural(item.at("available"));
            w.count=integer(item.at("count"));
            if(w.count<0 || w.count>TwoAuxMaxGates || item.at("gates").array.size()!=size_t(w.count))
                throw std::runtime_error("test witness gate count mismatch");
            w.ruleChecks=item.has("rule_checks")?natural(item.at("rule_checks")):uint64_t(w.count);
            w.sweeps=item.has("sweeps")?integer(item.at("sweeps")):0;
            for(int i=0;i<w.count;++i) {
                const auto &g=item.at("gates").array[i].array;
                if(g.size()!=5) throw std::runtime_error("test gate width mismatch");
                w.gates[i]={integer(g[0]),integer(g[1]),integer(g[2]),integer(g[3]),integer(g[4])};
            }
#ifdef FGM_CONSTRUCTOR_GPU_TEST
            if(gpu) { providedTasks.push_back(task); providedWitnesses.push_back(w); }
            else
#endif
            accept(task,w);
        }
#ifdef FGM_CONSTRUCTOR_GPU_TEST
        if(gpu && !providedTasks.empty()) {
            const TwoAuxSettings settings{"signed-two-aux-distinct-v1",budget};
            auto report=twoAuxReport(settings,"test"); uint64_t verification=0;
            validateTwoAuxBatch(p,providedTasks.data(),providedWitnesses.data(),int(providedTasks.size()),
                targets,false,false,AdmissionLimits{},report,verification);
            for(size_t i=0;i<providedTasks.size();++i) accept(providedTasks[i],providedWitnesses[i]);
            response.object["batch_validated"]=report.at("validated");
        }
#endif
    } else if(request.has("raw_slots")) {
        if(request.at("raw_slots").array.size()>128) throw std::runtime_error("test selection exceeds tile capacity");
#ifdef FGM_CONSTRUCTOR_GPU_TEST
        if(gpu && !prepareOnly) {
            ReductionBuffer<TwoAuxProblem> problem(1); *problem.data=p;
            ReductionBuffer<TwoAuxTask> tasks(128);
            ReductionBuffer<TwoAuxWitness> witnesses(128);
            int count=0;
            for(const auto &id:request.at("raw_slots").array) {
                TwoAuxFilter filter{};
                if(prepareTwoAuxTask(p,natural(id),tasks.data[count],filter)) ++count;
            }
            if(count) {
                metalDispatch("constructorTwoAuxClosureKernel",size_t((count+31)/32)*32,32,
                    problem.data,tasks.data,witnesses.data,count);
                const TwoAuxSettings settings{"signed-two-aux-distinct-v1",budget};
                auto report=twoAuxReport(settings,"test"); uint64_t verification=0;
                if(request.has("tamper_lane")) {
                    const int lane=integer(request.at("tamper_lane"));
                    if(lane<0 || lane>=count) throw std::runtime_error("test tamper lane out of range");
                    const std::string kind=request.has("tamper_kind")?request.at("tamper_kind").str():"mask";
                    if(kind=="premature_fixed_point") {
                        auto &w=witnesses.data[lane];
                        w.status=0; w.count=0; w.available=(uint64_t(1)<<p.inputs)-1;
                        w.ruleChecks=0; w.sweeps=0;
                    } else if(kind=="mask") witnesses.data[lane].available^=uint64_t(1)<<33;
                    else throw std::runtime_error("unknown test tamper kind");
                }
                validateTwoAuxBatch(p,tasks.data,witnesses.data,count,targets,false,false,
                    AdmissionLimits{},report,verification);
                for(int i=0;i<count;++i) accept(tasks.data[i],witnesses.data[i]);
                response.object["batch_validated"]=report.at("validated");
            }
        } else
#endif
        for(const auto &id:request.at("raw_slots").array) {
            TwoAuxTask task{}; TwoAuxFilter filter{};
            if(prepareTwoAuxTask(p,natural(id),task,filter) && !prepareOnly) accept(task,testTwoAuxClosure(p,task));
        }
    } else {
#ifdef FGM_CONSTRUCTOR_GPU_TEST
        if(gpu && !prepareOnly) {
            ReductionBuffer<TwoAuxProblem> problem(1); *problem.data=p;
            ReductionBuffer<TwoAuxTask> tasks(128);
            ReductionBuffer<TwoAuxWitness> witnesses(128);
            const TwoAuxSettings settings{"signed-two-aux-distinct-v1",budget};
            auto report=twoAuxReport(settings,"test"); uint64_t verification=0;
            uint64_t batchValidated=0;
            TwoAuxTask task{};
            for(;;) {
                int count=0;
                while(count<128 && stream.next(task)) tasks.data[count++]=task;
                if(!count) break;
                metalDispatch("constructorTwoAuxClosureKernel",size_t((count+31)/32)*32,32,
                    problem.data,tasks.data,witnesses.data,count);
                validateTwoAuxBatch(p,tasks.data,witnesses.data,count,targets,false,false,
                    AdmissionLimits{},report,verification);
                batchValidated+=uint64_t(count);
                for(int i=0;i<count;++i) accept(tasks.data[i],witnesses.data[i]);
            }
            response.object["batch_validated"]=Json(int64_t(batchValidated));
        } else
#endif
        {
            TwoAuxTask task{};
            while(stream.next(task)) if(!prepareOnly) accept(task,testTwoAuxClosure(p,task));
        }
    }
    const auto stats=stream.stats(); auto counts=Json::dict();
    counts.object["raw_scanned"]=Json(int64_t(stats.rawScanned));
    counts.object["first_invalid"]=Json(int64_t(stats.firstInvalid));
    counts.object["second_invalid"]=Json(int64_t(stats.secondInvalid));
    counts.object["symmetry_filtered"]=Json(int64_t(stats.symmetryFiltered));
    counts.object["prepared"]=Json(int64_t(stats.prepared));
    response.object["stats"]=counts;
    response.object["successes"]=Json(int64_t(successes));
    response.object["results"]=results;
    return response;
}
