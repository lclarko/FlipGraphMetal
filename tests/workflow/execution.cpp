#include "../../src/metal/host.h"
#include "execution_layout.h"
#include "search_execution.h"
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <fstream>

// A CPU dispatch stand-in exercises the production host transaction path. It
// does not establish Metal correctness or serve as an independent controller.
namespace {
unsigned dispatches=0;
std::string failure() { const auto *value=std::getenv("FGM_TEST_SEARCH_FAILURE");return value?value:""; }
void saveExpected(const fgm::PreparedRun &run) {
    std::ifstream head(run.config.history.path/"committed-head.json");
    std::string bytes((std::istreambuf_iterator<char>(head)),{});
    auto expected=run.receipt;expected.object["test_committed_head"]=fgm::Parser(bytes).parse();
    std::ofstream out(run.config.output.string()+".expected.json");
    out<<fgm::dump(expected);if(!out)throw std::runtime_error("test snapshot write failed");
}
struct Sink {
    SchemeInteger *captures;ControlledCaptureMeta *metadata;
    void capture(const SchemeInteger &scheme,ControlledOperation operation,uint64_t control,bool mandatory,bool optional,uint64_t index) {
        if(mandatory){captures[0]=scheme;metadata[0]={control,uint32_t(operation),0};}
        if(optional){captures[index+1]=scheme;metadata[index+1]={control,uint32_t(operation),0};}
    }
};
}
void *metalAllocateBytes(size_t size) {void *p=std::malloc(size);if(!p)throw std::bad_alloc();return p;}
void metalFree(void *p) {std::free(p);}
void metalLaunch(const char *name,size_t threads,size_t,std::initializer_list<MetalArgument> args) {
    if(std::strcmp(name,"controlledGeneralKernel")||args.size()!=8)throw std::runtime_error("unsupported CPU test dispatch");
    ++dispatches;
    auto a=args.begin();auto *current=(SchemeInteger *)a[0].pointer,*best=(SchemeInteger *)a[1].pointer;
    auto *states=(ControlledState *)a[2].pointer;auto *captures=(SchemeInteger *)a[3].pointer;
    if(failure()=="dispatch"&&dispatches==2){states[0].flips=999;current[0].m=999;
        throw std::runtime_error("injected partial dispatch failure");}
    auto *metadata=(ControlledCaptureMeta *)a[4].pointer;const auto &settings=*(const ControlledSettings *)a[5].pointer;
    const auto count=*(const uint64_t *)a[6].pointer,steps=*(const uint64_t *)a[7].pointer;
    if(count!=threads)throw std::runtime_error("CPU test dispatch count mismatch");
    for(uint64_t w=0;w<count;++w){const auto offset=w*(settings.optionalQuota+1);Sink sink{captures+offset,metadata+offset};
        for(uint64_t step=0;step<steps;++step){if(controlledTerminal(states[w])||states[w].pendingRestart)break;
            controlledStep(settings,states[w],current[w],best[w],sink);}}
    if(failure()=="append"&&dispatches==2)fgm::Journal::testFaults(0);
    if(failure()=="append-sync"&&dispatches==2)fgm::Journal::testFaults(-1,0);
}
namespace fgm {
void searchTestCheckpoint(const char *point,const PreparedRun &run) {
    const std::string at=point,mode=failure();
    if((at=="before_reserve"&&dispatches==1)||(at=="before_start"&&mode=="start")||(at=="before_run_end"&&mode=="end"))saveExpected(run);
    if(at=="before_reserve"&&dispatches==1&&mode=="reserve")throw Resource("injected next reservation failure");
    if(at=="before_start"&&mode=="start")throw Resource("injected resumed start failure");
    if(at=="before_run_end"&&mode=="end")Journal::testFaults(0);
}
Json searchTestEvaluation(const SchemeRecord &scheme,const ReductionSettings &settings,Json &receipt) {
    // A naive circuit exercises host acceptance and journal failure paths. This
    // test-only fixture builder is neither the constructor nor GPU evidence.
    if(failure()=="evaluation")throw std::runtime_error("injected interrupted evaluation");
    const char *dynamic=std::getenv("FGM_TEST_DYNAMIC_EVALUATION");
    if(settings.constructor && (!dynamic||std::strcmp(dynamic,"1"))) {
        const char *path=std::getenv("FGM_TEST_EVALUATION_RESULT");
        if(!path)throw std::runtime_error("constructor host test requires an explicit evaluation fixture");
        if(!std::filesystem::is_regular_file(path)||std::filesystem::file_size(path)>65536)
            throw std::runtime_error("invalid constructor host evaluation fixture");
        std::ifstream input(path,std::ios::binary);
        if(!input)throw std::runtime_error("cannot read constructor host evaluation fixture");
        const std::string bytes((std::istreambuf_iterator<char>(input)),{});
        if(input.bad())throw std::runtime_error("constructor host evaluation fixture read failed");
        auto result=Parser(bytes).parse();
        if(result.has("two_auxiliary"))receipt.object["two_auxiliary"]=result.at("two_auxiliary");
        return result;
    }
    Json circuit=Json::dict(),dimensions=Json::list(),costs=Json::dict();uint64_t total=0;
    for(auto n:scheme.n)dimensions.array.push_back(Json(int64_t(n)));
    circuit.object["n"]=dimensions;circuit.object["m"]=Json(int64_t(scheme.rank));circuit.object["z2"]=Json(false);
    for(size_t p=0;p<3;++p) {
        const std::string key(1,"uvw"[p]);Json outputs=Json::list();uint64_t cost=0;
        const auto count=p==2?scheme.f[p][0].size():scheme.rank;
        for(size_t r=0;r<count;++r) {
            Json expression=Json::list();
            const auto width=p==2?scheme.rank:scheme.f[p][r].size();
            for(size_t i=0;i<width;++i) {
                const auto coefficient=p==2?scheme.f[p][i][r]:scheme.f[p][r][i];
                if(coefficient){Json term=Json::dict();term.object["index"]=Json(int64_t(i));term.object["value"]=Json(coefficient);expression.array.push_back(term);}
            }
            if(!expression.array.empty())cost+=expression.array.size()-1;
            outputs.array.push_back(expression);
        }
        circuit.object[key]=outputs;circuit.object[key+"_fresh"]=Json::list();costs.object[key]=Json(int64_t(cost));total+=cost;
    }
    Json complexity=Json::dict();complexity.object["naive"]=Json(int64_t(total));complexity.object["reduced"]=Json(int64_t(total));
    circuit.object["complexity"]=complexity;
    Json result=Json::dict();result.object["circuit"]=circuit;result.object["verified_circuit_additions"]=Json(int64_t(total));
    result.object["verified_circuit_additions_by_stage"]=costs;
    result.object["baseline_additions"]=Json(int64_t(total));result.object["baseline_additions_by_stage"]=costs;
    for(const auto *key:{"phase_microseconds","construction","stage_sources"})result.object[key]=Json::dict();
    for(const auto *key:{"rounds_completed","flip_attempts","flips_applied"})result.object[key]=Json(int64_t(0));
    if(failure()=="evaluation-count")result.object["verified_circuit_additions"]=Json(int64_t(total-1));
    if(settings.constructor) {
        // A host-only protocol fixture for batch tests whose captured factors
        // are created during the run. Budget one filters before any dispatch.
        if(settings.constructor->maxPairSlots!=1)
            throw std::runtime_error("dynamic constructor host fixture requires one raw slot");
        auto report=Json::dict(),stages=Json::dict();
        AdmissionContext identity;
        report.object["effective_factors_id"]=Json(identity.identity(scheme,false));
        result.object["pre_two_aux_additions_by_stage"]=costs;
        for(int stage=0;stage<3;++stage) {
            const std::string name=stage==2?"wt":std::string(1,"uv"[stage]);
            const std::string output(1,"uvw"[stage]);
            const auto prepared=fgm_constructor::prepareTwoAux(scheme.f[stage],9);
            fgm_constructor::TwoAuxStream stream(prepared,1);
            fgm_constructor::TwoAuxTask task;
            if(stream.next(task)||stream.stats().prepared)
                throw std::runtime_error("dynamic constructor host fixture unexpectedly dispatched");
            const auto stats=stream.stats();
            const uint64_t rawTotal=stream.rawTotal();
            int q=0;for(int i=0;i<prepared.outputs;++i)q+=prepared.outputDirection[i]>=0;
            const int floor=prepared.directions-9+2+(stage==2?q-9:0);
            const bool skip=uint64_t(costs.at(output).num())<=uint64_t(floor);
            auto s=Json::dict();
            s.object["schema"]=Json("fgm-two-aux-report-v1");
            s.object["family"]=Json(settings.constructor->family);
            s.object["implementation"]=Json("metal-two-aux-v1");
            s.object["enumeration_order"]=Json("lexicographic-pairs-plus-first-v1");
            s.object["stage"]=Json(name);
            s.object["requested_auxiliaries"]=runNumber(2);
            s.object["max_pair_slots"]=runNumber(1);
            s.object["target_directions"]=runNumber(prepared.directions-9);
            s.object["nonzero_output_occurrences"]=runNumber(q);
            s.object["improvement_floor"]=runNumber(floor);
            s.object["raw_family_size"]=runNumber(rawTotal);
            s.object["prefix_limit"]=runNumber(1);
            s.object["baseline_cost"]=costs.at(output);
            s.object["incumbent_cost"]=costs.at(output);
            s.object["final_cost"]=costs.at(output);
            s.object["stop_reason"]=Json(skip?"proved-no-improvement":"budget-exhausted");
            s.object["coverage"]=Json(skip?"not-searched":"partial");
            s.object["selected_witness"]=Json();
            for(const auto *key:{"prepared","dispatched","completed","validated","gpu_rule_checks","gpu_sweeps",
                                  "negative_validation_rule_checks","batch_tail_candidates","used_auxiliaries"})
                s.object[key]=runNumber(0);
            s.object["raw_scanned"]=runNumber(skip?0:stats.rawScanned);
            s.object["first_invalid"]=runNumber(skip?0:stats.firstInvalid);
            s.object["second_invalid"]=runNumber(skip?0:stats.secondInvalid);
            s.object["symmetry_filtered"]=runNumber(skip?0:stats.symmetryFiltered);
            s.object["logical_prefix"]=runNumber(skip?0:stats.rawScanned);
            s.object["unvisited_raw_slots"]=runNumber(rawTotal-(skip?0:stats.rawScanned));
            auto times=Json::dict();
            for(const auto *key:{"allocation","preparation","dispatch","gpu","witness_validation","transposition","verification"})
                times.object[key]=runNumber(0);
            s.object["microseconds"]=std::move(times);
            stages.object[name]=std::move(s);
            auto old=Json::dict();old.object["status"]=Json("family-exhausted");
            result.object["construction"].object[name]=std::move(old);
            result.object["stage_sources"].object[output]=Json("baseline");
        }
        report.object["stages"]=std::move(stages);
        result.object["two_auxiliary"]=report;
        receipt.object["two_auxiliary"]=std::move(report);
    }
    return result;
}
void executeHostSearchTest(PreparedRun &run) {
    if(!run.config.policy||run.config.policy->domain!=ControlledConfig::Domain::Signed)
        throw std::runtime_error("CPU integration driver only executes signed search");
    if(failure()=="recover-sync"){
        saveExpected(run);
        Journal::testIoReset(run.config.history.path/"journal.bin");
    }
    executeSearch<SchemeInteger>(run);
}
}

int main(int argc, char **argv) {
    try {
        auto layout=nativeExecutionLayout();
        if(const char *value=std::getenv("FGM_TEST_TWO_AUX_SHARED_BYTES")) {
            uint64_t bytes=0;
            const auto end=value+std::strlen(value);
            const auto parsed=std::from_chars(value,end,bytes);
            if(parsed.ec!=std::errc{}||parsed.ptr!=end)
                throw std::runtime_error("invalid test two-auxiliary shared bytes");
            layout.twoAuxSharedBytes=bytes;
        }
        return fgm::runConfigured(argc, argv, "", "", layout,fgm::executeHostSearchTest);
    } catch (const fgm::Resource &error) {
        std::cerr << "resource_limit: " << error.what() << '\n';
        return 2;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
