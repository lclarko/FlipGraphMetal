#include "../../src/metal/host.h"
#include "execution_layout.h"
#include "search_execution.h"
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
Json searchTestEvaluation(const SchemeRecord &scheme,const ReductionSettings &,Json &) {
    // A naive circuit exercises host acceptance and journal failure paths. This
    // test-only fixture builder is neither the constructor nor GPU evidence.
    if(failure()=="evaluation")throw std::runtime_error("injected interrupted evaluation");
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
        return fgm::runConfigured(argc, argv, "", "", nativeExecutionLayout(),fgm::executeHostSearchTest);
    } catch (const fgm::Resource &error) {
        std::cerr << "resource_limit: " << error.what() << '\n';
        return 2;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
