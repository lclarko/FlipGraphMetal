// Host-only production controller adapter for independent reference comparisons.
#include "run_config.h"
#include <type_traits>
#include <memory>
#include "core.h"
#include "addition.h"
#include "flip_set.h"
#include "scheme_integer.h"
#include "scheme_z2.h"
#include "controlled.h"
#ifdef FGM_CONTROLLED_GPU
#include "controlled_capture.h"
#include "runtime.h"
#endif
using fgm::Json;
static const char *outcome(ControlledOutcome x) {
    const char *names[]={"active","applied","unsuccessful","tuple_rejection","coefficient_rejection",
        "proposal_exhausted","rank_blocked","capacity_error","restart_requested","target_pending",
        "existing_target_pending","target_met","budget_exhausted","invalid_state"};
    return names[unsigned(x)];
}
static Json number(uint64_t x) { if(x>INT64_MAX) return Json(std::to_string(x)); return Json(int64_t(x)); }
template<class S> Json scheme(const S &s) {
    Json j=Json::dict(),body=Json::dict(),n=Json::list(),pairs=Json::list();
    for(int p=0;p<3;++p) n.array.push_back(number(s.n[p]));
    body.object["n"]=n; body.object["m"]=number(s.m); body.object["z2"]=Json(std::is_same_v<S,SchemeZ2>);
    for(int p=0;p<3;++p) {
        Json rows=Json::list(),flips=Json::dict(),list=Json::list();
        for(int r=0;r<s.m;++r) {
            Json row=Json::list();
            for(int c=0;c<s.nn[p];++c) {
                int value;
                if constexpr(std::is_same_v<S,SchemeInteger>) value=s.uvw[p][r][c];
                else value=int((s.uvw[p][r]>>c)&1);
                row.array.emplace_back(int64_t(value));
            }
            rows.array.push_back(row);
        }
        body.object[std::string(1,"uvw"[p])]=rows;
        for(uint32_t i=0;i<s.flips[p].size;++i) {
            Json pair=Json::list(); pair.array.push_back(number(s.flips[p].index1(i))); pair.array.push_back(number(s.flips[p].index2(i))); list.array.push_back(pair);
        }
        flips.object["overflow"]=number(s.flips[p].overflow); flips.object["pairs"]=list; pairs.array.push_back(flips);
    }
    j.object["scheme"]=body; j.object["candidates"]=pairs; return j;
}
struct Sink {
    Json mandatory, optional=Json::list();
    template<class S> void capture(const S &s,ControlledOperation op,uint64_t control,bool required,bool extra,uint64_t) {
        Json sample=Json::dict(); sample.object["state"]=scheme(s); sample.object["rank"]=number(s.m);
        sample.object["operation"]=Json(op==ControlledOperation::Flip?"flip":op==ControlledOperation::Reduction?"reduction":"expansion");
        sample.object["control"]=number(control);
        if(required) mandatory=sample;
        if(extra) optional.array.push_back(sample);
    }
};
#ifdef FGM_CONTROLLED_GPU
template<class V> struct GPUBuffer {
    V *data=nullptr;
    explicit GPUBuffer(size_t count) { metalAllocate(&data,count*sizeof(V)); }
    ~GPUBuffer() { if(data) metalFree(data); }
    GPUBuffer(const GPUBuffer&)=delete;
    GPUBuffer& operator=(const GPUBuffer&)=delete;
};
template<class S> Json snapshot(const ControlledState &,const S &,const S &,const Sink &);
struct PackedBuffers {
    GPUBuffer<uint32_t> terms,pairs,scratch,bestTerms,bestPairs;
    explicit PackedBuffers(size_t padded):terms(350*3*padded),pairs(500*3*padded),scratch(500*3*padded),bestTerms(350*3*padded),bestPairs(500*3*padded) {}
};
template<class S> struct GPUWorker {
    GPUBuffer<S> current,best,captures;
    GPUBuffer<ControlledState> state;
    GPUBuffer<ControlledCaptureMeta> metadata;
    std::unique_ptr<PackedBuffers> packed;
    size_t count,slots;
    std::vector<Json> laneWords;
    explicit GPUWorker(size_t slotCount,bool usePacked,size_t workerCount,const ControlledSettings &c,const S &parent,uint32_t seed,uint32_t firstWorker):
        current(workerCount),best(workerCount),captures(slotCount*workerCount),state(workerCount),metadata(slotCount*workerCount),count(workerCount),slots(slotCount),laneWords(workerCount) {
        if(usePacked) packed=std::make_unique<PackedBuffers>((count+31)/32*32);
        for(size_t i=0;i<count;++i)
            if(!controlledInitialize(c,state.data[i],parent,current.data[i],best.data[i],seed,firstWorker+uint32_t(i))) throw std::runtime_error("lane initialization failed");
    }
    Sink restored(size_t lane) const {
        const auto &host=state.data[lane];
        if(host.optionalCount>=slots) throw std::runtime_error("GPU capture count exceeds quota");
        Sink result; auto base=lane*slots;
        if(host.mandatoryValid) result.capture(captures.data[base],ControlledOperation(metadata.data[base].operation),metadata.data[base].control,true,false,0);
        for(uint64_t i=0;i<host.optionalCount;++i)
            result.capture(captures.data[base+i+1],ControlledOperation(metadata.data[base+i+1].operation),metadata.data[base+i+1].control,false,true,i);
        return result;
    }
    Json snapshots() const {
        Json list=Json::list();
        for(size_t i=0;i<count;++i) {
            auto record=snapshot(state.data[i],current.data[i],best.data[i],restored(i));
            if(laneWords[i].kind==Json::Array) record.object["words"]=laneWords[i];
            list.array.push_back(record);
        }
        return list;
    }
    void dispatch(const ControlledSettings &c,ControlledState &host,S &scheme,S &bestScheme,Sink &sink,uint32_t command) {
        *current.data=scheme; *best.data=bestScheme; *state.data=host;
        std::vector<uint32_t> before(count);
        for(size_t i=0;i<count;++i) before[i]=state.data[i].rng.value;
        const uint64_t workers=count,steps=1;
        if(packed) {
            if(command) metalDispatch("controlledPackedTestKernel",count,32,current.data,best.data,state.data,captures.data,metadata.data,c,workers,steps,
                packed->terms.data,packed->pairs.data,packed->scratch.data,packed->bestTerms.data,packed->bestPairs.data,command);
            else metalDispatch(c.alternatives?"controlledPackedAlternativesKernel":"controlledPackedReductionKernel",count,32,current.data,best.data,state.data,captures.data,metadata.data,c,workers,steps,
                packed->terms.data,packed->pairs.data,packed->scratch.data,packed->bestTerms.data,packed->bestPairs.data);
        }
        else if(command) metalDispatch("controlledTestKernel",count,1,current.data,best.data,state.data,captures.data,metadata.data,c,workers,steps,command);
        else metalDispatch("controlledGeneralKernel",count,1,current.data,best.data,state.data,captures.data,metadata.data,c,workers,steps);
        scheme=*current.data; bestScheme=*best.data; host=*state.data;
        sink=restored(0);
        for(size_t i=0;i<count;++i) {
            Json words=Json::list(); RandomState replay{before[i]};
            if(replay.value!=state.data[i].rng.value) do {
                words.array.push_back(number(randomWord(&replay)));
                if(words.array.size()>1000000) throw std::runtime_error("lane RNG trace limit");
            } while(replay.value!=state.data[i].rng.value);
            laneWords[i]=std::move(words);
        }
    }
};
#endif
template<class S> Json snapshot(const ControlledState &s,const S &current,const S &best,const Sink &sink) {
    Json j=Json::dict(); j.object["state"]=scheme(current); j.object["best_state"]=scheme(best);
    j.object["rng"]=s.initialized?number(s.rng.value):Json();
    j.object["countdown"]=s.initialized?number(s.countdown):Json();
    j.object["terminal"]=s.terminal==ControlledOutcome::Active?Json():Json(outcome(s.terminal));
    j.object["outcome"]=Json(outcome(s.lastOutcome)); j.object["pending_restart"]=Json(bool(s.pendingRestart));
    j.object["mandatory_committed"]=Json(bool(s.mandatoryCommitted));
    j.object["mandatory"]=sink.mandatory; j.object["optional"]=sink.optional;
    j.object["stagnation"]=number(s.stagnation); j.object["flips"]=number(s.flips); j.object["controls"]=number(s.controls);
    j.object["optional_encounters"]=number(s.optionalEncounters); j.object["optional_drops"]=number(s.optionalDrops);
    j.object["proposals"]=number(s.proposals); j.object["tuple_rejections"]=number(s.tupleRejections);
    j.object["coefficient_rejections"]=number(s.coefficientRejections); j.object["proposal_exhausted"]=number(s.exhaustedProposals); j.object["rank_blocked"]=number(s.blocked);
    Json attempted=Json::dict(),applied=Json::dict(),removed=Json::dict();
    const char *ops[]={"flip","reduction","expansion"};
    for(int i=0;i<3;++i) { attempted.object[ops[i]]=number(s.attempted[i]); applied.object[ops[i]]=number(s.applied[i]); }
    removed.object["flip"]=number(s.removed[0]); removed.object["reduction"]=number(s.removed[1]); removed.object["total"]=number(s.removed[2]);
    j.object["attempted"]=attempted; j.object["applied"]=applied; j.object["removed_terms"]=removed;
    return j;
}
template<class S> int run(const Json &request,bool gpu,bool packed) {
    const auto config=fgm::parseControlledConfig(request.at("config"),std::is_same_v<S,SchemeInteger>?"ZT":"F2");
    ControlledSettings c{};
    c.anchor=config.anchor; c.ceiling=config.ceiling(); c.intervalMin=config.intervalMin; c.intervalMax=config.intervalMax;
    c.flipBudget=config.flipBudget; c.controlBudget=config.controlBudget; c.stagnationLimit=config.stagnationLimit;
    c.optionalQuota=config.optionalQuota; c.proposalLimit=config.proposalLimit; c.reductionQ=config.reductionQ;
    c.alternatives=config.mode==fgm::ControlledConfig::Mode::Alternatives;
    c.targetEnabled=config.targetRank.has_value(); c.targetRank=config.targetRank.value_or(0);
    if(packed && (!gpu || std::is_same_v<S,SchemeZ2> || config.dimensions!=std::array<uint64_t,3>{3,3,3})) throw std::runtime_error("packed requires signed 3x3 GPU execution");
    if(c.optionalQuota>1024) throw std::runtime_error("test capture quota exceeds 1024");
    auto parent=std::make_unique<S>();
    std::istringstream input(request.at("parent_text").str());
    if(!parent->read(input,true)) throw std::runtime_error("invalid parent");
    for(int p=0;p<3;++p) if(uint64_t(parent->n[p])!=config.dimensions[p]) throw std::runtime_error("dimensions mismatch");
    auto current=std::make_unique<S>(),best=std::make_unique<S>(); ControlledState state{}; Sink sink;
    uint64_t worker=request.has("worker")?uint64_t(request.at("worker").num()):0;
    if(worker>UINT32_MAX) throw std::runtime_error("worker out of range");
    if(!controlledInitialize(c,state,*parent,*current,*best,config.seed,uint32_t(worker))) throw std::runtime_error("initialization failed");
    uint64_t workers=request.has("workers")?uint64_t(request.at("workers").num()):1;
    if(!workers || workers>64 || worker+workers-1>UINT32_MAX || (workers>1 && (!gpu || request.has("rng")))) throw std::runtime_error("invalid test worker count");
    if(request.has("rng")) { auto value=request.at("rng").num(); if(value<=0 || uint64_t(value)>UINT32_MAX || !state.initialized) throw std::runtime_error("invalid explicit RNG"); state.rng.value=uint32_t(value); }
#ifdef FGM_CONTROLLED_GPU
    std::unique_ptr<GPUWorker<S>> device;
    if(gpu) device=std::make_unique<GPUWorker<S>>(size_t(c.optionalQuota+1),packed,size_t(workers),c,*parent,config.seed,uint32_t(worker));
#else
    if(gpu) throw std::runtime_error("GPU support was not compiled");
#endif
    Json snapshots=Json::list(); auto initial=snapshot(state,*current,*best,sink);
#ifdef FGM_CONTROLLED_GPU
    if(workers>1) initial.object["lanes"]=device->snapshots();
#endif
    snapshots.array.push_back(initial);
    const auto &commands=request.at("commands"); if(commands.kind!=Json::Array || commands.array.size()>10000) throw std::runtime_error("bounded commands required");
    for(const auto &command:commands.array) {
        auto name=command.str(); uint32_t before=state.rng.value;
        if(workers>1 && name!="step" && name!="plus" && name!="random" && name!="existing") throw std::runtime_error("multi-lane test supports arithmetic commands only");
        if(name=="step" || name=="plus" || name=="random" || name=="existing") {
            const uint32_t command=name=="step"?0:name=="plus"?1:name=="random"?2:3;
#ifdef FGM_CONTROLLED_GPU
            if(device) device->dispatch(c,state,*current,*best,sink,command);
            else
#endif
            if(!command) controlledStep(c,state,*current,*best,sink);
            else controlledPrimitive(c,state,*current,command-1);
        }
        else if(name=="commit") { if(!controlledCommit(state,true,true)) throw std::runtime_error("commit rejected"); }
        else if(name=="boundary") { if(!controlledBoundary(state)) throw std::runtime_error("boundary rejected"); sink=Sink{}; }
        else if(name=="restart") { if(!controlledInstallRestart(c,state,*parent,*current,*best,true)) throw std::runtime_error("restart rejected"); }
        else if(name=="stage") { if(!controlledChargeStage(c,state)) throw std::runtime_error("stage credit rejected"); }
        else throw std::runtime_error("unknown command");
        auto record=snapshot(state,*current,*best,sink);
#ifdef FGM_CONTROLLED_GPU
        if(workers>1) record.object["lanes"]=device->snapshots();
#endif
        // Diagnostic-only replay recovers exact words without changing production RNG.
        Json words=Json::list(); RandomState replay{before};
        if(before!=state.rng.value) do { words.array.push_back(number(randomWord(&replay))); if(words.array.size()>1000000) throw std::runtime_error("test draw limit"); } while(replay.value!=state.rng.value);
        record.object["words"]=words; snapshots.array.push_back(record);
    }
    Json result=Json::dict(); result.object["snapshots"]=snapshots;
    result.object["state_bytes"]=number(sizeof(ControlledState)); result.object["settings_bytes"]=number(sizeof(ControlledSettings)); result.object["scheme_bytes"]=number(sizeof(S));
    result.object["persistent_buffer_bytes"]=number(workers*((c.optionalQuota+3)*sizeof(S)+sizeof(ControlledState)+(c.optionalQuota+1)*16));
    result.object["dispatch_error_buffer_bytes"]=number(workers*sizeof(int));
    if(packed) result.object["packed_buffer_bytes"]=number(26400*((workers+31)/32*32));
    std::cout<<fgm::dump(result)<<'\n'; return 0;
}
int main(int argc,char **argv) {
    try {
        if(argc<2) throw std::runtime_error("expected ZT or F2 [--gpu] [--request FILE]");
        bool gpu=false,packed=false; std::string path;
        for(int i=2;i<argc;++i) {
            std::string arg=argv[i];
            if(arg=="--gpu" && !gpu) gpu=true;
            else if(arg=="--packed" && !packed) packed=true;
            else if(arg=="--request" && path.empty() && i+1<argc) path=argv[++i];
            else throw std::runtime_error("invalid driver option");
        }
        std::ifstream file; std::istream *input=&std::cin;
        if(!path.empty()) { file.open(path); if(!file) throw std::runtime_error("cannot open request"); input=&file; }
        std::string bytes; char c; while(input->get(c)) { if(bytes.size()==1048576) throw std::runtime_error("test input too large"); bytes+=c; }
        auto request=fgm::Parser(bytes).parse();
        auto execute=[&](const Json &item) {
#if !defined(FGM_CONTROLLED_GPU) || !defined(METAL_F2)
            if(std::string(argv[1])=="ZT") return run<SchemeInteger>(item,gpu,packed);
#endif
#if !defined(FGM_CONTROLLED_GPU) || defined(METAL_F2)
            if(std::string(argv[1])=="F2") return run<SchemeZ2>(item,gpu,packed);
#endif
            throw std::runtime_error("unknown domain");
        };
        if(request.has("cases")) {
            const auto &cases=request.at("cases");
            if(cases.kind!=Json::Array || cases.array.empty() || cases.array.size()>32) throw std::runtime_error("expected 1..32 cases");
            for(const auto &item:cases.array) execute(item);
            return 0;
        }
        return execute(request);
    } catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
