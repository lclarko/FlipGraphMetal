#pragma once
// Included once by the production executable, after its arithmetic declarations.
#include "execution.h"
#include "journal.h"
#include "../metal/controlled.h"
#include "../metal/controlled_capture.h"
#include <memory>
#include <random>
#include <sstream>

namespace fgm {
inline Json runNumber(uint64_t value) {
    if(value>INT64_MAX)throw Resource("run record integer overflow");return Json(int64_t(value));
}
template<class Value> struct RunBuffer {
    Value *data=nullptr;
    explicit RunBuffer(uint64_t count) {metalAllocate(&data,size_t(configCheckedMultiply(count,sizeof(Value))));}
    ~RunBuffer(){if(data)metalFree(data);}
    RunBuffer(const RunBuffer&)=delete;RunBuffer&operator=(const RunBuffer&)=delete;
};
struct PackedRunBuffers {
    RunBuffer<uint32_t> terms,pairs,scratch,bestTerms,bestPairs;
    explicit PackedRunBuffers(uint64_t workers):terms(workers*1050),pairs(workers*1500),scratch(workers*1500),bestTerms(workers*1050),bestPairs(workers*1500){}
};
template<class S> void loadRunScheme(const SchemeRecord &source,S &target) {
    std::ostringstream text;text<<source.n[0]<<' '<<source.n[1]<<' '<<source.n[2]<<' '<<source.rank<<'\n';
    for(const auto &matrix:source.f)for(const auto &row:matrix){for(auto value:row)text<<value<<' ';text<<'\n';}
    std::istringstream input(text.str());
    if(!target.read(input,true)||candidateOverflow(target))throw std::runtime_error("verified parent cannot enter native representation");
}
template<class S> SchemeRecord runScheme(const S &source) {
    if(source.m<1||source.m>350||candidateOverflow(source))throw std::runtime_error("invalid observed representation");
    SchemeRecord result;result.rank=uint32_t(source.m);result.f2=std::is_same_v<S,SchemeZ2>;
    for(int p=0;p<3;++p){
        if(source.n[p]<1||source.n[p]>16||source.nn[p]!=source.n[p]*source.n[(p+1)%3]||source.nn[p]>64)
            throw std::runtime_error("invalid observed dimensions");
        result.n[p]=uint32_t(source.n[p]);result.f[p].resize(result.rank);
        for(int r=0;r<source.m;++r){
            if constexpr(std::is_same_v<S,SchemeInteger>) {
                if(!source.uvw[p][r].valid||source.uvw[p][r].n!=source.nn[p]||
                    (source.nn[p]<64&&(source.uvw[p][r].values>>source.nn[p])))throw std::runtime_error("invalid observed coefficient representation");
            } else if(source.nn[p]<64&&(source.uvw[p][r]>>source.nn[p]))throw std::runtime_error("invalid F2 coefficient width");
            for(int c=0;c<source.nn[p];++c){
            int value;
            if constexpr(std::is_same_v<S,SchemeInteger>)value=source.uvw[p][r][c];
            else value=int((source.uvw[p][r]>>c)&1);
            result.f[p][r].push_back(value);
            }
        }
        if(source.flips[p].size>500)throw std::runtime_error("invalid observed candidate count");
        std::set<std::pair<uint32_t,uint32_t>> expected,observed;
        for(uint32_t i=0;i<result.rank;++i)for(uint32_t j=i+1;j<result.rank;++j)
            if(result.f[p][i]==result.f[p][j])expected.emplace(i,j);
        for(uint32_t i=0;i<source.flips[p].size;++i){auto a=source.flips[p].index1(i),b=source.flips[p].index2(i);
            if(a>=result.rank||b>=result.rank||a==b||!observed.emplace(std::min(a,b),std::max(a,b)).second)
                throw std::runtime_error("invalid observed candidate pair");}
        if(expected!=observed)throw std::runtime_error("observed candidate neighborhood is incomplete");
    }
    return result;
}
inline PoolMember verifiedMember(const SchemeRecord &source,const AdmissionLimits &limits) {
    AdmissionContext context(limits);
    if(!context.verify(source))throw std::runtime_error("native observation failed exact tensor verification");
    auto analysis=context.analyze(source);
    if(!analysis.at("search_eligible").boolean)throw std::runtime_error("observed tensor exceeds search representation");
    return {context.identity(source,true),source,uint64_t(analysis.at("potential_pairs").num())};
}
inline ControlledSettings walkSettings(const ControlledConfig &p) {
    ControlledSettings c{};c.anchor=p.anchor;c.ceiling=p.ceiling();c.intervalMin=p.intervalMin;c.intervalMax=p.intervalMax;
    c.flipBudget=p.flipBudget;c.controlBudget=p.controlBudget;c.stagnationLimit=p.stagnationLimit;
    c.optionalQuota=p.optionalQuota;c.proposalLimit=p.proposalLimit;c.targetRank=p.targetRank.value_or(0);
    c.reductionQ=p.reductionQ;c.alternatives=p.mode==ControlledConfig::Mode::Alternatives;c.targetEnabled=p.targetRank.has_value();return c;
}
inline const char *walkOutcome(ControlledOutcome value) {
    const char *names[]={"active","applied","unsuccessful_flip","tuple_rejection","coefficient_rejection","proposal_exhausted",
        "rank_blocked","capacity_error","restart_requested","target_pending","existing_target_pending","target_met","budget_exhausted","invalid_state"};
    if(uint32_t(value)>=sizeof(names)/sizeof(*names))throw std::runtime_error("invalid worker outcome");return names[uint32_t(value)];
}
inline Json workerRecord(const ControlledState &s,uint64_t id,int current,int best) {
    Json result=Json::dict();result.object["worker"]=runNumber(id);result.object["rng"]=runNumber(s.rng.value);
    result.object["initialized"]=Json(bool(s.initialized));result.object["countdown"]=runNumber(s.countdown);
    result.object["stagnation"]=runNumber(s.stagnation);result.object["flips"]=runNumber(s.flips);result.object["controls"]=runNumber(s.controls);
    result.object["terminal"]=Json(walkOutcome(s.terminal));result.object["outcome"]=Json(walkOutcome(s.lastOutcome));
    result.object["pending_restart"]=Json(bool(s.pendingRestart));result.object["current_rank"]=runNumber(current);result.object["best_rank"]=runNumber(best);
    result.object["mandatory_committed"]=Json(bool(s.mandatoryCommitted));
    Json attempts=Json::list(),applied=Json::list();for(int p=0;p<3;++p){attempts.array.push_back(runNumber(s.attempted[p]));applied.array.push_back(runNumber(s.applied[p]));}
    result.object["attempted"]=attempts;result.object["applied"]=applied;result.object["terms_removed"]=runNumber(s.removed[2]);return result;
}
template<class S> void executeSearch(PreparedRun &run) {
    auto &config=run.config;auto &policy=*config.policy;auto settings=walkSettings(policy);
    const auto workers=config.execution.workers,slots=configCheckedAdd(policy.optionalQuota,1);
    const bool packed=config.execution.backend!="general"&&run.receipt.at("packed_eligible").boolean;
    // Pools are copied transactionally; serialization and recovery buffers have
    // explicit reserves in addition to shared Metal storage and admitted inputs.
    const auto reserved=configCheckedAdd(configCheckedMultiply(config.pool.memoryBytes,2),
        configCheckedAdd(configCheckedMultiply(config.history.transactionBytes,4),config.history.indexMemoryBytes));
    const auto planned=uint64_t(run.receipt.at("planned_buffer_bytes").num());
    if(configCheckedAdd(configCheckedAdd(planned,uint64_t(run.receipt.at("admission_content_bytes").num())),reserved)>config.execution.memoryBytes)
        throw Resource("run memory budget cannot reserve pools and transactions");
    run.receipt.object["reserved_host_bytes"]=runNumber(reserved);
    const bool resume=config.input.kind==RunInput::Kind::Resume;
    Journal journal(config.history.path,{config.history.storageBytes,config.history.transactionBytes,config.history.indexMemoryBytes},!resume);
    if(resume&&journal.state().hash!=run.recoveredHead)throw std::runtime_error("journal head changed after preflight");
    RankPools pools(config.pool);AdmissionContext serialization(config.limits);
    if(resume)pools.restore(run.recoveredPools,serialization);
    std::mt19937_64 hostRng(policy.seed);uint64_t historical=run.historicalDiscoveries,currentDiscoveries=0,batch=0;
    uint64_t mandatoryCaptures=0,optionalCaptures=0,captureDrops=0;
    Json binding=Json::dict();binding.object["domain"]=Json(std::is_same_v<S,SchemeZ2>?"F2":"ZT");
    binding.object["mode"]=policy.resolved().at("mode");binding.object["dimensions"]=policy.resolved().at("dimensions");
    if(settings.alternatives)binding.object["collection_rank"]=runNumber(policy.anchor);
    const std::string runId=sha256Bytes(run.receipt.at("configuration_sha256").str()+journal.state().hash+std::to_string(journal.state().sequence));
    auto transaction=[&](const char *kind,const RankPools &next) {
        Json value=Json::dict();value.object["schema"]=Json("fgm-search-transaction-v1");value.object["kind"]=Json(kind);
        value.object["workflow"]=binding;value.object["run_id"]=Json(runId);value.object["batch"]=runNumber(batch);
        value.object["stage"]=runNumber(policy.anchor);value.object["admissions"]=Json::list();value.object["pools"]=next.snapshot(serialization);return value;
    };
    auto admission=[&](const PoolMember &member,const char *origin) {
        Json item=Json::dict();item.object["scheme_id"]=Json(member.id);item.object["origin"]=Json(origin);
        item.object["rank"]=runNumber(member.scheme.rank);item.object["domain"]=binding.at("domain");item.object["scheme"]=serialization.schemeJson(member.scheme);return item;
    };
    auto recordCommit=[&](const CommitReceipt &receipt,const Json &tx) {
        std::set<std::string> credited(receipt.creditedIds.begin(),receipt.creditedIds.end());
        for(const auto &entry:tx.at("admissions").array)if(credited.count(entry.at("scheme_id").str())&&(!settings.alternatives||uint64_t(entry.at("rank").num())==policy.anchor)){
            historical=configCheckedAdd(historical,1);currentDiscoveries=configCheckedAdd(currentDiscoveries,1);
        }
        run.receipt.object["journal_sequence"]=runNumber(receipt.sequence);run.receipt.object["journal_head_sha256"]=Json(receipt.hash);
    };
    Json imports=Json::list();
    if(!resume)for(const auto &input:run.inputs){auto member=verifiedMember(input.effective,config.limits);pools.admit(member);imports.array.push_back(admission(member,"import"));}
    if(!resume)pools.stageEntry(uint32_t(policy.anchor));
    pools.refill(uint32_t(policy.anchor));
    auto start=transaction("run_start",pools);start.object["admissions"]=std::move(imports);start.object["run_record"]=run.receipt;
    start.object["seed"]=runNumber(policy.seed);start.object["resume"]=Json(resume);start.object["walker_continuation"]=Json(false);
    auto startReceipt=journal.append(start,journal.reserve(config.history.transactionBytes));recordCommit(startReceipt,start);
    run.receipt.object["run_id"]=Json(runId);run.receipt.object["execution_started"]=Json(true);
    std::string terminal;
    if(config.discoveryTarget&&historical>=*config.discoveryTarget)terminal="discovery_target_met";
    // All explicit starting parents are target-checked in input order before any
    // worker or host selection RNG is consumed.
    for(const auto &input:run.inputs)if(controlledTarget(settings,int(input.effective.rank))){terminal="existing_target_met";break;}
    if(terminal.empty()&&!pools.hasParent(uint32_t(policy.anchor)))terminal="no_eligible_parent";
    if(!terminal.empty()) {
        auto end=transaction("run_end",pools);end.object["terminal_reason"]=Json(terminal);
        recordCommit(journal.append(end,journal.reserve(config.history.transactionBytes)),end);
        run.receipt.object["status"]=Json("complete");run.receipt.object["terminal_reason"]=Json(terminal);
        run.receipt.object["counters"].object["discoveries_historical"]=runNumber(historical);return;
    }
    RunBuffer<S> current(workers),best(workers),captures(configCheckedMultiply(workers,slots));
    RunBuffer<ControlledState> states(workers);RunBuffer<ControlledCaptureMeta> metadata(configCheckedMultiply(workers,slots));
    std::unique_ptr<PackedRunBuffers> compact;
    if(packed)compact=std::make_unique<PackedRunBuffers>((workers+31)/32*32);
    std::vector<std::string> parents(static_cast<size_t>(workers));
    auto parent=std::make_unique<S>();
    for(uint64_t w=0;w<workers;++w){
        const auto *member=pools.select(uint32_t(policy.anchor),hostRng);
        if(!member)throw std::runtime_error("eligible initial parent roster disappeared");
        parents[w]=member->id;
        loadRunScheme(member->scheme,*parent);
        if(!controlledInitialize(settings,states.data[w],*parent,current.data[w],best.data[w],policy.seed,uint32_t(w)))throw std::runtime_error("worker initialization rejected");
    }
    auto workerRecords=[&](){Json all=Json::list();for(uint64_t w=0;w<workers;++w){auto item=workerRecord(states.data[w],w,current.data[w].m,best.data[w].m);
        item.object["parent_id"]=Json(parents[w]);item.object["remaining_flips"]=runNumber(settings.flipBudget-states.data[w].flips);
        item.object["remaining_controls"]=runNumber(settings.controlBudget-states.data[w].controls);all.array.push_back(std::move(item));}return all;};
    auto started=std::chrono::steady_clock::now();
    while(terminal.empty()) {
        bool live=false;
        for(uint64_t w=0;w<workers;++w)if(!controlledTerminal(states.data[w])) {
            // The first controlled-step check has no arithmetic or RNG work.
            // Resolve exhausted workers here rather than dispatch an empty batch.
            if(controlledBudget(settings,states.data[w]))controlledFail(states.data[w],ControlledOutcome::BudgetExhausted);
            else live=true;
        }
        if(!live){terminal="budget_exhausted";break;}
        // Reservation occurs before dispatch, including bounded mandatory and
        // optional evidence and durable transaction/index completion.
        uint64_t maxRank=settings.ceiling,width=0;
        for(unsigned p=0;p<3;++p)width=configCheckedAdd(width,policy.dimensions[p]*policy.dimensions[(p+1)%3]);
        for(uint64_t w=0;w<workers;++w)maxRank=std::max(maxRank,uint64_t(current.data[w].m));
        const auto perCapture=configCheckedAdd(configCheckedMultiply(maxRank,configCheckedAdd(configCheckedMultiply(width,3),16)),2048);
        const auto maxCaptures=configCheckedMultiply(workers,slots);
        const auto evidenceBound=configCheckedAdd(dump(pools.snapshot(serialization)).size(),
            configCheckedAdd(configCheckedMultiply(maxCaptures,configCheckedMultiply(perCapture,3)),configCheckedAdd(32768,workers*4096)));
        if(evidenceBound>config.history.transactionBytes)throw Resource("transaction limit cannot reserve all bounded capture evidence");
        const auto jsonBound=configCheckedAdd(jsonMemoryBytes(pools.snapshot(serialization))*2,
            configCheckedMultiply(configCheckedMultiply(maxCaptures,maxRank),configCheckedMultiply(width+6,sizeof(Json)*6)));
        if(configCheckedAdd(configCheckedAdd(configCheckedAdd(planned,reserved),uint64_t(run.receipt.at("admission_content_bytes").num())),jsonBound)>config.execution.memoryBytes)
            throw Resource("run memory cannot reserve capture serialization");
        auto reservation=journal.reserve(config.history.transactionBytes);
        ++batch;
        run.receipt.object["actual_backend"]=Json(packed?"packed":"general");
        if(compact)metalDispatch(settings.alternatives?"controlledPackedAlternativesKernel":"controlledPackedReductionKernel",size_t(workers),32,
            current.data,best.data,states.data,captures.data,metadata.data,settings,workers,config.execution.batchSteps,
            compact->terms.data,compact->pairs.data,compact->scratch.data,compact->bestTerms.data,compact->bestPairs.data);
        else metalDispatch("controlledGeneralKernel",size_t(workers),size_t(config.execution.blockSize),current.data,best.data,states.data,captures.data,metadata.data,settings,workers,config.execution.batchSteps);
        RankPools next=pools;Json observations=Json::list(),admissions=Json::list();std::set<std::string> unique;
        auto observe=[&](uint64_t w,uint64_t slot,bool mandatory){
            const auto offset=w*slots+slot;auto member=verifiedMember(runScheme(captures.data[offset]),config.limits);
            next.admit(member);if(unique.insert(member.id).second)admissions.array.push_back(admission(member,"discovery"));
            Json observation=Json::dict();observation.object["worker"]=runNumber(w);observation.object["mandatory"]=Json(mandatory);
            observation.object["slot"]=runNumber(slot);observation.object["scheme_id"]=Json(member.id);observation.object["parent_id"]=Json(parents[w]);
            observation.object["factors_id"]=Json(serialization.identity(member.scheme,false));observation.object["control"]=runNumber(metadata.data[offset].control);
            observation.object["scheme"]=serialization.schemeJson(member.scheme);
            observation.object["operation"]=runNumber(metadata.data[offset].operation);observations.array.push_back(std::move(observation));
        };
        // Check the complete dispatch before allowing any observation to commit.
        for(uint64_t w=0;w<workers;++w){
            auto &state=states.data[w];if(state.terminal==ControlledOutcome::CapacityError||state.terminal==ControlledOutcome::InvalidState||state.optionalCount>policy.optionalQuota)
                throw std::runtime_error("controlled dispatch failed validation");
            verifiedMember(runScheme(current.data[w]),config.limits);verifiedMember(runScheme(best.data[w]),config.limits);
        }
        for(uint64_t w=0;w<workers;++w)if(states.data[w].mandatoryValid){observe(w,0,true);++mandatoryCaptures;}
        for(uint64_t w=0;w<workers;++w){for(uint64_t i=0;i<states.data[w].optionalCount;++i){observe(w,i+1,false);++optionalCaptures;}captureDrops=configCheckedAdd(captureDrops,states.data[w].optionalDrops);}
        auto tx=transaction("batch",next);tx.object["admissions"]=std::move(admissions);tx.object["observations"]=std::move(observations);tx.object["workers"]=workerRecords();
        recordCommit(journal.append(tx,reservation),tx);pools=std::move(next);
        for(uint64_t w=0;w<workers;++w){if(!controlledCommit(states.data[w],true,true))throw std::runtime_error("committed observation acknowledgement rejected");
            if(states.data[w].terminal==ControlledOutcome::TargetMet)terminal="rank_target_met";
            if(!controlledBoundary(states.data[w]))throw std::runtime_error("committed capture boundary rejected");}
        if(config.discoveryTarget&&historical>=*config.discoveryTarget)terminal="discovery_target_met";
        if(terminal.empty()&&!settings.alternatives){
            const auto stage=pools.nextStage(uint32_t(policy.anchor));
            if(stage<policy.anchor){
                uint64_t charged=0;while(charged<workers&&states.data[charged].controls>=settings.controlBudget)++charged;
                if(charged==workers){terminal="control_budget_exhausted";}
                else {
                    controlledChargeStage(settings,states.data[charged]);policy.anchor=stage;settings=walkSettings(policy);
                    RankPools advanced=pools;advanced.stageEntry(stage);auto stageTx=transaction("stage",advanced);stageTx.object["charged_worker"]=runNumber(charged);stageTx.object["workers"]=workerRecords();
                    recordCommit(journal.append(stageTx,journal.reserve(config.history.transactionBytes)),stageTx);pools=std::move(advanced);
                    for(uint64_t w=0;w<workers;++w)if(!controlledTerminal(states.data[w]))states.data[w].pendingRestart=1;
                }
            }
        }
        bool restartHandled=false;Json installations=Json::list();
        if(terminal.empty())for(uint64_t w=0;w<workers;++w){
            auto &state=states.data[w];if(controlledTerminal(state)||!state.pendingRestart)continue;
            const auto *member=pools.select(uint32_t(policy.anchor),hostRng);
            if(!member){terminal="no_eligible_parent";break;}
            verifiedMember(member->scheme,config.limits);loadRunScheme(member->scheme,*parent);
            const auto before=state.controls;
            if(!controlledInstallRestart(settings,state,*parent,current.data[w],best.data[w],true))throw std::runtime_error("restart installation rejected");
            const bool installed=state.controls>before;
            if(installed)parents[w]=member->id;
            Json event=Json::dict();event.object["worker"]=runNumber(w);event.object["selected_parent_id"]=Json(member->id);
            event.object["installed"]=Json(installed);event.object["outcome"]=Json(walkOutcome(state.terminal));installations.array.push_back(std::move(event));
            restartHandled=true;
            if(state.terminal==ControlledOutcome::ExistingTargetPending){terminal="existing_target_met";break;}
        }
        if(restartHandled){auto restart=transaction("restart",pools);restart.object["workers"]=workerRecords();restart.object["installations"]=std::move(installations);
            recordCommit(journal.append(restart,journal.reserve(config.history.transactionBytes)),restart);
            for(uint64_t w=0;w<workers;++w)if(states.data[w].terminal==ControlledOutcome::ExistingTargetPending)controlledCommit(states.data[w],true,true);}
        std::cout<<"Controlled batch "<<batch<<" stage "<<policy.anchor<<" committed "<<journal.state().sequence
                 <<" discoveries "<<historical<<" current_run "<<currentDiscoveries<<std::endl;
    }
    auto end=transaction("run_end",pools);end.object["terminal_reason"]=Json(terminal);end.object["workers"]=workerRecords();
    recordCommit(journal.append(end,journal.reserve(config.history.transactionBytes)),end);
    run.receipt.object["status"]=Json("complete");run.receipt.object["terminal_reason"]=Json(terminal);run.receipt.object["workers"]=workerRecords();
    run.receipt.object["completed_batches"]=runNumber(batch);run.receipt.object["final_stage"]=runNumber(policy.anchor);
    run.receipt.object["batches_through_final_commit_microseconds"]=runNumber(uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-started).count()));
    auto &counters=run.receipt.object["counters"].object;
    auto add=[&](const char *key,uint64_t amount){counters[key]=runNumber(configCheckedAdd(uint64_t(counters[key].num()),amount));};
    for(uint64_t w=0;w<workers;++w){const auto &s=states.data[w];add("flip_attempts",s.flips);add("flips_applied",s.applied[0]);add("control_steps",s.controls);
        add("reduction_attempts",s.attempted[1]);add("terms_removed",s.removed[2]);add("expansion_attempts",s.attempted[2]);add("expansions_applied",s.applied[2]);
        add("tuple_rejections",s.tupleRejections);add("coefficient_rejections",s.coefficientRejections);add("proposal_exhaustions",s.exhaustedProposals);add("rank_blocked",s.blocked);}
    counters["mandatory_captures"]=runNumber(mandatoryCaptures);counters["optional_captures"]=runNumber(optionalCaptures);counters["capture_drops"]=runNumber(captureDrops);
    counters["discoveries_historical"]=runNumber(historical);counters["discoveries_current_run"]=runNumber(currentDiscoveries);
}
} // namespace fgm
