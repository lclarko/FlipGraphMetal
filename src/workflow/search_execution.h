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
#ifdef FGM_SEARCH_TESTING
void searchTestCheckpoint(const char *,const PreparedRun &);
Json searchTestEvaluation(const SchemeRecord &,const ReductionSettings &,Json &);
#endif
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
    // Retain completed snapshots directly in the receipt. Unwinding must not
    // read partially written device buffers or reconstruct destroyed locals.
    auto &receiptFields=run.receipt.object;
    receiptFields["committed_counters"]=run.receipt.at("counters");
    Json accounting=Json::dict();accounting.object["schema"]=Json("fgm-search-accounting-v1");
    accounting.object["work_snapshot_batch"]=runNumber(0);accounting.object["committed_batches"]=runNumber(0);
    accounting.object["dispatch_unverified"]=Json(false);receiptFields["accounting"]=std::move(accounting);
    receiptFields["completed_batches"]=runNumber(0);receiptFields["final_stage"]=runNumber(policy.anchor);
    if(!run.receipt.has("journal_sequence")){receiptFields["journal_sequence"]=runNumber(0);receiptFields["journal_head_sha256"]=Json();}
    const auto workers=config.execution.workers,slots=configCheckedAdd(policy.optionalQuota,1);
    const bool packed=config.execution.backend!="general"&&run.receipt.at("packed_eligible").boolean;
    // Pools are copied transactionally; serialization and recovery buffers have
    // explicit reserves in addition to shared Metal storage and admitted inputs.
    auto reserved=configCheckedAdd(configCheckedMultiply(config.pool.memoryBytes,2),
        configCheckedAdd(configCheckedMultiply(config.history.transactionBytes,4),config.history.indexMemoryBytes));
    if(config.additive)reserved=configCheckedAdd(reserved,configCheckedMultiply(config.limits.record,128));
    const auto planned=uint64_t(run.receipt.at("planned_buffer_bytes").num());
    if(configCheckedAdd(configCheckedAdd(planned,uint64_t(run.receipt.at("admission_content_bytes").num())),reserved)>config.execution.memoryBytes)
        throw Resource("run memory budget cannot reserve pools and transactions");
    run.receipt.object["reserved_host_bytes"]=runNumber(reserved);
    const bool resume=config.input.kind==RunInput::Kind::Resume;
    // Preflight also sees complete, unacknowledged frames. Keep its acknowledged
    // floor in the receipt until writable recovery durably promotes that prefix.
    Json recoveredCounters=run.receipt.at("counters");
    recoveredCounters.object.at("discoveries_historical")=runNumber(run.historicalDiscoveries);
    Json recoveredCommitted=recoveredCounters,recoveredHead(resume?run.recoveredHead:std::string(64,'0')),recoveredSequence=runNumber(0);
    std::array<Json*,4> recoveryDestinations{&receiptFields.at("journal_sequence"),&receiptFields.at("journal_head_sha256"),
        &receiptFields.at("counters"),&receiptFields.at("committed_counters")};
    Journal journal(config.history.path,{config.history.storageBytes,config.history.transactionBytes,config.history.indexMemoryBytes},!resume);
    const auto &recovered=journal.state();
    if(resume&&recovered.hash!=run.recoveredHead)throw std::runtime_error("journal head changed after preflight");
    recoveredSequence.integer=int64_t(recovered.sequence);
    static_assert(std::is_nothrow_move_assignable_v<Json>);
    *recoveryDestinations[0]=std::move(recoveredSequence);*recoveryDestinations[1]=std::move(recoveredHead);
    *recoveryDestinations[2]=std::move(recoveredCounters);*recoveryDestinations[3]=std::move(recoveredCommitted);
    RankPools pools(config.pool);AdmissionContext serialization(config.limits);
    if(resume)pools.restore(run.recoveredPools,serialization);
    std::mt19937_64 hostRng(policy.seed);uint64_t historical=run.historicalDiscoveries,currentDiscoveries=0,batch=0;
    uint64_t mandatoryCaptures=0,optionalCaptures=0,captureDrops=0;
    uint64_t evaluatedCount=config.additive&&resume?uint64_t(run.recoveredAdditive.at("evaluated_count").num()):0;
    const auto initialEvaluated=evaluatedCount;
    Json bestEvaluation=config.additive&&resume?run.recoveredAdditive.at("best"):Json();
    Json contract=config.additive?additiveContract(run):Json();
    if(config.additive) {
        receiptFields["best_evaluation"]=bestEvaluation;receiptFields["evaluated_historical"]=runNumber(evaluatedCount);
        receiptFields["evaluated_current_run"]=runNumber(0);
        receiptFields["pool_evictions"]=pools.snapshot(serialization).at("evictions");
    }
    auto evaluate=[&](PoolMember member) {
#if !defined(METAL_F2)
        const auto sourceId=serialization.identity(member.scheme,false);
        member.scheme=serialization.normalized(member.scheme);
        auto settings=*config.evaluation;
        const auto factorsId=serialization.identity(member.scheme,false);
        settings.seed=evaluationSeed(settings.seed,factorsId);
#ifdef FGM_SEARCH_TESTING
        Json result=searchTestEvaluation(member.scheme,settings,run.receipt);
#else
        Json result=evaluateReduction(member.scheme,settings,config.execution.blockSize,config.limits,run.receipt);
#endif
        Json value=Json::dict();value.object["schema"]=Json("fgm-additive-evaluation-v1");
        value.object["scheme_id"]=Json(member.id);value.object["factors_id"]=Json(factorsId);
        value.object["source_factors_id"]=Json(sourceId);value.object["seed"]=runNumber(settings.seed);
        value.object["settings"]=contract.at("evaluation");value.object["producer"]=contract.at("producer");
        value.object["admission_order"]=runNumber(evaluatedCount);
        value.object["additions"]=result.at("verified_circuit_additions");
        value.object["additions_by_stage"]=result.at("verified_circuit_additions_by_stage");
        for(const auto *key:{"circuit","phase_microseconds","construction","stage_sources","baseline_additions",
                            "baseline_additions_by_stage","rounds_completed","flip_attempts","flips_applied"})
            value.object[key]=result.at(key);
        serialization.verifyEvaluation(value,member.scheme);
        if(dump(value).size()>65536)throw Resource("evaluation exceeds reserved record capacity");
        member.evaluation=value;member.admissionOrder=evaluatedCount;
        member.weight=uint64_t(serialization.analyze(member.scheme).at("potential_pairs").num());
        evaluatedCount=configCheckedAdd(evaluatedCount,1);runNumber(evaluatedCount);
        if(bestEvaluation.kind==Json::Null || value.at("additions").num()<bestEvaluation.at("additions").num())bestEvaluation=value;
        return member;
#else
        (void)member;throw std::runtime_error("additive evaluation requires the signed Metal executable");
        return member;
#endif
    };
    auto publishBest=[&]() {
        if(!config.additive || receiptFields.at("best_evaluation").kind==Json::Null)return;
        const auto text=dump(receiptFields.at("best_evaluation").at("circuit"))+"\n";
        if(text.size()>config.limits.record)throw Resource("best circuit exceeds record limit");
        const std::filesystem::path path=config.output.string()+".circuits.jsonl";
        {ReceiptPhaseTimer timer(receiptFields.at("persistence_microseconds"));publishNew(path,text);}
        Json artifact=Json::dict();artifact.object["path"]=Json(path.string());artifact.object["sha256"]=Json(sha256Bytes(text));
        artifact.object["format"]=Json("jsonl");artifact.object["records"]=runNumber(1);receiptFields["circuit_artifact"]=std::move(artifact);
    };
    Json binding=Json::dict();binding.object["domain"]=Json(std::is_same_v<S,SchemeZ2>?"F2":"ZT");
    binding.object["mode"]=policy.resolved().at("mode");binding.object["dimensions"]=policy.resolved().at("dimensions");
    if(settings.alternatives)binding.object["collection_rank"]=runNumber(policy.anchor);
    if(config.additive)binding.object["additive"]=contract;
    const std::string runId=sha256Bytes(run.receipt.at("configuration_sha256").str()+journal.state().hash+std::to_string(journal.state().sequence));
    auto transaction=[&](const char *kind,const RankPools &next,const Json &evaluations=Json::list()) {
        Json value=Json::dict();value.object["schema"]=Json(config.additive?"fgm-additive-search-transaction-v1":"fgm-search-transaction-v1");value.object["kind"]=Json(kind);
        value.object["workflow"]=binding;value.object["run_id"]=Json(runId);value.object["batch"]=runNumber(batch);
        value.object["stage"]=runNumber(policy.anchor);value.object["admissions"]=Json::list();value.object["pools"]=next.snapshot(serialization);
        if(config.additive){value.object["evaluations"]=evaluations;value.object["evaluated_count"]=runNumber(evaluatedCount);value.object["best_evaluation"]=bestEvaluation;}
        return value;
    };
    auto admission=[&](const PoolMember &member,const char *origin) {
        Json item=Json::dict();item.object["scheme_id"]=Json(member.id);item.object["origin"]=Json(origin);
        item.object["rank"]=runNumber(member.scheme.rank);item.object["domain"]=binding.at("domain");item.object["scheme"]=serialization.schemeJson(member.scheme);return item;
    };
    auto commit=[&](const Json &tx,const Journal::Reservation &reservation) {
        std::set<std::string> eligible;
        for(const auto &entry:tx.at("admissions").array)
            if(!settings.alternatives||uint64_t(entry.at("rank").num())==policy.anchor)eligible.insert(entry.at("scheme_id").str());
        // Bound possible credit and prepare every allocation before append can
        // acknowledge durable work. Publishing the returned head cannot throw.
        runNumber(configCheckedAdd(historical,eligible.size()));runNumber(configCheckedAdd(currentDiscoveries,eligible.size()));
        Json counters=run.receipt.at("counters"),phase=run.receipt.at("accounting");
        Json committed=counters,head(std::string{}),sequence=runNumber(0),stage=tx.at("stage");
        Json committedBest=bestEvaluation,committedEvaluated=runNumber(evaluatedCount),committedNew=runNumber(evaluatedCount-initialEvaluated);
        Json *bestDestination=config.additive?&receiptFields.at("best_evaluation"):nullptr;
        Json *evaluatedDestination=config.additive?&receiptFields.at("evaluated_historical"):nullptr;
        Json *newDestination=config.additive?&receiptFields.at("evaluated_current_run"):nullptr;
        Json committedEvictions=config.additive?tx.at("pools").at("evictions"):Json();
        Json *evictionDestination=config.additive?&receiptFields.at("pool_evictions"):nullptr;
        auto &historyCount=counters.object.at("discoveries_historical"),&runCount=counters.object.at("discoveries_current_run");
        auto &committedHistory=committed.object.at("discoveries_historical"),&committedRun=committed.object.at("discoveries_current_run");
        if(tx.at("kind").str()=="batch")phase.object.at("committed_batches")=runNumber(batch);
        std::array<Json*,6> destinations{&receiptFields.at("counters"),&receiptFields.at("committed_counters"),
            &receiptFields.at("accounting"),&receiptFields.at("journal_sequence"),&receiptFields.at("journal_head_sha256"),&receiptFields.at("final_stage")};
        CommitReceipt acknowledged;
        { ReceiptPhaseTimer timer(receiptFields.at("persistence_microseconds"));
          acknowledged=journal.append(tx,reservation); }
        for(const auto &id:acknowledged.creditedIds)if(eligible.count(id)){++historical;++currentDiscoveries;}
        historyCount.integer=committedHistory.integer=int64_t(historical);
        runCount.integer=committedRun.integer=int64_t(currentDiscoveries);
        head.string=std::move(acknowledged.hash);sequence.integer=int64_t(acknowledged.sequence);
        static_assert(std::is_nothrow_move_assignable_v<Json>);
        *destinations[0]=std::move(counters);*destinations[1]=std::move(committed);*destinations[2]=std::move(phase);
        *destinations[3]=std::move(sequence);*destinations[4]=std::move(head);*destinations[5]=std::move(stage);
        if(config.additive){*bestDestination=std::move(committedBest);*evaluatedDestination=std::move(committedEvaluated);*newDestination=std::move(committedNew);*evictionDestination=std::move(committedEvictions);}
    };
    const auto initialBound=config.additive?configCheckedAdd(dump(run.receipt).size()+65536,
        configCheckedMultiply(configCheckedAdd(configCheckedMultiply(run.inputs.size(),3),1),65536)):0;
    if(initialBound>config.history.transactionBytes)throw Resource("initial evaluations exceed reserved transaction capacity");
    const auto startReservation=journal.reserve(config.history.transactionBytes);
    Json imports=Json::list(),initialEvaluations=Json::list();
    if(!resume)for(const auto &input:run.inputs){
        auto member=verifiedMember(input.effective,config.limits);imports.array.push_back(admission(member,"import"));
        if(config.additive){member=evaluate(std::move(member));initialEvaluations.array.push_back(member.evaluation);}
        pools.admit(member);
    }
    if(!resume)pools.stageEntry(uint32_t(policy.anchor));
    pools.refill(uint32_t(policy.anchor));
    auto start=transaction("run_start",pools,initialEvaluations);start.object["admissions"]=std::move(imports);start.object["run_record"]=run.receipt;
    start.object["seed"]=runNumber(policy.seed);start.object["resume"]=Json(resume);start.object["walker_continuation"]=Json(false);
#ifdef FGM_SEARCH_TESTING
    searchTestCheckpoint("before_start",run);
#endif
    commit(start,startReservation);
    run.receipt.object["run_id"]=Json(runId);run.receipt.object["execution_started"]=Json(true);
    std::string terminal;
    if(config.discoveryTarget&&historical>=*config.discoveryTarget)terminal="discovery_target_met";
    if(config.circuitTarget && bestEvaluation.kind!=Json::Null && uint64_t(bestEvaluation.at("additions").num())<=*config.circuitTarget)terminal="circuit_target_met";
    // All explicit starting parents are target-checked in input order before any
    // worker or host selection RNG is consumed.
    for(const auto &input:run.inputs)if(controlledTarget(settings,int(input.effective.rank))){terminal="existing_target_met";break;}
    if(terminal.empty()&&!pools.hasParent(uint32_t(policy.anchor)))terminal="no_eligible_parent";
    if(!terminal.empty()) {
        auto end=transaction("run_end",pools);end.object["terminal_reason"]=Json(terminal);
#ifdef FGM_SEARCH_TESTING
        searchTestCheckpoint("before_run_end",run);
#endif
        commit(end,journal.reserve(config.history.transactionBytes));
        run.receipt.object["status"]=Json("complete");run.receipt.object["terminal_reason"]=Json(terminal);
        run.receipt.object["counters"].object["discoveries_historical"]=runNumber(historical);publishBest();return;
    }
    RunBuffer<S> current(workers),best(workers),captures(configCheckedMultiply(workers,slots));
    RunBuffer<ControlledState> states(workers);RunBuffer<ControlledCaptureMeta> metadata(configCheckedMultiply(workers,slots));
    std::unique_ptr<PackedRunBuffers> compact;
    if(packed)compact=std::make_unique<PackedRunBuffers>((workers+31)/32*32);
    std::vector<std::string> parents(static_cast<size_t>(workers));
    auto parent=std::make_unique<S>();
    Json initialInstallations=Json::list();
    auto installation=[&](const PoolMember &member,uint64_t worker,const std::string &group,bool installed) {
        Json event=Json::dict();event.object["worker"]=runNumber(worker);event.object["selected_parent_id"]=Json(member.id);
        event.object["installed"]=Json(installed);
        if(config.additive){event.object["selection_group"]=Json(group);event.object["selected_factors_id"]=Json(serialization.identity(member.scheme,false));
            event.object["selected_additions"]=member.evaluation.at("additions");}
        return event;
    };
    for(uint64_t w=0;w<workers;++w){
        std::string group;const auto *member=pools.select(uint32_t(policy.anchor),hostRng,&group);
        if(!member)throw std::runtime_error("eligible initial parent roster disappeared");
        parents[w]=member->id;
        loadRunScheme(member->scheme,*parent);
        if(!controlledInitialize(settings,states.data[w],*parent,current.data[w],best.data[w],policy.seed,uint32_t(w)))throw std::runtime_error("worker initialization rejected");
        if(config.additive)initialInstallations.array.push_back(installation(*member,w,group,true));
    }
    auto workerRecords=[&](){Json all=Json::list();for(uint64_t w=0;w<workers;++w){auto item=workerRecord(states.data[w],w,current.data[w].m,best.data[w].m);
        item.object["parent_id"]=Json(parents[w]);item.object["remaining_flips"]=runNumber(settings.flipBudget-states.data[w].flips);
        item.object["remaining_controls"]=runNumber(settings.controlBudget-states.data[w].controls);all.array.push_back(std::move(item));}return all;};
    receiptFields["workers"]=Json::list();
    auto snapshot=[&]() {
        Json counters=run.receipt.at("counters"),phase=run.receipt.at("accounting"),records=workerRecords();
        for(auto &entry:counters.object)if(entry.first!="discoveries_historical"&&entry.first!="discoveries_current_run")entry.second=runNumber(0);
        auto add=[&](const char *key,uint64_t amount){auto &value=counters.object.at(key);value=runNumber(configCheckedAdd(uint64_t(value.num()),amount));};
        for(uint64_t w=0;w<workers;++w){const auto &s=states.data[w];add("flip_attempts",s.flips);add("flips_applied",s.applied[0]);add("control_steps",s.controls);
            add("reduction_attempts",s.attempted[1]);add("terms_removed",s.removed[2]);add("expansion_attempts",s.attempted[2]);add("expansions_applied",s.applied[2]);
            add("tuple_rejections",s.tupleRejections);add("coefficient_rejections",s.coefficientRejections);add("proposal_exhaustions",s.exhaustedProposals);add("rank_blocked",s.blocked);}
        counters.object.at("mandatory_captures")=runNumber(mandatoryCaptures);counters.object.at("optional_captures")=runNumber(optionalCaptures);
        counters.object.at("capture_drops")=runNumber(captureDrops);
        phase.object.at("work_snapshot_batch")=runNumber(batch);phase.object.at("dispatch_unverified")=Json(false);
        Json completed=runNumber(batch);
        receiptFields.at("counters")=std::move(counters);receiptFields.at("workers")=std::move(records);
        receiptFields.at("accounting")=std::move(phase);receiptFields.at("completed_batches")=std::move(completed);
    };
    snapshot();
    if(config.additive) {
        auto initialized=transaction("initialize",pools);initialized.object["installations"]=std::move(initialInstallations);
        initialized.object["workers"]=workerRecords();commit(initialized,journal.reserve(config.history.transactionBytes));
    }
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
        auto evidenceBound=configCheckedAdd(dump(pools.snapshot(serialization)).size(),
            configCheckedAdd(configCheckedMultiply(maxCaptures,configCheckedMultiply(perCapture,3)),configCheckedAdd(32768,workers*4096)));
        if(config.additive)evidenceBound=configCheckedAdd(evidenceBound,configCheckedMultiply(configCheckedAdd(configCheckedMultiply(maxCaptures,2),1),65536));
        if(evidenceBound>config.history.transactionBytes)throw Resource("transaction limit cannot reserve all bounded capture evidence");
        const auto jsonBound=configCheckedAdd(jsonMemoryBytes(pools.snapshot(serialization))*2,
            configCheckedMultiply(configCheckedMultiply(maxCaptures,maxRank),configCheckedMultiply(width+6,sizeof(Json)*6)));
        if(configCheckedAdd(configCheckedAdd(configCheckedAdd(planned,reserved),uint64_t(run.receipt.at("admission_content_bytes").num())),jsonBound)>config.execution.memoryBytes)
            throw Resource("run memory cannot reserve capture serialization");
#ifdef FGM_SEARCH_TESTING
        searchTestCheckpoint("before_reserve",run);
#endif
        auto reservation=journal.reserve(config.history.transactionBytes);
        ++batch;
        run.receipt.object["actual_backend"]=Json(packed?"packed":"general");
        receiptFields.at("accounting").object.at("dispatch_unverified")=Json(true);
        { ReceiptPhaseTimer timer(receiptFields.at("dispatch_microseconds"));
          if(compact)metalDispatch(settings.alternatives?"controlledPackedAlternativesKernel":"controlledPackedReductionKernel",size_t(workers),32,
              current.data,best.data,states.data,captures.data,metadata.data,settings,workers,config.execution.batchSteps,
              compact->terms.data,compact->pairs.data,compact->scratch.data,compact->bestTerms.data,compact->bestPairs.data);
          else metalDispatch("controlledGeneralKernel",size_t(workers),size_t(config.execution.blockSize),current.data,best.data,states.data,captures.data,metadata.data,settings,workers,config.execution.batchSteps); }
        RankPools next=pools;Json observations=Json::list(),admissions=Json::list();std::set<std::string> unique;
        auto observe=[&](uint64_t w,uint64_t slot,bool mandatory){
            const auto offset=w*slots+slot;auto member=verifiedMember(runScheme(captures.data[offset]),config.limits);
            if(!config.additive)next.admit(member);if(unique.insert(member.id).second)admissions.array.push_back(admission(member,"discovery"));
            Json observation=Json::dict();observation.object["worker"]=runNumber(w);observation.object["mandatory"]=Json(mandatory);
            observation.object["slot"]=runNumber(slot);observation.object["scheme_id"]=Json(member.id);observation.object["parent_id"]=Json(parents[w]);
            observation.object["factors_id"]=Json(serialization.identity(member.scheme,false));observation.object["control"]=runNumber(metadata.data[offset].control);
            observation.object["scheme"]=serialization.schemeJson(member.scheme);
            observation.object["operation"]=runNumber(metadata.data[offset].operation);observations.array.push_back(std::move(observation));
        };
        // Check the complete dispatch before allowing any observation to commit.
        { ReceiptPhaseTimer timer(receiptFields.at("verification_microseconds"));
        for(uint64_t w=0;w<workers;++w){
            auto &state=states.data[w];if(state.terminal==ControlledOutcome::CapacityError||state.terminal==ControlledOutcome::InvalidState||state.optionalCount>policy.optionalQuota)
                throw std::runtime_error("controlled dispatch failed validation");
            verifiedMember(runScheme(current.data[w]),config.limits);verifiedMember(runScheme(best.data[w]),config.limits);
        }
        for(uint64_t w=0;w<workers;++w)if(states.data[w].mandatoryValid){observe(w,0,true);++mandatoryCaptures;}
        for(uint64_t w=0;w<workers;++w){for(uint64_t i=0;i<states.data[w].optionalCount;++i){observe(w,i+1,false);++optionalCaptures;}captureDrops=configCheckedAdd(captureDrops,states.data[w].optionalDrops);}
        }
        Json evaluations=Json::list();
        if(config.additive)for(const auto &item:admissions.array) {
            if(item.at("rank").num()!=23 || journal.contains(item.at("scheme_id").str()))continue;
            auto member=verifiedMember(serialization.fromJson(item.at("scheme")),config.limits);
            member=evaluate(std::move(member));next.admit(member);evaluations.array.push_back(member.evaluation);
        }
        snapshot();
        auto tx=transaction("batch",next,evaluations);tx.object["admissions"]=std::move(admissions);tx.object["observations"]=std::move(observations);tx.object["workers"]=run.receipt.at("workers");
        commit(tx,reservation);pools=std::move(next);
        for(uint64_t w=0;w<workers;++w){if(!controlledCommit(states.data[w],true,true))throw std::runtime_error("committed observation acknowledgement rejected");
            if(states.data[w].terminal==ControlledOutcome::TargetMet)terminal="rank_target_met";
            if(!controlledBoundary(states.data[w]))throw std::runtime_error("committed capture boundary rejected");}
        if(config.discoveryTarget&&historical>=*config.discoveryTarget)terminal="discovery_target_met";
        if(config.circuitTarget && bestEvaluation.kind!=Json::Null && uint64_t(bestEvaluation.at("additions").num())<=*config.circuitTarget)terminal="circuit_target_met";
        if(terminal.empty()&&config.additive&&batch>=config.execution.maxBatches)terminal="batch_limit";
        if(terminal.empty()&&!settings.alternatives){
            const auto stage=pools.nextStage(uint32_t(policy.anchor));
            if(stage<policy.anchor){
                uint64_t charged=0;while(charged<workers&&states.data[charged].controls>=settings.controlBudget)++charged;
                if(charged==workers){terminal="control_budget_exhausted";}
                else {
                    controlledChargeStage(settings,states.data[charged]);policy.anchor=stage;settings=walkSettings(policy);
                    snapshot();
                    RankPools advanced=pools;advanced.stageEntry(stage);auto stageTx=transaction("stage",advanced);stageTx.object["charged_worker"]=runNumber(charged);stageTx.object["workers"]=workerRecords();
                    commit(stageTx,journal.reserve(config.history.transactionBytes));pools=std::move(advanced);
                    for(uint64_t w=0;w<workers;++w)if(!controlledTerminal(states.data[w]))states.data[w].pendingRestart=1;
                }
            }
        }
        bool restartHandled=false;Json installations=Json::list();
        if(terminal.empty())for(uint64_t w=0;w<workers;++w){
            auto &state=states.data[w];if(controlledTerminal(state)||!state.pendingRestart)continue;
            std::string group;const auto *member=pools.select(uint32_t(policy.anchor),hostRng,&group);
            if(!member){terminal="no_eligible_parent";break;}
            verifiedMember(member->scheme,config.limits);loadRunScheme(member->scheme,*parent);
            const auto before=state.controls;
            if(!controlledInstallRestart(settings,state,*parent,current.data[w],best.data[w],true))throw std::runtime_error("restart installation rejected");
            const bool installed=state.controls>before;
            if(installed)parents[w]=member->id;
            snapshot();
            Json event=installation(*member,w,group,installed);
            event.object["outcome"]=Json(walkOutcome(state.terminal));installations.array.push_back(std::move(event));
            restartHandled=true;
            if(state.terminal==ControlledOutcome::ExistingTargetPending){terminal="existing_target_met";break;}
        }
        if(restartHandled){auto restart=transaction("restart",pools);restart.object["workers"]=workerRecords();restart.object["installations"]=std::move(installations);
            commit(restart,journal.reserve(config.history.transactionBytes));
            for(uint64_t w=0;w<workers;++w)if(states.data[w].terminal==ControlledOutcome::ExistingTargetPending)controlledCommit(states.data[w],true,true);}
        std::cout<<"Controlled batch "<<batch<<" stage "<<policy.anchor<<" committed "<<journal.state().sequence
                 <<" discoveries "<<historical<<" current_run "<<currentDiscoveries<<std::endl;
    }
    snapshot();
    auto end=transaction("run_end",pools);end.object["terminal_reason"]=Json(terminal);end.object["workers"]=run.receipt.at("workers");
#ifdef FGM_SEARCH_TESTING
    searchTestCheckpoint("before_run_end",run);
#endif
    commit(end,journal.reserve(config.history.transactionBytes));
    run.receipt.object["status"]=Json("complete");run.receipt.object["terminal_reason"]=Json(terminal);run.receipt.object["workers"]=workerRecords();
    run.receipt.object["completed_batches"]=runNumber(batch);run.receipt.object["final_stage"]=runNumber(policy.anchor);
    run.receipt.object["batches_through_final_commit_microseconds"]=runNumber(uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-started).count()));
    publishBest();
}
} // namespace fgm
