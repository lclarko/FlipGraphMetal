#pragma once
#include "scheme_io.h"
#include "run_config.h"
#include "journal.h"
#include <deque>
#include <random>
#include <tuple>

namespace fgm {
struct PoolSettings {
    uint64_t capacity = 16, reserve = 16, memoryBytes = 1024*1024, threshold = 1;
    std::string selector = "uniform";
    bool additive = false;
    uint64_t eliteCapacity = 8;
};
struct PoolMember {
    std::string id;
    SchemeRecord scheme;
    uint64_t weight = 0;
    Json evaluation;
    uint64_t admissionOrder = 0;
};
// Only active parents and bounded stage-entry reserves live in memory. Historical
// membership is authoritative in Journal and is deliberately absent here.
class RankPools {
    PoolSettings settings;
    std::map<uint32_t,std::deque<PoolMember>> active, reserves;
    uint64_t evictions=0;
    static uint64_t bytes(const PoolMember &m) {
        uint64_t result=sizeof(m)+m.id.capacity()+1;
        for(const auto &matrix:m.scheme.f) {
            result=configCheckedAdd(result,matrix.capacity()*sizeof(Matrix::value_type));
            for(const auto &row:matrix) result=configCheckedAdd(result,row.capacity()*sizeof(int64_t));
        }
        result=configCheckedAdd(result,jsonMemoryBytes(m.scheme.sourceMetadata));
        return configCheckedAdd(result,jsonMemoryBytes(m.evaluation));
    }
    void bounded() const {
        // Charge deque block/map overhead conservatively as one full member per
        // entry plus a block per rank, in addition to actual owned factor data.
        uint64_t total=sizeof(*this);
        for(const auto *map:{&active,&reserves}) for(const auto &entry:*map) {
            total=configCheckedAdd(total,4096+sizeof(entry)+8*sizeof(void*));
            for(const auto &member:entry.second) total=configCheckedAdd(total,configCheckedAdd(bytes(member),sizeof(PoolMember)));
        }
        if(total>settings.memoryBytes) throw Resource("retained pools exceed memory budget");
    }
public:
    explicit RankPools(PoolSettings value):settings(std::move(value)) {
        if(!settings.capacity || !settings.reserve || !settings.threshold || settings.threshold>settings.capacity ||
           (settings.additive ? (settings.selector!="uniform"&&settings.selector!="cost-diverse") :
                                (settings.selector!="uniform"&&settings.selector!="flips")) ||
           (settings.additive && (!settings.eliteCapacity || settings.eliteCapacity>=settings.capacity)))
            throw std::runtime_error("invalid retained pool settings");
    }
    bool admit(const PoolMember &member) {
        if(settings.additive && (member.scheme.rank!=23 || member.evaluation.kind!=Json::Object ||
            member.evaluation.at("additions").num()<0))throw std::runtime_error("evaluated rank-23 pool member required");
        auto &roster=active[member.scheme.rank];
        for(const auto &old:roster) if(old.id==member.id) return false;
        if(settings.additive && roster.size()==settings.capacity)evictions=configCheckedAdd(evictions,1);
        // The caller works on a transaction copy, so a capacity exception cannot
        // alter committed live pools. Duplicates never refresh FIFO position.
        if(settings.selector=="cost-diverse") {
            std::vector<PoolMember> candidates(roster.begin(),roster.end());candidates.push_back(member);
            std::stable_sort(candidates.begin(),candidates.end(),[](const auto &a,const auto &b){
                return std::make_tuple(a.evaluation.at("additions").num(),a.admissionOrder,a.id)<
                       std::make_tuple(b.evaluation.at("additions").num(),b.admissionOrder,b.id);});
            const auto elite=std::min<size_t>(settings.eliteCapacity,candidates.size());
            std::stable_sort(candidates.begin()+elite,candidates.end(),[](const auto &a,const auto &b){
                return a.admissionOrder!=b.admissionOrder?a.admissionOrder>b.admissionOrder:a.id<b.id;});
            if(candidates.size()>settings.capacity)candidates.resize(size_t(settings.capacity));
            roster.assign(candidates.begin(),candidates.end());
        } else {
            if(roster.size()==settings.capacity) roster.pop_front();
            roster.push_back(member);
        }
        bounded(); return true;
    }
    void stageEntry(uint32_t rank) {
        auto &reserve=reserves[rank]; reserve.clear();
        const auto &roster=active[rank];
        for(const auto &member:roster) {
            if(reserve.size()==settings.reserve) break;
            reserve.push_back(member);
        }
        bounded();
    }
    void refill(uint32_t rank) {
        auto &roster=active[rank];
        if(!roster.empty()) return;
        for(const auto &member:reserves[rank]) {
            if(roster.size()==settings.capacity) break;
            roster.push_back(member);
        }
        bounded();
    }
    uint32_t nextStage(uint32_t rank) const {
        for(const auto &entry:active) if(entry.first<rank && entry.second.size()>=settings.threshold) return entry.first;
        return rank;
    }
    bool hasParent(uint32_t rank) const {
        const auto found=active.find(rank);
        return found!=active.end()&&!found->second.empty();
    }
    static uint64_t draw(std::mt19937_64 &rng,uint64_t bound) {
        if(!bound) throw std::runtime_error("empty parent selection bound");
        const uint64_t threshold=(uint64_t(0)-bound)%bound;
        for(unsigned i=0;i<64;++i) {const auto x=rng();if(x>=threshold)return x%bound;}
        throw std::runtime_error("host selection exhausted 64 draws");
    }
    const PoolMember *select(uint32_t rank,std::mt19937_64 &rng,std::string *group=nullptr) {
        refill(rank); auto &roster=active[rank]; if(roster.empty()) return nullptr;
        if(settings.selector=="cost-diverse") {
            const auto elite=std::min<uint64_t>(settings.eliteCapacity,roster.size());
            const bool explore=roster.size()>elite && draw(rng,2)!=0;
            if(group)*group=explore?"exploration":"elite";
            return &roster[size_t(explore?elite+draw(rng,roster.size()-elite):draw(rng,elite))];
        }
        if(group)*group=settings.selector;
        uint64_t total=0;
        if(settings.selector=="flips") for(const auto &m:roster) total=configCheckedAdd(total,m.weight);
        if(!total) return &roster[size_t(draw(rng,roster.size()))];
        const auto sample=draw(rng,total); uint64_t sum=0;
        for(const auto &m:roster) {sum=configCheckedAdd(sum,m.weight);if(sum>sample)return &m;}
        throw std::runtime_error("invalid weighted parent roster");
    }
    Json snapshot(AdmissionContext &context) const {
        Json result=Json::dict(); result.object["schema"]=Json(settings.additive?"additive-rank-pool-v1":"active-rank-pool-v1");
        if(settings.additive)result.object["evictions"]=Json(int64_t(evictions));
        auto encode=[&](const auto &map) {
            Json ranks=Json::list();
            for(const auto &entry:map) {
                Json rank=Json::dict(),members=Json::list(); rank.object["rank"]=Json(int64_t(entry.first));
                for(const auto &member:entry.second) {
                    Json item=Json::dict(); item.object["scheme_id"]=Json(member.id);
                    item.object["scheme"]=context.schemeJson(member.scheme);
                    item.object["weight"]=Json(int64_t(member.weight));
                    if(settings.additive){item.object["evaluation"]=member.evaluation;item.object["admission_order"]=Json(int64_t(member.admissionOrder));}
                    members.array.push_back(std::move(item));
                }
                rank.object["members"]=std::move(members); ranks.array.push_back(std::move(rank));
            }
            return ranks;
        };
        result.object["active"]=encode(active);result.object["reserves"]=encode(reserves);return result;
    }
    void restore(const Json &snapshot,AdmissionContext &context) {
        if(snapshot.at("schema").str()!=(settings.additive?"additive-rank-pool-v1":"active-rank-pool-v1"))throw std::runtime_error("unsupported pool history");
        active.clear();reserves.clear();
        if(settings.additive) {
            if(snapshot.at("evictions").num()<0)throw std::runtime_error("invalid eviction counter");
            evictions=uint64_t(snapshot.at("evictions").num());
        }
        auto decode=[&](const Json &ranks,auto &map,uint64_t capacity) {
            if(ranks.kind!=Json::Array || ranks.array.size()>350)throw std::runtime_error("invalid pool rank inventory");
            for(const auto &entry:ranks.array) {
                const auto rank=entry.at("rank").num();
                if(rank<1||rank>350||map.count(uint32_t(rank)))throw std::runtime_error("invalid duplicate pool rank");
                const auto &members=entry.at("members");
                if(members.kind!=Json::Array||members.array.size()>capacity)throw std::runtime_error("pool history exceeds capacity");
                auto &roster=map[uint32_t(rank)];std::set<std::string> ids;
                for(const auto &item:members.array) {
                    PoolMember m{item.at("scheme_id").str(),context.fromJson(item.at("scheme")),0};
                    if(!context.verify(m.scheme))throw std::runtime_error("invalid tensor in pool history");
                    if(m.scheme.rank!=rank||context.identity(m.scheme,true)!=m.id||!ids.insert(m.id).second)throw std::runtime_error("invalid pool history member");
                    const auto weight=item.at("weight").num();if(weight<0||weight>1500)throw std::runtime_error("invalid pool history weight");
                    const auto analysis=context.analyze(m.scheme);
                    if(!analysis.at("search_eligible").boolean||analysis.at("potential_pairs").num()!=weight)
                        throw std::runtime_error("pool history eligibility or weight mismatch");
                    if(settings.additive) {
                        m.evaluation=item.at("evaluation");context.verifyEvaluation(m.evaluation,m.scheme);
                        if(item.at("admission_order").num()<0)throw std::runtime_error("invalid pool admission order");
                        m.admissionOrder=uint64_t(item.at("admission_order").num());
                        if(m.evaluation.at("admission_order").num()!=int64_t(m.admissionOrder))
                            throw std::runtime_error("pool evaluation order mismatch");
                    }
                    m.weight=uint64_t(weight);roster.push_back(std::move(m));bounded();
                }
            }
        };
        decode(snapshot.at("active"),active,settings.capacity);decode(snapshot.at("reserves"),reserves,settings.reserve);bounded();
    }
};

inline Json additiveContract(const Json &configuration,const Json &receipt) {
    if(configuration.at("workflow").str()!="additive-search")throw std::runtime_error("additive workflow required");
    Json contract=Json::dict(),producer=Json::dict(),execution=Json::dict();
    for(const auto *key:{"executable_sha256","library_mode","library_sha256"})producer.object[key]=receipt.at(key);
    for(const auto *key:{"workers","batch_steps","block_size","backend"})execution.object[key]=configuration.at("execution").at(key);
    contract.object["producer"]=std::move(producer);contract.object["execution"]=std::move(execution);
    contract.object["evaluation"]=configuration.at("evaluation");contract.object["pool"]=configuration.at("pool");
    contract.object["generation"]=configuration.at("policy");contract.object["generation"].object["seed"]=Json(int64_t(0));
    return contract;
}

// Bounded mathematical and population replay shared by native resume/export.
// Historical uniqueness comes from Journal's checked novelty list, not a new
// unbounded identity cache. A snapshot is checked against derived transitions.
class AdditiveReplay {
    AdmissionLimits limits;
    std::optional<RankPools> pools;
    Json contract,best;
    uint64_t count=0,workers=0;
    std::string runId,previousKind,previousHash=std::string(64,'0');
    uint64_t previousSequence=0,batch=0;
    Json workerState;
    std::mt19937_64 rng;
    std::map<uint64_t,std::string> parents;
public:
    explicit AdditiveReplay(AdmissionLimits value={}):limits(value) {}
    Json state() const {
        Json value=Json::dict();value.object["contract"]=contract;value.object["best"]=best;
        value.object["evaluated_count"]=Json(int64_t(count));
        if(pools){AdmissionContext context(limits);value.object["evictions"]=pools->snapshot(context).at("evictions");}
        return value;
    }
    void apply(const Json &tx,const CommitReceipt &commit) {
        const auto &novelIds=commit.novelIds;
        if(commit.sequence!=previousSequence+1)throw std::runtime_error("nonsequential additive history");
        if(tx.at("schema").str()!="fgm-additive-search-transaction-v1")throw std::runtime_error("additive transaction required");
        const auto &bound=tx.at("workflow").at("additive");const auto kind=tx.at("kind").str();
        const bool first=!pools.has_value();
        if(first) {
            if(kind!="run_start" || tx.at("resume").boolean)throw std::runtime_error("additive history requires initial admissions");
            contract=bound;
            const auto &p=contract.at("pool");
            PoolSettings settings{uint64_t(p.at("capacity_per_rank").num()),uint64_t(p.at("reserve_per_rank").num()),
                uint64_t(p.at("memory_bytes").num()),uint64_t(p.at("stage_threshold").num()),p.at("selector").str(),true,
                uint64_t(p.at("elite_capacity").num())};
            for(const auto *key:{"capacity_per_rank","reserve_per_rank","memory_bytes","stage_threshold","elite_capacity"})
                if(p.at(key).num()<=0)throw std::runtime_error("invalid additive pool contract");
            pools.emplace(settings);
        }
        if(dump(bound)!=dump(contract))throw std::runtime_error("additive history contract mismatch");
        AdmissionContext context(limits);
        if(kind=="run_start") {
            if(tx.at("resume").kind!=Json::Boolean || tx.at("resume").boolean==first)
                throw std::runtime_error("invalid additive resume boundary");
            const auto &record=tx.at("run_record");
            const auto nextRun=tx.at("run_id").str();
            if(nextRun==runId || record.at("journal_sequence").num()!=int64_t(previousSequence) ||
               record.at("journal_head_sha256").str()!=previousHash ||
               nextRun!=sha256Bytes(record.at("configuration_sha256").str()+previousHash+std::to_string(previousSequence)) ||
               tx.at("batch").num()!=0 || tx.at("walker_continuation").kind!=Json::Boolean || tx.at("walker_continuation").boolean)
                throw std::runtime_error("additive run predecessor mismatch");
            batch=0;workerState=Json();
            if(dump(additiveContract(record.at("configuration"),record))!=dump(contract))
                throw std::runtime_error("additive run record contract mismatch");
            const auto seed=tx.at("seed").num();
            if(seed<0 || seed>UINT32_MAX || seed!=record.at("configuration").at("policy").at("seed").num())
                throw std::runtime_error("additive parent seed mismatch");
            rng.seed(uint64_t(seed));parents.clear();runId=tx.at("run_id").str();
            if(contract.at("execution").at("workers").num()<=0)throw std::runtime_error("invalid additive worker count");
            workers=uint64_t(contract.at("execution").at("workers").num());
        } else {
            if(tx.at("run_id").str()!=runId)throw std::runtime_error("additive run boundary mismatch");
            if(kind=="batch") {
                if(previousKind!="initialize"&&previousKind!="batch"&&previousKind!="restart")
                    throw std::runtime_error("batch outside initialized run");
                if(workerState.kind!=Json::Array)throw std::runtime_error("missing preceding worker state");
                for(const auto &worker:workerState.array)
                    if(worker.at("terminal").str()=="active"&&worker.at("pending_restart").boolean)
                        throw std::runtime_error("batch bypassed required restart");
                ++batch;
            } else if((kind=="initialize"&&previousKind!="run_start") ||
                      (kind=="restart"&&previousKind!="batch") || previousKind=="run_end")
                throw std::runtime_error("invalid additive transaction order");
            if(tx.at("batch").num()!=int64_t(batch))throw std::runtime_error("additive batch number mismatch");
        }
        if(tx.at("evaluations").kind!=Json::Array || tx.at("admissions").kind!=Json::Array)
            throw std::runtime_error("additive evaluations and admissions must be arrays");
        if(kind!="run_start"&&kind!="batch"&&kind!="initialize"&&kind!="restart"&&kind!="run_end")
            throw std::runtime_error("invalid additive transaction kind");
        if(kind!="batch"&&!(first&&kind=="run_start")&&!tx.at("admissions").array.empty())
            throw std::runtime_error("admission outside additive capture boundary");
        if(kind=="batch") {
            // Admissions retain the first presentation in deterministic capture
            // order. A valid but unrelated tensor is not evidence of a capture.
            if(tx.at("observations").kind!=Json::Array)throw std::runtime_error("additive captures must be array");
            std::set<std::string> seen;size_t admitted=0;
            for(const auto &capture:tx.at("observations").array) {
                auto scheme=context.fromJson(capture.at("scheme"),"ZT");
                const auto id=context.identity(scheme,true);
                if(!context.verify(scheme)||id!=capture.at("scheme_id").str()||
                   context.identity(scheme,false)!=capture.at("factors_id").str())
                    throw std::runtime_error("invalid additive capture binding");
                if(!seen.insert(id).second)continue;
                if(admitted>=tx.at("admissions").array.size())throw std::runtime_error("missing captured admission");
                const auto &entry=tx.at("admissions").array[admitted++];
                if(entry.at("scheme_id").str()!=id ||
                   context.identity(context.fromJson(entry.at("scheme"),"ZT"),false)!=context.identity(scheme,false))
                    throw std::runtime_error("additive admission differs from first capture");
            }
            if(admitted!=tx.at("admissions").array.size())throw std::runtime_error("uncaptured additive admission");
        }
        const std::set<std::string> novel(novelIds.begin(),novelIds.end());
        size_t index=0,initialIndex=0;
        std::vector<std::string> startingFactors;
        if(first) {
            std::set<std::string> seen;
            for(const auto &presentation:tx.at("run_record").at("presentations").array)
                if(seen.insert(presentation.at("scheme_id").str()).second)
                    startingFactors.push_back(presentation.at("effective_factors_id").str());
        }
        for(const auto &admission:tx.at("admissions").array) {
            auto source=context.fromJson(admission.at("scheme"),"ZT");
            const auto id=context.identity(source,true);
            if(!context.verify(source) || id!=admission.at("scheme_id").str() || source.rank!=admission.at("rank").num())
                throw std::runtime_error("invalid additive admission");
            if(admission.at("origin").str()!=(first?"import":"discovery") || admission.at("domain").str()!="ZT")
                throw std::runtime_error("additive admission provenance mismatch");
            if(first && (initialIndex>=startingFactors.size() ||
                         startingFactors[initialIndex++]!=context.identity(source,false)))
                throw std::runtime_error("initial evaluation differs from admitted input");
            if(source.rank!=23 || !novel.count(id))continue;
            if(kind!="batch" && !(first&&kind=="run_start"))throw std::runtime_error("evaluation outside admission boundary");
            if(index>=tx.at("evaluations").array.size())throw std::runtime_error("missing additive evaluation");
            const auto &evaluation=tx.at("evaluations").array[index++];
            auto effective=context.normalized(source);
            if(evaluation.at("source_factors_id").str()!=context.identity(source,false) ||
               evaluation.at("admission_order").num()!=int64_t(count) ||
               dump(evaluation.at("settings"))!=dump(contract.at("evaluation")) ||
               dump(evaluation.at("producer"))!=dump(contract.at("producer")))
                throw std::runtime_error("additive evaluation admission binding mismatch");
            const auto cost=context.verifyEvaluation(evaluation,effective);
            const auto analysis=context.analyze(effective);
            if(!analysis.at("search_eligible").boolean)throw std::runtime_error("ineligible additive parent");
            PoolMember member{id,effective,uint64_t(analysis.at("potential_pairs").num()),evaluation,count};
            if(!pools->admit(member))throw std::runtime_error("duplicate additive evaluation");
            if(best.kind==Json::Null || cost<uint64_t(best.at("additions").num()))best=evaluation;
            count=configCheckedAdd(count,1);if(count>INT64_MAX)throw Resource("evaluation counter overflow");
        }
        if(first&&initialIndex!=startingFactors.size())throw std::runtime_error("missing initial parent evaluation");
        if(index!=tx.at("evaluations").array.size())throw std::runtime_error("evaluation lacks a novel admission");
        if(first)pools->stageEntry(23);
        if(kind=="initialize" || kind=="restart") {
            const auto &events=tx.at("installations");
            if(events.kind!=Json::Array || (kind=="initialize" && (events.array.size()!=workers||!parents.empty())))
                throw std::runtime_error("invalid additive parent installations");
            std::vector<uint64_t> expected;
            if(kind=="restart") {
                if(workerState.kind!=Json::Array || workerState.array.size()!=workers)throw std::runtime_error("missing pre-restart workers");
                for(uint64_t w=0;w<workers;++w)if(workerState.array[w].at("pending_restart").boolean &&
                    workerState.array[w].at("terminal").str()=="active")expected.push_back(w);
                if(expected.empty()||events.array.size()!=expected.size())throw std::runtime_error("unjustified restart selection");
            }
            uint64_t previous=0;bool havePrevious=false;size_t eventIndex=0;
            for(const auto &event:events.array) {
                const auto worker=event.at("worker").num();
                if(worker<0 || uint64_t(worker)>=workers || (havePrevious&&uint64_t(worker)<=previous))
                    throw std::runtime_error("invalid additive installation order");
                if(kind=="restart"&&uint64_t(worker)!=expected[eventIndex++])throw std::runtime_error("restart worker roster mismatch");
                previous=uint64_t(worker);havePrevious=true;std::string group;
                const auto *member=pools->select(23,rng,&group);
                if(!member || event.at("selected_parent_id").str()!=member->id || event.at("selection_group").str()!=group ||
                   event.at("selected_factors_id").str()!=context.identity(member->scheme,false) ||
                   event.at("selected_additions").num()!=member->evaluation.at("additions").num() ||
                   event.at("installed").kind!=Json::Boolean)
                    throw std::runtime_error("additive parent selection mismatch");
                if(kind=="initialize" && !event.at("installed").boolean)throw std::runtime_error("initial parent not installed");
                if(kind=="restart") {
                    const auto &before=workerState.array[size_t(worker)];
                    const auto &after=tx.at("workers").array.at(size_t(worker));
                    const auto &generation=contract.at("generation");
                    const bool budget=before.at("flips").num()>=generation.at("flip_budget").num() ||
                                      before.at("controls").num()>=generation.at("control_budget").num();
                    if(event.at("installed").boolean==budget ||
                       after.at("controls").num()!=before.at("controls").num()+(budget?0:1) ||
                       after.at("flips").num()!=before.at("flips").num() ||
                       after.at("terminal").str()!=(budget?"budget_exhausted":"active") ||
                       after.at("pending_restart").boolean!=budget)
                        throw std::runtime_error("restart installation state mismatch");
                }
                if(event.at("installed").boolean)parents[uint64_t(worker)]=member->id;
            }
        }
        if(tx.has("observations"))for(const auto &capture:tx.at("observations").array) {
            const auto worker=capture.at("worker").num();
            if(worker<0 || !parents.count(uint64_t(worker)) || parents.at(uint64_t(worker))!=capture.at("parent_id").str())
                throw std::runtime_error("additive capture parent mismatch");
        }
        if((kind=="initialize"||kind=="batch"||kind=="restart")&&!tx.has("workers"))
            throw std::runtime_error("missing additive worker inventory");
        if(tx.has("workers")) {
            const auto &states=tx.at("workers");
            if(states.kind!=Json::Array||states.array.size()!=workers)throw std::runtime_error("invalid additive worker inventory");
            for(uint64_t w=0;w<workers;++w)if(states.array[w].at("worker").num()!=int64_t(w)||
                !parents.count(w)||states.array[w].at("parent_id").str()!=parents.at(w))
                throw std::runtime_error("additive installed parent binding mismatch");
            workerState=states;
        }
        previousSequence=commit.sequence;previousHash=commit.hash;previousKind=kind;
        if(dump(pools->snapshot(context))!=dump(tx.at("pools")) || tx.at("evaluated_count").num()!=int64_t(count) ||
           dump(tx.at("best_evaluation"))!=dump(best))throw std::runtime_error("additive population transition mismatch");
    }
};
} // namespace fgm
