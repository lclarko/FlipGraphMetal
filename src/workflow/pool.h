#pragma once
#include "scheme_io.h"
#include "run_config.h"
#include <deque>
#include <random>

namespace fgm {
struct PoolSettings {
    uint64_t capacity = 16, reserve = 16, memoryBytes = 1024*1024, threshold = 1;
    std::string selector = "uniform";
};
struct PoolMember {
    std::string id;
    SchemeRecord scheme;
    uint64_t weight = 0;
};
// Only active parents and bounded stage-entry reserves live in memory. Historical
// membership is authoritative in Journal and is deliberately absent here.
class RankPools {
    PoolSettings settings;
    std::map<uint32_t,std::deque<PoolMember>> active, reserves;
    static uint64_t bytes(const PoolMember &m) {
        uint64_t result=sizeof(m)+m.id.capacity()+1;
        for(const auto &matrix:m.scheme.f) {
            result=configCheckedAdd(result,matrix.capacity()*sizeof(Matrix::value_type));
            for(const auto &row:matrix) result=configCheckedAdd(result,row.capacity()*sizeof(int64_t));
        }
        return configCheckedAdd(result,jsonMemoryBytes(m.scheme.sourceMetadata));
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
           (settings.selector!="uniform"&&settings.selector!="flips")) throw std::runtime_error("invalid retained pool settings");
    }
    bool admit(const PoolMember &member) {
        auto &roster=active[member.scheme.rank];
        for(const auto &old:roster) if(old.id==member.id) return false;
        // The caller works on a transaction copy, so a capacity exception cannot
        // alter committed live pools. Duplicates never refresh FIFO position.
        if(roster.size()==settings.capacity) roster.pop_front();
        roster.push_back(member); bounded(); return true;
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
    const PoolMember *select(uint32_t rank,std::mt19937_64 &rng) {
        refill(rank); auto &roster=active[rank]; if(roster.empty()) return nullptr;
        uint64_t total=0;
        if(settings.selector=="flips") for(const auto &m:roster) total=configCheckedAdd(total,m.weight);
        if(!total) return &roster[size_t(draw(rng,roster.size()))];
        const auto sample=draw(rng,total); uint64_t sum=0;
        for(const auto &m:roster) {sum=configCheckedAdd(sum,m.weight);if(sum>sample)return &m;}
        throw std::runtime_error("invalid weighted parent roster");
    }
    Json snapshot(AdmissionContext &context) const {
        Json result=Json::dict(); result.object["schema"]=Json("active-rank-pool-v1");
        auto encode=[&](const auto &map) {
            Json ranks=Json::list();
            for(const auto &entry:map) {
                Json rank=Json::dict(),members=Json::list(); rank.object["rank"]=Json(int64_t(entry.first));
                for(const auto &member:entry.second) {
                    Json item=Json::dict(); item.object["scheme_id"]=Json(member.id);
                    item.object["scheme"]=context.schemeJson(member.scheme);
                    item.object["weight"]=Json(int64_t(member.weight)); members.array.push_back(std::move(item));
                }
                rank.object["members"]=std::move(members); ranks.array.push_back(std::move(rank));
            }
            return ranks;
        };
        result.object["active"]=encode(active);result.object["reserves"]=encode(reserves);return result;
    }
    void restore(const Json &snapshot,AdmissionContext &context) {
        if(snapshot.at("schema").str()!="active-rank-pool-v1")throw std::runtime_error("unsupported pool history");
        active.clear();reserves.clear();
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
                    m.weight=uint64_t(weight);roster.push_back(std::move(m));bounded();
                }
            }
        };
        decode(snapshot.at("active"),active,settings.capacity);decode(snapshot.at("reserves"),reserves,settings.reserve);bounded();
    }
};
} // namespace fgm
