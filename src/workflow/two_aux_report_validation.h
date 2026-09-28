#pragma once
#include "scheme_io.h"
#include <algorithm>

// Replay the bounded report and its selected trace, never search for a circuit.
// Work counters describe the producer's run; replay checks their consistency,
// not the completeness of an enumeration whose negative traces are not stored.
namespace fgm::two_aux_report {
inline void require(bool value) {
    if(!value)throw std::runtime_error("invalid additive two-auxiliary report");
}
inline uint64_t number(const Json &value,uint64_t maximum=INT64_MAX) {
    const auto n=value.num();require(n>=0 && uint64_t(n)<=maximum);return uint64_t(n);
}
inline void object(const Json &value,size_t size) {
    require(value.kind==Json::Object && value.object.size()==size);
}
inline void array(const Json &value,size_t size) {
    require(value.kind==Json::Array && value.array.size()==size);
}
using Vector=std::vector<int64_t>;
inline int normalize(Vector &v) {
    for(auto x:v)if(x) {
        const int sign=x<0?-1:1;
        for(auto &c:v)c*=sign;
        return sign;
    }
    return 0;
}
inline Matrix directions(const Matrix &targets,uint64_t &q) {
    Matrix values(9,Vector(9));for(int i=0;i<9;++i)values[i][i]=1;
    q=0;
    for(auto v:targets)if(normalize(v)) {
        ++q;if(std::find(values.begin(),values.end(),v)==values.end())values.push_back(v);
    }
    require(values.size()<=32);return values;
}
inline std::array<int,5> gate(const Json &value,int n) {
    array(value,5);std::array<int,5> g{};
    for(int i=0;i<3;++i)g[i]=int(number(value.array[i],uint64_t(n+1)));
    for(int i=3;i<5;++i) { const auto sign=value.array[i].num();require(sign==1||sign==-1);g[i]=int(sign); }
    require(g[0]>=9 && g[1]<g[2]);return g;
}
inline std::array<int,3> route(uint64_t id,int n) {
    const int sign=(id&1)?-1:1;id/=2;
    for(int left=0;left<n-1;++left) {
        if(id<uint64_t(n-left-1))return {left,left+1+int(id),sign};
        id-=uint64_t(n-left-1);
    }
    throw std::runtime_error("invalid additive two-auxiliary route");
}
inline uint64_t routeId(int left,int right,int sign,int n) {
    return uint64_t(left*(2*n-left-1)+2*(right-left-1)+(sign<0));
}
inline void witness(const Json &w,Matrix values,uint64_t prefix,uint64_t logical) {
    object(w,8);const int n=int(values.size());
    const uint64_t second=uint64_t(n)*(n+1),raw=number(w.at("raw_slot"),prefix-1);
    require(number(w.at("first_route"))==raw/second && number(w.at("second_route"))==raw%second);
    require(logical==raw+1);
    array(w.at("helpers"),2);array(w.at("creations"),2);
    std::array<std::array<int,5>,2> creations;
    for(int h=0;h<2;++h) {
        const auto decoded=route(h?raw%second:raw/second,n+h);
        const auto [left,right,sign]=decoded;
        Vector helper(9);for(int c=0;c<9;++c)helper[c]=values[left][c]+sign*values[right][c];
        const auto norm=normalize(helper);
        require(norm && std::find(values.begin(),values.end(),helper)==values.end());
        if(h && right<n)require(raw/second<routeId(left,right,sign,n));
        const auto &stored=w.at("helpers").array[h];array(stored,9);
        for(int c=0;c<9;++c)require(stored.array[c].num()==helper[c]);
        creations[h]=gate(w.at("creations").array[h],n);
        require(creations[h]==std::array<int,5>{n+h,left,right,norm,norm*sign});
        values.push_back(std::move(helper));
    }
    const auto count=number(w.at("count"),25);array(w.at("gates"),size_t(count));
    std::vector<std::array<int,5>> gates;
    uint64_t available=(uint64_t(1)<<9)-1;
    for(const auto &encoded:w.at("gates").array) {
        const auto g=gate(encoded,n);
        require(!(available&(uint64_t(1)<<g[0])) && (available&(uint64_t(1)<<g[1])) &&
                (available&(uint64_t(1)<<g[2])));
        if(g[0]>=n)require(g==creations[g[0]-n]);
        for(int c=0;c<9;++c)require(values[g[0]][c]==g[3]*values[g[1]][c]+g[4]*values[g[2]][c]);
        available|=uint64_t(1)<<g[0];gates.push_back(g);
    }
    const uint64_t required=(uint64_t(1)<<n)-1;
    require(number(w.at("available"),(uint64_t(1)<<(n+2))-1)==available && (available&required)==required);
    uint64_t live=required;int liveCount=0;
    for(auto it=gates.rbegin();it!=gates.rend();++it)if(live&(uint64_t(1)<<(*it)[0])) {
        live|=(uint64_t(1)<<(*it)[1])|(uint64_t(1)<<(*it)[2]);++liveCount;
    }
    require((live&(uint64_t(3)<<n))==(uint64_t(3)<<n) && liveCount==n-9+2);
}
inline void validate(const Json &evaluation,const SchemeRecord &expected,const std::string &factors) {
    const auto &settings=evaluation.at("settings");
    if(!settings.has("constructor")) {
        require(!evaluation.has("two_auxiliary") && !evaluation.has("pre_two_aux_additions_by_stage"));
        if(evaluation.has("stage_sources"))for(const auto &entry:evaluation.at("stage_sources").object)
            require(entry.second.str()!="two-auxiliary" && entry.second.str()!="transpose-two-auxiliary");
        return;
    }
    const auto &option=settings.at("constructor");object(option,2);
    require(option.at("family").str()=="signed-two-aux-distinct-v1");
    const auto budget=number(option.at("max_pair_slots"),1047552);require(budget>0);
    const auto &report=evaluation.at("two_auxiliary");object(report,2);
    require(report.at("effective_factors_id").str()==factors);object(report.at("stages"),3);
    const auto &pre=evaluation.at("pre_two_aux_additions_by_stage");object(pre,3);
    const auto &baseline=evaluation.at("baseline_additions_by_stage");object(baseline,3);
    const auto &sources=evaluation.at("stage_sources");object(sources,3);
    object(evaluation.at("construction"),3);
    uint64_t baselineTotal=0;
    for(int p=0;p<3;++p) {
        const std::string key=p==2?"wt":std::string(1,"uv"[p]),out(1,"uvw"[p]);
        const auto &s=report.at("stages").at(key);
        require(s.at("schema").str()=="fgm-two-aux-report-v1" && s.at("family").str()==option.at("family").str() &&
                s.at("implementation").str()=="metal-two-aux-v1" && s.at("enumeration_order").str()=="lexicographic-pairs-plus-first-v1" &&
                s.at("stage").str()==key && number(s.at("requested_auxiliaries"))==2 && number(s.at("max_pair_slots"))==budget);
        uint64_t q=0;auto vectors=directions(expected.f[p],q);const auto n=uint64_t(vectors.size()),m=n-9;
        const auto total=n*(n-1)*n*(n+1),prefix=std::min(total,budget),floor=m+2+(p==2?q-9:0);
        require(number(s.at("target_directions"))==m && number(s.at("nonzero_output_occurrences"))==q &&
                number(s.at("improvement_floor"))==floor && number(s.at("raw_family_size"))==total && number(s.at("prefix_limit"))==prefix);
        const auto before=number(pre.at(out)),after=number(evaluation.at("additions_by_stage").at(out)),base=number(baseline.at(out),1000);
        baselineTotal+=base;
        require(after<=before && before<=base && number(s.at("baseline_cost"))==base &&
                number(s.at("incumbent_cost"))==before && number(s.at("final_cost"))==after);
        auto counter=[&](const char *name,uint64_t max=1047552){return number(s.at(name),max);};
        const auto raw=counter("raw_scanned"),prepared=counter("prepared"),logical=counter("logical_prefix"),tail=counter("batch_tail_candidates",127);
        require(raw<=prefix && logical<=raw && raw==counter("first_invalid")+counter("second_invalid")+counter("symmetry_filtered")+prepared &&
                counter("dispatched")==prepared && counter("completed")==prepared && counter("validated")==prepared &&
                counter("unvisited_raw_slots")==total-raw && tail<=prepared);
        const auto gpu=counter("gpu_rule_checks",prepared*29224),sweeps=counter("gpu_sweeps",prepared*26);
        const auto negative=counter("negative_validation_rule_checks",prepared*1124);
        require(sweeps>=prepared && gpu>=2*sweeps && tail<=raw-logical);
        object(s.at("microseconds"),7);
        for(const auto *name:{"allocation","preparation","dispatch","gpu","witness_validation","transposition","verification"})
            number(s.at("microseconds").at(name));
        const auto old=evaluation.at("construction").at(key).at("status").str(),reason=s.at("stop_reason").str();
        require(old=="target-only" || old=="one-auxiliary" || old=="family-exhausted");
        if(old!="family-exhausted")require(before<=m+(old=="one-auxiliary")+(p==2?q-9:0));
        const bool skip=reason=="smaller-family-succeeded" || reason=="proved-no-improvement";
        require(s.at("coverage").str()==(skip?"not-searched":raw==total?"full":"partial"));
        const auto source=sources.at(out).str();
        require(source=="baseline" || (p<2 && source=="cancellation") || (p==2 && (source=="transpose-pair" || source=="transpose-cancellation")) ||
                source==(p==2?"transpose-two-auxiliary":"two-auxiliary"));
        if(source=="baseline")require(after==base);
        else require(after<base);
        if(source=="cancellation" || source=="transpose-cancellation")require(old!="family-exhausted");
        if(skip) {
            require(raw==0 && logical==0 && tail==0 && gpu==0 && sweeps==0 && negative==0);
            require(reason=="smaller-family-succeeded"?old!="family-exhausted":old=="family-exhausted" && before<=floor);
        } else {
            require(old=="family-exhausted" && before>floor);
            if(reason=="bound-attained") {
                require(prepared>0 && tail<prepared && negative<=(prepared-1)*1124 &&
                        after==floor && after<before && counter("used_auxiliaries",2)==2 &&
                        source==(p==2?"transpose-two-auxiliary":"two-auxiliary"));
                witness(s.at("selected_witness"),std::move(vectors),prefix,logical);
                continue;
            }
            require(reason=="budget-exhausted" || reason=="family-exhausted");
            require(raw==prefix && logical==raw && tail==0 && negative>=2*prepared &&
                    (reason=="family-exhausted")== (prefix==total));
        }
        require(s.at("selected_witness").kind==Json::Null && counter("used_auxiliaries",2)==0 && before==after &&
                source!="two-auxiliary" && source!="transpose-two-auxiliary");
    }
    require(number(evaluation.at("baseline_additions"))==baselineTotal);
}
} // namespace fgm::two_aux_report
