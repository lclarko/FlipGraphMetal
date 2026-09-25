// Host-only pinned arithmetic adapter. Header instrumentation is receipted by scalar_bridge.py.
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
static uint64_t reference_draws = 0, reference_appends = 0;
static std::vector<uint32_t> reference_words;
#include "core.h"
#include "addition.h"
#include "flip_set.h"
#include "scheme_integer.h"
#include "scheme_z2.h"

static uint64_t number(const std::string &s, uint64_t maximum) {
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("invalid unsigned argument");
    auto n = std::stoull(s);
    if (n > maximum) throw std::runtime_error("argument out of range");
    return n;
}
static int readInt(int lo, int hi) {
    long long value;
    if (!(std::cin >> value) || value < lo || value > hi) throw std::runtime_error("invalid input integer");
    return int(value);
}
template<class S> static int coefficient(const S &s, int p, int r, int c) {
    if constexpr (std::is_same_v<S, SchemeInteger>) return s.uvw[p][r][c];
    else return int((s.uvw[p][r] >> c) & 1);
}
template<class S> static bool overflow(const S &s) {
    return s.flips[0].overflow || s.flips[1].overflow || s.flips[2].overflow;
}
template<class S> static S input() {
    S s;
    for (int p=0;p<3;p++) s.n[p]=readInt(1,16);
    s.m=readInt(1,MAX_RANK);
    for (int p=0;p<3;p++) {
        s.nn[p]=s.n[p]*s.n[(p+1)%3];
        if(s.nn[p]>64) throw std::runtime_error("factor too wide");
        for(int r=0;r<s.m;r++) {
            if constexpr(std::is_same_v<S,SchemeInteger>) s.uvw[p][r]=Addition(s.nn[p]);
            else s.uvw[p][r]=0;
            for(int c=0;c<s.nn[p];c++) {
                int v=readInt(std::is_same_v<S,SchemeInteger> ? -1:0,1);
                if constexpr(std::is_same_v<S,SchemeInteger>) s.uvw[p][r].set(c,v);
                else s.uvw[p][r] |= T(v)<<c;
            }
        }
    }
    const int supplied=readInt(0,1);
    if(!supplied) s.initFlips();
    else for(int p=0;p<3;p++) {
        s.flips[p].overflow=readInt(0,1);
        s.flips[p].size=readInt(0,MAX_PAIRS);
        std::vector<uint32_t> seen;
        for(unsigned i=0;i<s.flips[p].size;i++) {
            int a=readInt(0,s.m-1), b=readInt(0,s.m-1);
            if(a==b || !(s.uvw[p][a]==s.uvw[p][b])) throw std::runtime_error("invalid supplied candidate");
            uint32_t pair=(uint32_t(a)<<16)|uint32_t(b);
            uint32_t identity=(uint32_t(std::min(a,b))<<16)|uint32_t(std::max(a,b));
            if(std::find(seen.begin(),seen.end(),identity)!=seen.end()) throw std::runtime_error("duplicate candidate");
            seen.push_back(identity); s.flips[p].pairs[i]=pair;
        }
        if(!s.flips[p].overflow) {
            size_t expected=0;
            for(int a=0;a<s.m;a++) for(int b=a+1;b<s.m;b++)
                if(s.uvw[p][a]==s.uvw[p][b]) expected++;
            if(expected!=s.flips[p].size) throw std::runtime_error("incomplete supplied candidates");
        }
    }
    std::cin >> std::ws;
    if(!std::cin.eof()) throw std::runtime_error("trailing input");
    if(!s.validate()) throw std::runtime_error("input tensor invalid");
    return s;
}
template<class S> static std::string proposal(S &s, RandomState &state, const std::string &op) {
    S candidate=s;
    int a,b,i,j,k,permutation[3];
    const uint64_t before=reference_appends;
    if(op=="plus") {
        a=randomWord(&state)%s.m; b=randomWord(&state)%s.m;
        if(a==b || s.uvw[0][a]==s.uvw[0][b] || s.uvw[1][a]==s.uvw[1][b] || s.uvw[2][a]==s.uvw[2][b]) return "tuple_rejection";
        randomPermutation(permutation,3,state);
        candidate.plus(permutation[0],permutation[1],permutation[2],a,b,randomWord(&state)%3);
        if(reference_appends==before) return "coefficient_rejection";
    } else if(op=="random") {
        a=randomWord(&state)%s.m;
        randomPermutation(permutation,3,state); i=permutation[0]; j=permutation[1]; k=permutation[2];
        if constexpr(std::is_same_v<S,SchemeInteger>) {
            Addition value(s.nn[i]); value.random(state);
            if(!s.uvw[i][a].limitSub(value,i!=2)) return "coefficient_rejection";
            candidate.split(i,j,k,a,value);
        } else {
            T value=randomWord(&state);
            if(s.nn[i]>32) value |= T(randomWord(&state))<<32;
            if(s.nn[i]<64) value &= (T(1)<<s.nn[i])-1;
            if(value==s.uvw[i][a]) return "coefficient_rejection";
            candidate.split(i,j,k,a,value);
        }
    } else if(op=="existing") {
        if constexpr(std::is_same_v<S,SchemeInteger>) {
            a=randomWord(&state)%s.m; b=randomWord(&state)%s.m; i=randomWord(&state)%3;
            j=(i+1)%3; k=(i+2)%3;
        } else {
            randomPermutation(permutation,3,state); i=permutation[0]; j=permutation[1]; k=permutation[2];
            a=randomWord(&state)%s.m; b=randomWord(&state)%s.m;
        }
        if(a==b || s.uvw[i][a]==s.uvw[i][b]) return "tuple_rejection";
        if constexpr(std::is_same_v<S,SchemeInteger>)
            if(!s.uvw[i][a].limitSub(s.uvw[i][b],i!=2)) return "coefficient_rejection";
        candidate.split(i,j,k,a,s.uvw[i][b]);
    } else throw std::runtime_error("unknown operation");
    s=candidate;
    return "applied";
}
template<class S> static void output(const S &s, const RandomState &state, const std::string &outcome, int removedTerms, std::ostream &out=std::cout, size_t wordStart=0, const std::string &observations="") {
    out << "{\"outcome\":\""<<outcome<<"\",\"rng\":"<<state.value<<",\"draws\":"<<(reference_words.size()-wordStart)<<",\"words\":[";
    for(size_t i=wordStart;i<reference_words.size();i++) out<<(i>wordStart?",":"")<<reference_words[i];
    out<<"],\"removed_terms\":";
    if(removedTerms<0) out<<"null"; else out<<removedTerms;
    out<<",\"reduction_operations\":null,\"scheme\":{\"n\":["<<s.n[0]<<","<<s.n[1]<<","<<s.n[2]<<"],\"m\":"<<s.m<<",\"z2\":"<<(std::is_same_v<S,SchemeInteger>?"false":"true");
    const char* names[]={"u","v","w"};
    for(int p=0;p<3;p++) {
        out<<",\""<<names[p]<<"\":[";
        for(int r=0;r<s.m;r++) { out<<(r?",":"")<<"[";
            for(int c=0;c<s.nn[p];c++) out<<(c?",":"")<<coefficient(s,p,r,c);
            out<<"]";
        } out<<"]";
    }
    out<<"},\"candidates\":[";
    for(int p=0;p<3;p++) {
        out<<(p?",":"")<<"{\"overflow\":"<<s.flips[p].overflow<<",\"pairs\":[";
        for(unsigned q=0;q<s.flips[p].size;q++)
            out<<(q?",":"")<<"["<<s.flips[p].index1(q)<<","<<s.flips[p].index2(q)<<"]";
        out<<"]}";
    }
    out<<"]";
    if(!observations.empty()) out<<",\"observations\":["<<observations<<"]";
    out<<"}\n";
}
template<class S> static void run(const std::string &op, uint32_t rng, unsigned ceiling, uint32_t limit, bool bounded) {
    S s=input<S>(); const int initialRank=s.m; RandomState state{rng}; std::string outcome;
    std::ostringstream observations;
    const bool expansion=op=="plus" || op=="random" || op=="existing";
    const uint32_t attempts=expansion ? limit : 1;
    for(uint32_t attempt=0; attempt<attempts; ++attempt) {
        const uint32_t before=state.value;
        const size_t wordStart=reference_words.size();
        const int rankBefore=s.m;
        if(overflow(s)) outcome="capacity_error";
        else if(op=="inspect") outcome="inspected";
        else if(op=="flip") outcome=s.tryFlip(state)?"applied":"unsuccessful";
        else if(op=="reduce") outcome=s.tryReduce()?"applied":"unsuccessful";
        else if(!expansion) throw std::runtime_error("unknown operation");
        else if(uint64_t(s.m)+1>std::min<unsigned>({ceiling,unsigned(MAX_RANK),unsigned(s.n[0]*s.n[1]*s.n[2])})) outcome="rank_blocked";
        else outcome=proposal(s,state,op);
        if(overflow(s)) outcome="capacity_error";
        if(!s.validate()) throw std::runtime_error("result tensor invalid");
        if(bounded) {
            observations<<(attempt ? "," : "")<<"{\"operation\":\""<<op<<"\",\"rng_before\":"<<before<<",\"result\":";
            output(s,state,outcome,(op=="flip" || op=="reduce") ? rankBefore-s.m : -1,
                   observations,wordStart);
            observations<<"}";
        }
        if(!bounded || (outcome!="tuple_rejection" && outcome!="coefficient_rejection")) break;
        if(attempt+1==attempts) outcome="proposal_exhausted";
    }
    output(s,state,outcome,(op=="flip" || op=="reduce") ? initialRank-s.m : -1,
           std::cout,0,observations.str());
}
int main(int argc,char **argv) {
    try {
        if(argc!=5 && argc!=6) throw std::runtime_error("usage: scalar_reference ZT|F2 inspect|flip|reduce|plus|random|existing rng ceiling [proposal_limit]");
        uint32_t rng=number(argv[3],UINT32_MAX); if(!rng) throw std::runtime_error("zero RNG state");
        unsigned ceiling=number(argv[4],MAX_RANK);
        uint32_t limit=argc==6 ? number(argv[5],UINT32_MAX) : 1;
        if(!limit) throw std::runtime_error("positive proposal limit required");
        if(std::string(argv[1])=="ZT") run<SchemeInteger>(argv[2],rng,ceiling,limit,argc==6);
        else if(std::string(argv[1])=="F2") run<SchemeZ2>(argv[2],rng,ceiling,limit,argc==6);
        else throw std::runtime_error("unsupported domain");
    } catch(const std::exception &e) { std::cerr<<e.what()<<"\n"; return 1; }
}
