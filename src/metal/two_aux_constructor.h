#pragma once

#ifndef __METAL_VERSION__
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>
#endif

namespace fgm_constructor {

constant constexpr int TwoAuxMaxDirections=34;
constant constexpr int TwoAuxMaxGates=25;
constant constexpr int TwoAuxMaxExtraRules=130;
constant constexpr int TwoAuxMaxRules=1124;
constant constexpr uint64_t TwoAuxMaxPairSlots=1047552;

struct TwoAuxProblem {
    int inputs, directions, outputs, operationCount;
    int vectors[32][9];
    int outputDirection[23], outputSign[23];
    Gate operations[992];
};

struct TwoAuxTask {
    uint64_t rawSlot;
    int helpers[2][9];
    Gate creations[2];
    int operationCount;
    Gate operations[130];
};

struct TwoAuxWitness {
    int status, count;
    uint64_t available, ruleChecks;
    int sweeps;
    Gate gates[25];
};

// Shared-buffer ABI. Compile these checks in both the host and Metal sources.
static_assert(sizeof(Gate)==20,"two-aux gate ABI");
static_assert(__builtin_offsetof(TwoAuxProblem,inputs)==0 &&
              __builtin_offsetof(TwoAuxProblem,directions)==4 &&
              __builtin_offsetof(TwoAuxProblem,outputs)==8 &&
              __builtin_offsetof(TwoAuxProblem,operationCount)==12 &&
              __builtin_offsetof(TwoAuxProblem,vectors)==16 &&
              __builtin_offsetof(TwoAuxProblem,outputDirection)==1168 &&
              __builtin_offsetof(TwoAuxProblem,outputSign)==1260 &&
              __builtin_offsetof(TwoAuxProblem,operations)==1352 && sizeof(TwoAuxProblem)==21192,
              "two-aux problem ABI");
static_assert(__builtin_offsetof(TwoAuxTask,rawSlot)==0 &&
              __builtin_offsetof(TwoAuxTask,helpers)==8 &&
              __builtin_offsetof(TwoAuxTask,creations)==80 &&
              __builtin_offsetof(TwoAuxTask,operationCount)==120 &&
              __builtin_offsetof(TwoAuxTask,operations)==124 && sizeof(TwoAuxTask)==2728,
              "two-aux task ABI");
static_assert(__builtin_offsetof(TwoAuxWitness,status)==0 &&
              __builtin_offsetof(TwoAuxWitness,count)==4 &&
              __builtin_offsetof(TwoAuxWitness,available)==8 &&
              __builtin_offsetof(TwoAuxWitness,ruleChecks)==16 &&
              __builtin_offsetof(TwoAuxWitness,sweeps)==24 &&
              __builtin_offsetof(TwoAuxWitness,gates)==28 && sizeof(TwoAuxWitness)==528,
              "two-aux witness ABI");

#ifdef __METAL_VERSION__

inline bool twoAuxApply(LOCAL TwoAuxWitness &w,LOCAL const Gate &g,int inputs,int directions,
                        LOCAL bool &changed) {
    if(g.out<inputs || g.out>=directions+2 || g.left<0 || g.left>=directions+2 ||
       g.right<0 || g.right>=directions+2 || g.left>=g.right) {
        w.status=-1; return false;
    }
    ++w.ruleChecks;
    const uint64_t result=uint64_t(1)<<g.out;
    if((w.available&result) || !(w.available&(uint64_t(1)<<g.left)) ||
       !(w.available&(uint64_t(1)<<g.right))) return true;
    if(w.count>=TwoAuxMaxGates) { w.status=-1; return false; }
    w.gates[w.count++]=g;
    w.available|=result;
    changed=true;
    return true;
}

// The host supplies all permitted rules. This device closure only tracks
// reachability and records the first applicable gate in fixed scan order.
inline void closeTwoAux(GLOBAL const TwoAuxProblem &p,GLOBAL const TwoAuxTask &task,
                        LOCAL TwoAuxWitness &w) {
    w.status=0; w.count=0; w.available=0; w.ruleChecks=0; w.sweeps=0;
    if(p.inputs<1 || p.inputs>9 || p.directions<p.inputs || p.directions>32 ||
       p.outputs<0 || p.outputs>23 || p.operationCount<0 || p.operationCount>992 ||
       task.operationCount<0 || task.operationCount>TwoAuxMaxExtraRules) {
        w.status=-1; return;
    }
    w.available=(uint64_t(1)<<p.inputs)-1;
    const uint64_t required=(uint64_t(1)<<p.directions)-1;
    bool changed=true;
    while(changed && (w.available&required)!=required) {
        if(w.sweeps>=TwoAuxMaxGates+1) { w.status=-1; return; }
        changed=false; ++w.sweeps;
        for(int i=0;i<p.operationCount;++i) {
            Gate gate=p.operations[i];
            if(!twoAuxApply(w,gate,p.inputs,p.directions,changed)) return;
        }
        for(int i=0;i<task.operationCount;++i) {
            Gate gate=task.operations[i];
            if(!twoAuxApply(w,gate,p.inputs,p.directions,changed)) return;
        }
        for(int i=0;i<2;++i) {
            Gate gate=task.creations[i];
            if(!twoAuxApply(w,gate,p.inputs,p.directions,changed)) return;
        }
    }
    w.status=(w.available&required)==required?1:0;
}

#endif

#ifndef __METAL_VERSION__

static_assert(std::is_standard_layout_v<TwoAuxProblem> && std::is_trivial_v<TwoAuxProblem>);
static_assert(std::is_standard_layout_v<TwoAuxTask> && std::is_trivial_v<TwoAuxTask>);
static_assert(std::is_standard_layout_v<TwoAuxWitness> && std::is_trivial_v<TwoAuxWitness>);
static_assert(sizeof(TwoAuxProblem::vectors)/sizeof(int)==32*9);
static_assert(sizeof(TwoAuxProblem::operations)/sizeof(Gate)==992);
static_assert(sizeof(TwoAuxTask::operations)/sizeof(Gate)==TwoAuxMaxExtraRules);
static_assert(sizeof(TwoAuxWitness::gates)/sizeof(Gate)==TwoAuxMaxGates);
static_assert(32*31*33*32==TwoAuxMaxPairSlots);
static_assert(992+TwoAuxMaxExtraRules+2==TwoAuxMaxRules);
enum class TwoAuxFilter { None, FirstInvalid, SecondInvalid, Symmetry };

struct TwoAuxStats {
    uint64_t rawScanned=0, firstInvalid=0, secondInvalid=0, symmetryFiltered=0, prepared=0;
};

// Pair IDs enumerate left < right in lexicographic order, with + before -.
inline void twoAuxDecodeRoute(uint64_t route,int directions,int &left,int &right,int &sign) {
    sign=(route&1)?-1:1;
    route/=2;
    for(left=0;left<directions-1;++left) {
        const uint64_t count=uint64_t(directions-left-1);
        if(route<count) { right=left+1+int(route); return; }
        route-=count;
    }
    throw std::runtime_error("two-aux route out of range");
}

inline bool twoAuxMatches(const int64_t *normalized,const TwoAuxProblem &p,int &direction) {
    for(int t=0;t<p.directions;++t) {
        bool equal=true;
        for(int c=0;c<p.inputs;++c) equal &= normalized[c]==p.vectors[t][c];
        if(equal) { direction=t; return true; }
    }
    return false;
}

// Return false for the zero vector; all arithmetic remains int64 until a
// normalized helper has passed the int32 storage bound.
inline bool twoAuxCombine(const int *left,const int *right,int sign,int inputs,
                          int64_t *normalized,int &normalization) {
    int64_t first=0;
    for(int c=0;c<inputs;++c) {
        normalized[c]=int64_t(left[c])+int64_t(sign)*right[c];
        if(!first && normalized[c]) first=normalized[c];
    }
    if(!first) return false;
    normalization=first<0?-1:1;
    for(int c=0;c<inputs;++c) normalized[c]*=normalization;
    return true;
}

inline uint64_t twoAuxRawTotal(const TwoAuxProblem &p) {
    if(p.inputs<1 || p.inputs>9 || p.directions<p.inputs || p.directions>32 ||
       p.outputs<0 || p.outputs>23 || p.operationCount<0 || p.operationCount>992)
        throw std::runtime_error("two-aux problem capacity exceeded");
    const uint64_t n=uint64_t(p.directions);
    return n*(n-1)*n*(n+1);
}

inline TwoAuxProblem prepareTwoAux(const std::vector<std::vector<int64_t>> &targets,int inputs) {
    // The existing preparation fixes first-seen canonical directions and the
    // base target rules. Its coefficient bound also bounds both helper levels.
    const Problem base=prepare(targets,inputs);
    TwoAuxProblem p{};
    p.inputs=base.inputs; p.directions=base.directions; p.outputs=base.outputs;
    p.operationCount=base.operationCount;
    for(int t=0;t<p.directions;++t)
        for(int c=0;c<p.inputs;++c) p.vectors[t][c]=base.vectors[t][c];
    for(int t=0;t<p.outputs;++t) {
        p.outputDirection[t]=base.outputDirection[t];
        p.outputSign[t]=base.outputSign[t];
    }
    for(int i=0;i<p.operationCount;++i) p.operations[i]=base.operations[i];
    return p;
}

inline bool prepareTwoAuxTask(const TwoAuxProblem &p,uint64_t rawSlot,
                              TwoAuxTask &task,TwoAuxFilter &filter) {
    const uint64_t total=twoAuxRawTotal(p);
    if(rawSlot>=total) throw std::runtime_error("two-aux raw slot out of range");
    const int n=p.directions;
    const uint64_t secondRoutes=uint64_t(n)*(n+1);
    const uint64_t r1=rawSlot/secondRoutes, r2=rawSlot%secondRoutes;
    int a,b,sign,normalization,match;
    int64_t value[9]{};
    twoAuxDecodeRoute(r1,n,a,b,sign);
    if(!twoAuxCombine(p.vectors[a],p.vectors[b],sign,p.inputs,value,normalization) ||
       twoAuxMatches(value,p,match)) {
        filter=TwoAuxFilter::FirstInvalid;
        return false;
    }
    int helpers[2][9]{};
    for(int c=0;c<p.inputs;++c) {
        if(value[c]>INT32_MAX || value[c]<INT32_MIN)
            throw std::runtime_error("two-aux first helper coefficient exceeded int32");
        helpers[0][c]=int(value[c]);
    }
    const Gate creation1{n,a,b,normalization,normalization*sign};

    int x,y,secondSign;
    twoAuxDecodeRoute(r2,n+1,x,y,secondSign);
    const int *left=x==n?helpers[0]:p.vectors[x];
    const int *right=y==n?helpers[0]:p.vectors[y];
    if(!twoAuxCombine(left,right,secondSign,p.inputs,value,normalization) ||
       twoAuxMatches(value,p,match)) {
        filter=TwoAuxFilter::SecondInvalid;
        return false;
    }
    bool sameFirst=true;
    for(int c=0;c<p.inputs;++c) sameFirst &= value[c]==helpers[0][c];
    if(sameFirst) { filter=TwoAuxFilter::SecondInvalid; return false; }
    // The enlarged pair space inserts one (x, h1) pair after each D row.
    // Remove those 2*x signed slots to compare two D-only route IDs.
    if(y<n && r1>=r2-2*uint64_t(x)) {
        filter=TwoAuxFilter::Symmetry;
        return false;
    }
    for(int c=0;c<p.inputs;++c) {
        if(value[c]>INT32_MAX || value[c]<INT32_MIN)
            throw std::runtime_error("two-aux second helper coefficient exceeded int32");
        helpers[1][c]=int(value[c]);
    }
    const Gate creation2{n+1,x,y,normalization,normalization*secondSign};

    task={}; task.rawSlot=rawSlot;
    for(int h=0;h<2;++h)
        for(int c=0;c<p.inputs;++c) task.helpers[h][c]=helpers[h][c];
    task.creations[0]=creation1; task.creations[1]=creation2;
    for(int l=0;l<n+2;++l) for(int r=l+1;r<n+2;++r) {
        if(r<n) continue;
        const int *lv=l==n?helpers[0]:(l==n+1?helpers[1]:p.vectors[l]);
        const int *rv=r==n?helpers[0]:(r==n+1?helpers[1]:p.vectors[r]);
        for(int s=1;s>=-1;s-=2) {
            if(!twoAuxCombine(lv,rv,s,p.inputs,value,normalization)) continue;
            for(int t=p.inputs;t<n;++t) {
                bool equal=true;
                for(int c=0;c<p.inputs;++c) equal &= value[c]==p.vectors[t][c];
                if(!equal) continue;
                if(task.operationCount>=TwoAuxMaxExtraRules)
                    throw std::runtime_error("two-aux extra rule capacity exceeded");
                task.operations[task.operationCount++]={t,l,r,normalization,normalization*s};
                break;
            }
        }
    }
    filter=TwoAuxFilter::None;
    return true;
}

class TwoAuxStream {
    const TwoAuxProblem &problem_;
    uint64_t total_, limit_, next_=0;
    TwoAuxStats stats_{};
public:
    TwoAuxStream(const TwoAuxProblem &p,uint64_t budget):problem_(p),total_(twoAuxRawTotal(p)),limit_(0) {
        if(!budget || budget>TwoAuxMaxPairSlots)
            throw std::runtime_error("two-aux slot budget out of range");
        limit_=std::min(budget,total_);
    }
    bool next(TwoAuxTask &task) {
        while(next_<limit_) {
            const uint64_t slot=next_++;
            ++stats_.rawScanned;
            TwoAuxFilter filter=TwoAuxFilter::None;
            if(prepareTwoAuxTask(problem_,slot,task,filter)) {
                ++stats_.prepared;
                return true;
            }
            switch(filter) {
            case TwoAuxFilter::FirstInvalid: {
                const uint64_t secondRoutes=uint64_t(problem_.directions)*(problem_.directions+1);
                const uint64_t skipped=std::min(secondRoutes-slot%secondRoutes,limit_-slot);
                stats_.firstInvalid+=skipped;
                stats_.rawScanned+=skipped-1;
                next_+=skipped-1;
                break;
            }
            case TwoAuxFilter::SecondInvalid: ++stats_.secondInvalid; break;
            case TwoAuxFilter::Symmetry: ++stats_.symmetryFiltered; break;
            case TwoAuxFilter::None: throw std::runtime_error("two-aux filter missing");
            }
        }
        return false;
    }
    const TwoAuxStats &stats() const { return stats_; }
    uint64_t rawTotal() const { return total_; }
    uint64_t limit() const { return limit_; }
};

#endif
}
