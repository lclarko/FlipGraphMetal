#pragma once

// Shared bounded representation. Production closure is dispatched on Metal;
// the host prepares relations and checks returned witnesses.
namespace fgm_constructor {
using PairReducer=AdditionsReducer<23,184,9,828>;
constant constexpr int MaxInputs=9, MaxTargets=23, MaxDirections=33;
constant constexpr int MaxGates=24, MaxCandidates=992;
struct Gate { int out, left, right, leftSign, rightSign; };
struct Problem {
    int inputs, directions, outputs, operationCount, candidateCount;
    int vectors[MaxDirections-1][MaxInputs];
    int outputDirection[MaxTargets], outputSign[MaxTargets];
    Gate operations[MaxCandidates], candidates[MaxCandidates];
};
struct Witness {
    int status, count;
    uint64_t available;
    Gate gates[MaxGates];
};
inline bool append(LOCAL Witness &w,LOCAL const Gate &g) {
    const uint64_t bit=uint64_t(1)<<g.out;
    if(w.available&bit) return false;
    if(!(w.available&(uint64_t(1)<<g.left)) || !(w.available&(uint64_t(1)<<g.right))) return false;
    if(w.count>=MaxGates) { w.status=-1; return false; }
    w.gates[w.count++]=g; w.available|=bit; return true;
}
inline void close(GLOBAL const Problem &p,int candidate,LOCAL Witness &w) {
    w.status=0; w.count=0; w.available=0;
    if(p.inputs<1 || p.inputs>MaxInputs || p.directions<p.inputs || p.directions>32 ||
       p.outputs<0 || p.outputs>MaxTargets || p.operationCount<0 || p.operationCount>MaxCandidates ||
       p.candidateCount<0 || p.candidateCount>MaxCandidates || candidate< -1 || candidate>=p.candidateCount) {
        w.status=-1; return;
    }
    w.available=(uint64_t(1)<<p.inputs)-1;
    Gate extra[65]; int extraCount=0;
    if(candidate>=0) {
        Gate creation=p.candidates[candidate];
        if(creation.out<0) { w.status=2; return; }
        extra[extraCount++]=creation;
        int aux[MaxInputs];
        for(int c=0;c<p.inputs;++c)
            aux[c]=creation.leftSign*p.vectors[creation.left][c]+creation.rightSign*p.vectors[creation.right][c];
        for(int a=0;a<p.directions;++a) for(int sign=1;sign>=-1;sign-=2) {
            int sum[MaxInputs], first=0;
            for(int c=0;c<p.inputs;++c) { sum[c]=aux[c]+sign*p.vectors[a][c]; if(!first && sum[c]) first=sum[c]; }
            if(!first) continue;
            int normalize=first<0?-1:1;
            for(int t=p.inputs;t<p.directions;++t) {
                bool equal=true;
                for(int c=0;c<p.inputs;++c) equal &= normalize*sum[c]==p.vectors[t][c];
                if(equal) { extra[extraCount++]={t,p.directions,a,normalize,normalize*sign}; break; }
            }
        }
    }
    const uint64_t required=(uint64_t(1)<<p.directions)-1;
    bool changed=true;
    while(changed && w.status>=0 && (w.available&required)!=required) {
        changed=false;
        for(int i=0;i<p.operationCount;++i) {
            Gate g=p.operations[i]; changed=append(w,g)||changed;
        }
        for(int i=0;i<extraCount;++i) changed=append(w,extra[i])||changed;
    }
    if(w.status>=0 && (w.available&required)==required) w.status=1;
}

#ifndef __METAL_VERSION__
// Bounds allow small nonternary test maps as well as production ternary maps.
// Two successive signed sums remain representable in int.
inline Problem prepare(const std::vector<std::vector<int64_t>> &targets,int inputs) {
    if(inputs<1 || inputs>MaxInputs || targets.size()>MaxTargets) throw std::runtime_error("constructor map capacity exceeded");
    Problem p{}; p.inputs=inputs; p.directions=inputs; p.outputs=int(targets.size());
    for(int i=0;i<inputs;++i) p.vectors[i][i]=1;
    for(int i=0;i<p.outputs;++i) {
        if(targets[i].size()!=size_t(inputs)) throw std::runtime_error("constructor target width mismatch");
        int sign=0;
        for(auto x:targets[i]) { if(x>INT32_MAX/4 || x< -INT32_MAX/4) throw std::runtime_error("constructor coefficient bound exceeded"); if(!sign && x) sign=x<0?-1:1; }
        p.outputSign[i]=sign; p.outputDirection[i]=-1;
        if(!sign) continue;
        int direction=0;
        for(;direction<p.directions;++direction) {
            bool equal=true;
            for(int c=0;c<inputs;++c) equal &= p.vectors[direction][c]==sign*targets[i][c];
            if(equal) break;
        }
        if(direction==p.directions) {
            if(p.directions>=MaxDirections-1) throw std::runtime_error("constructor direction capacity exceeded");
            for(int c=0;c<inputs;++c) p.vectors[direction][c]=int(sign*targets[i][c]);
            ++p.directions;
        }
        p.outputDirection[i]=direction;
    }
    for(int a=0;a<p.directions;++a) for(int b=a+1;b<p.directions;++b) for(int sign=1;sign>=-1;sign-=2) {
        int sum[MaxInputs]{}, first=0;
        for(int c=0;c<inputs;++c) { sum[c]=p.vectors[a][c]+sign*p.vectors[b][c]; if(!first && sum[c]) first=sum[c]; }
        int normalize=first<0?-1:1, match=-1;
        if(first) for(int t=0;t<p.directions;++t) {
            bool equal=true;
            for(int c=0;c<inputs;++c) equal &= normalize*sum[c]==p.vectors[t][c];
            if(equal) { match=t; break; }
        }
        if(p.candidateCount>=MaxCandidates) throw std::runtime_error("constructor candidate capacity exceeded");
        p.candidates[p.candidateCount++]={first && match<0?p.directions:-1,a,b,normalize,normalize*sign};
        if(match>=inputs) {
            if(p.operationCount>=MaxCandidates) throw std::runtime_error("constructor operation capacity exceeded");
            p.operations[p.operationCount++]={match,a,b,normalize,normalize*sign};
        }
    }
    return p;
}
#endif
}
