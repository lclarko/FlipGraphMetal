#pragma once
#if defined(__METAL_VERSION__) && !defined(METAL_F2)
// Deep copies retain each lane's allocated storage and preserve candidate order.
// CompactScheme's implicit assignment would alias device pointers instead.
struct ControlledPackedScheme {
    CompactScheme arithmetic;
    int n[3],nn[3],m;
    CompactFlipSet flips[3];
    device CompactStoredAddition &term(int p,int r) const thread { return arithmetic.term(p,r); }
    void before() thread {
        arithmetic.m=m;
        for(int p=0;p<3;++p) { arithmetic.n[p]=n[p]; arithmetic.nn[p]=nn[p]; arithmetic.flips[p]=flips[p]; }
    }
    void after() thread {
        m=arithmetic.m;
        for(int p=0;p<3;++p) flips[p]=arithmetic.flips[p];
    }
    bool tryFlip(thread RandomState &rng,bool reduce) thread { before(); bool result=arithmetic.tryFlip(rng,reduce); after(); return result; }
    bool tryReduce() thread { before(); bool result=arithmetic.tryReduce(); after(); return result; }
    void plus(int i,int j,int k,int a,int b,int variant) thread { before(); arithmetic.plus(i,j,k,a,b,variant); after(); }
    void split(int i,int j,int k,int a,CompactAddition value) thread { before(); arithmetic.split(i,j,k,a,value); after(); }

    thread ControlledPackedScheme &operator=(thread const ControlledPackedScheme &source) thread {
        m=source.m;
        for(int p=0;p<3;++p) {
            n[p]=source.n[p]; nn[p]=source.nn[p];
            for(int r=0;r<m;++r) term(p,r)=source.term(p,r);
            flips[p].size=source.flips[p].size; flips[p].overflow=source.flips[p].overflow;
            for(uint32_t i=0;i<flips[p].size;++i) flips[p].pairs[i*32]=source.flips[p].pairs[i*32];
        }
        return *this;
    }
    void bind(device CompactStoredAddition *termStorage,device uint32_t *pairStorage,device int *work,uint index) thread {
        uint64_t tile=index/32,lane=index%32;
        arithmetic.terms=termStorage+tile*(MAX_RANK*3*32)+lane;
        arithmetic.scratch=work?work+tile*(MAX_PAIRS*3*32)+lane:nullptr;
        for(int p=0;p<3;++p) flips[p].pairs=pairStorage+tile*(MAX_PAIRS*3*32)+p*MAX_PAIRS*32+lane;
    }
    uint32_t load(device const SchemeInteger &source) thread {
        if(source.m<1 || source.m>MAX_RANK) return 5;
        for(int p=0;p<3;++p) {
            if(source.n[p]!=3 || source.nn[p]!=9 || source.flips[p].size>MAX_PAIRS) return 5;
            if(source.flips[p].overflow) return 4;
            for(int r=0;r<source.m;++r) {
                device const Addition &value=source.uvw[p][r];
                if(value.n!=9 || !value.valid || (value.values&~T(511)) || (value.signs&~T(511))) return 5;
            }
        }
        m=source.m;
        for(int p=0;p<3;++p) {
            n[p]=3; nn[p]=9;
            for(int r=0;r<m;++r) {
                CompactAddition value(9);
                value.values=ushort(source.uvw[p][r].values); value.signs=ushort(source.uvw[p][r].signs); value.valid=source.uvw[p][r].valid;
                term(p,r)=value;
            }
            flips[p].size=source.flips[p].size; flips[p].overflow=source.flips[p].overflow;
            for(uint32_t i=0;i<flips[p].size;++i) flips[p].pairs[i*32]=source.flips[p].pairs[i];
        }
        return 0;
    }
    void store(device SchemeInteger &target) const thread {
        target.m=m;
        for(int p=0;p<3;++p) {
            target.n[p]=n[p]; target.nn[p]=nn[p];
            for(int r=0;r<m;++r) {
                CompactAddition value=term(p,r);
                target.uvw[p][r].n=9; target.uvw[p][r].values=value.values;
                target.uvw[p][r].signs=value.signs; target.uvw[p][r].valid=value.valid;
            }
            target.flips[p].size=flips[p].size; target.flips[p].overflow=flips[p].overflow;
            for(uint32_t i=0;i<flips[p].size;++i) target.flips[p].pairs[i]=flips[p].pairs[i*32];
        }
    }
};
// Only representation-specific proposal admission differs. All control ordering,
// limits, observations, budgets and RNG decisions use controlled.h unchanged.
template<> inline ControlledOutcome ControlledArithmetic::proposal<ControlledPackedScheme>(thread ControlledPackedScheme &s,thread RandomState &rng,uint32_t op) {
    if(op==2) {
        int a=randomWord(&rng)%s.m,b=randomWord(&rng)%s.m,i=randomWord(&rng)%3;
        if(a==b || s.term(i,a)==s.term(i,b)) return ControlledOutcome::TupleRejection;
        if(!s.term(i,a).limitSub(s.term(i,b),i!=2)) return ControlledOutcome::CoefficientRejection;
        s.split(i,(i+1)%3,(i+2)%3,a,s.term(i,b)); return ControlledOutcome::Applied;
    }
    int a=randomWord(&rng)%s.m;
    if(op==0) {
        int b=randomWord(&rng)%s.m;
        if(a==b || s.term(0,a)==s.term(0,b) || s.term(1,a)==s.term(1,b) || s.term(2,a)==s.term(2,b)) return ControlledOutcome::TupleRejection;
        int p[3]; randomPermutation(p,3,rng); int variant=randomWord(&rng)%3;
        int i=p[0],j=p[1],k=p[2];
        CompactAddition aAdd=s.term(i,a)+s.term(i,b),bAdd=s.term(j,a)+s.term(j,b),cAdd=s.term(k,a)+s.term(k,b);
        CompactAddition aSub=s.term(i,b)-s.term(i,a),bSub=s.term(j,b)-s.term(j,a),cSub=s.term(k,b)-s.term(k,a);
        bool admissible=(variant==0 && aSub.limit(i!=2) && bAdd.limit(j!=2) && cSub.limit(k!=2)) ||
            (variant==1 && aSub.limit(i!=2) && bSub.limit(j!=2) && cAdd.limit(k!=2)) ||
            (aAdd.limit(i!=2) && bSub.limit(j!=2) && cSub.limit(k!=2));
        if(!admissible) return ControlledOutcome::CoefficientRejection;
        s.plus(i,j,k,a,b,variant); return ControlledOutcome::Applied;
    }
    int p[3]; randomPermutation(p,3,rng);
    CompactAddition value(9); value.random(rng);
    if(!s.term(p[0],a).limitSub(value,p[0]!=2)) return ControlledOutcome::CoefficientRejection;
    s.split(p[0],p[1],p[2],a,value); return ControlledOutcome::Applied;
}
struct ControlledPackedSink {
    device SchemeInteger *captures;
    device ControlledCaptureMeta *metadata;
    void capture(thread const ControlledPackedScheme &s,ControlledOperation operation,uint64_t control,bool mandatory,bool optional,uint64_t index) thread {
        if(mandatory) { s.store(captures[0]); metadata[0]={control,uint32_t(operation),0}; }
        if(optional) { s.store(captures[index+1]); metadata[index+1]={control,uint32_t(operation),0}; }
    }
};
#endif
