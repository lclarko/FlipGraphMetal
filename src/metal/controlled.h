#pragma once
// Included after the domain arithmetic headers. No host library or allocation.
enum class ControlledOutcome : uint32_t {
    Active, Applied, UnsuccessfulFlip, TupleRejection, CoefficientRejection,
    ProposalExhausted, RankBlocked, CapacityError, RestartRequested,
    TargetPending, ExistingTargetPending, TargetMet, BudgetExhausted, InvalidState
};
enum class ControlledOperation : uint32_t { Flip, Reduction, Expansion };
struct ControlledSettings {
    uint64_t anchor, ceiling, intervalMin, intervalMax;
    uint64_t flipBudget, controlBudget, stagnationLimit, optionalQuota, proposalLimit;
    uint64_t targetRank;
    uint32_t reductionQ, alternatives, targetEnabled;
};
struct ControlledState {
    RandomState rng;
    uint32_t initialized, pendingRestart, mandatoryValid, mandatoryCommitted;
    uint32_t mandatoryRank;
    ControlledOutcome terminal, lastOutcome;
    uint64_t countdown, stagnation, flips, controls;
    uint64_t attempted[3], applied[3];
    uint64_t removed[3]; // flip, explicit reduction, total
    uint64_t proposals, tupleRejections, coefficientRejections, exhaustedProposals, blocked;
    uint64_t optionalCount, optionalEncounters, optionalDrops;
};
inline bool controlledTerminal(LOCAL const ControlledState &s) { return s.terminal!=ControlledOutcome::Active; }
inline bool controlledTarget(LOCAL const ControlledSettings &c,int rank) { return c.targetEnabled && uint64_t(rank)<=c.targetRank; }
inline bool controlledBudget(LOCAL const ControlledSettings &c,LOCAL const ControlledState &s) {
    return s.flips>=c.flipBudget || s.controls>=c.controlBudget;
}
inline void controlledFail(LOCAL ControlledState &s,ControlledOutcome outcome) { s.terminal=outcome; s.lastOutcome=outcome; }
inline bool controlledAdd(LOCAL ControlledState &s,LOCAL uint64_t &value,uint64_t amount) {
    if(value>~uint64_t(0)-amount) { controlledFail(s,ControlledOutcome::InvalidState); return false; }
    value+=amount; return true;
}
inline bool controlledIncrement(LOCAL ControlledState &s,LOCAL uint64_t &value) { return controlledAdd(s,value,1); }
inline void controlledRedraw(LOCAL const ControlledSettings &c,LOCAL ControlledState &s) {
    s.countdown=c.intervalMin+uint64_t(randomWord(&s.rng))%(c.intervalMax-c.intervalMin+1);
}
// Friend access is confined to bounded proposals. Pinned arithmetic stays intact.
struct ControlledArithmetic {
    static bool plusAdmissible(LOCAL const SchemeInteger &s,int i,int j,int k,int a,int b,int variant) {
        Addition aAdd=s.uvw[i][a]+s.uvw[i][b], bAdd=s.uvw[j][a]+s.uvw[j][b], cAdd=s.uvw[k][a]+s.uvw[k][b];
        Addition aSub=s.uvw[i][b]-s.uvw[i][a], bSub=s.uvw[j][b]-s.uvw[j][a], cSub=s.uvw[k][b]-s.uvw[k][a];
        if(variant==0 && aSub.limit(i!=2) && bAdd.limit(j!=2) && cSub.limit(k!=2)) return true;
        if(variant==1 && aSub.limit(i!=2) && bSub.limit(j!=2) && cAdd.limit(k!=2)) return true;
        return aAdd.limit(i!=2) && bSub.limit(j!=2) && cSub.limit(k!=2);
    }
    static bool plusAdmissible(LOCAL const SchemeZ2 &,int,int,int,int,int,int) { return true; }
    static ControlledOutcome randomSplit(LOCAL SchemeInteger &s,LOCAL RandomState &rng,int a,int i,int j,int k) {
        Addition value(s.nn[i]); value.random(rng);
        if(!s.uvw[i][a].limitSub(value,i!=2)) return ControlledOutcome::CoefficientRejection;
        s.split(i,j,k,a,value); return ControlledOutcome::Applied;
    }
    static ControlledOutcome randomSplit(LOCAL SchemeZ2 &s,LOCAL RandomState &rng,int a,int i,int j,int k) {
        T value=randomWord(&rng);
        if(s.nn[i]>32) value|=T(randomWord(&rng))<<32;
        if(s.nn[i]<64) value&=(T(1)<<s.nn[i])-1;
        if(value==s.uvw[i][a]) return ControlledOutcome::CoefficientRejection;
        s.split(i,j,k,a,value); return ControlledOutcome::Applied;
    }
    static ControlledOutcome existingSplit(LOCAL SchemeInteger &s,LOCAL RandomState &rng) {
        int a=randomWord(&rng)%s.m,b=randomWord(&rng)%s.m,i=randomWord(&rng)%3;
        if(a==b || s.uvw[i][a]==s.uvw[i][b]) return ControlledOutcome::TupleRejection;
        if(!s.uvw[i][a].limitSub(s.uvw[i][b],i!=2)) return ControlledOutcome::CoefficientRejection;
        s.split(i,(i+1)%3,(i+2)%3,a,s.uvw[i][b]); return ControlledOutcome::Applied;
    }
    static ControlledOutcome existingSplit(LOCAL SchemeZ2 &s,LOCAL RandomState &rng) {
        int p[3]; randomPermutation(p,3,rng);
        int a=randomWord(&rng)%s.m,b=randomWord(&rng)%s.m;
        if(a==b || s.uvw[p[0]][a]==s.uvw[p[0]][b]) return ControlledOutcome::TupleRejection;
        s.split(p[0],p[1],p[2],a,s.uvw[p[0]][b]); return ControlledOutcome::Applied;
    }
    template<class S> static ControlledOutcome proposal(LOCAL S &s,LOCAL RandomState &rng,uint32_t op) {
        if(op==2) return existingSplit(s,rng);
        int a=randomWord(&rng)%s.m;
        if(op==0) {
            int b=randomWord(&rng)%s.m;
            if(a==b || s.uvw[0][a]==s.uvw[0][b] || s.uvw[1][a]==s.uvw[1][b] || s.uvw[2][a]==s.uvw[2][b]) return ControlledOutcome::TupleRejection;
            int p[3]; randomPermutation(p,3,rng); int variant=randomWord(&rng)%3;
            if(!plusAdmissible(s,p[0],p[1],p[2],a,b,variant)) return ControlledOutcome::CoefficientRejection;
            s.plus(p[0],p[1],p[2],a,b,variant); return ControlledOutcome::Applied;
        }
        int p[3]; randomPermutation(p,3,rng);
        return randomSplit(s,rng,a,p[0],p[1],p[2]);
    }
};
// The caller verifies the parent and supplies a checked ceiling before entry.
// Full assignment preserves candidate order; arithmetic copyTo rebuilds it.
template<class S> bool controlledInitialize(LOCAL const ControlledSettings &c,LOCAL ControlledState &s,
        LOCAL const S &parent,LOCAL S &current,LOCAL S &best,uint32_t seed,uint32_t worker) {
    s={}; s.mandatoryCommitted=1;
    if(!c.anchor || !c.ceiling || c.ceiling>MAX_RANK || !c.intervalMin || c.intervalMax<c.intervalMin ||
       !c.stagnationLimit || !c.proposalLimit || c.proposalLimit>uint64_t(0xffffffffu) || c.alternatives>1 || c.targetEnabled>1 ||
       parent.m<1 || parent.m>MAX_RANK || candidateOverflow(parent)) {
        controlledFail(s,ControlledOutcome::InvalidState); return false;
    }
    for(int p=0;p<3;++p) if(parent.n[p]<1 || parent.n[p]>16) {
        controlledFail(s,ControlledOutcome::InvalidState); return false;
    }
    for(int p=0;p<3;++p) if(parent.nn[p]!=parent.n[p]*parent.n[(p+1)%3] || parent.nn[p]>MAX_MATRIX_ELEMENTS) {
        controlledFail(s,ControlledOutcome::InvalidState); return false;
    }
    current=parent; best=parent;
    if(controlledTarget(c,parent.m)) { controlledFail(s,ControlledOutcome::ExistingTargetPending); return true; }
    s.rng.value=seed^(uint32_t(0x9e3779b9u)*uint32_t(worker+1u));
    if(!s.rng.value) s.rng.value=1;
    s.initialized=1; controlledRedraw(c,s); return true;
}
template<class S,class Best,class Sink> void controlledObserve(LOCAL const ControlledSettings &c,LOCAL ControlledState &s,
        LOCAL const S &current,LOCAL Best &best,ControlledOperation op,LOCAL Sink &sink) {
    if(candidateOverflow(current)) { controlledFail(s,ControlledOutcome::CapacityError); return; }
    if(current.m<best.m) { best=current; s.stagnation=0; }
    bool mandatory=!s.mandatoryValid || uint32_t(current.m)<s.mandatoryRank;
    if(mandatory) { s.mandatoryValid=1; s.mandatoryCommitted=0; s.mandatoryRank=uint32_t(current.m); }
    bool eligible=c.alternatives ? uint64_t(current.m)==c.anchor : uint64_t(current.m)<c.anchor;
    bool optional=false; uint64_t slot=s.optionalCount;
    if(eligible) {
        if(!controlledIncrement(s,s.optionalEncounters)) return;
        if(s.optionalCount<c.optionalQuota) { optional=true; ++s.optionalCount; }
        else if(!controlledIncrement(s,s.optionalDrops)) return;
    }
    if(mandatory || optional) sink.capture(current,op,s.controls,mandatory,optional,slot);
    if(controlledTarget(c,current.m)) controlledFail(s,ControlledOutcome::TargetPending);
}
template<class S> ControlledOutcome controlledPrimitive(LOCAL const ControlledSettings &c,LOCAL ControlledState &s,LOCAL S &current,uint32_t op) {
    if(candidateOverflow(current)) { controlledFail(s,ControlledOutcome::CapacityError); return s.terminal; }
    if(current.m<1 || current.m>MAX_RANK || c.ceiling>MAX_RANK || !c.proposalLimit || op>2) { controlledFail(s,ControlledOutcome::InvalidState); return s.terminal; }
    if(uint64_t(current.m)+1>c.ceiling) { controlledIncrement(s,s.blocked); return s.lastOutcome=ControlledOutcome::RankBlocked; }
    if(!controlledIncrement(s,s.attempted[2])) return s.terminal;
    for(uint64_t attempt=0;attempt<c.proposalLimit;++attempt) {
        if(!controlledIncrement(s,s.proposals)) return s.terminal;
        auto result=ControlledArithmetic::proposal(current,s.rng,op);
        if(result==ControlledOutcome::Applied) {
            if(candidateOverflow(current)) { controlledFail(s,ControlledOutcome::CapacityError); return s.terminal; }
            if(!controlledIncrement(s,s.applied[2])) return s.terminal;
            return s.lastOutcome=result;
        }
        if(!controlledIncrement(s,result==ControlledOutcome::TupleRejection?s.tupleRejections:s.coefficientRejections)) return s.terminal;
    }
    if(!controlledIncrement(s,s.exhaustedProposals)) return s.terminal;
    return s.lastOutcome=ControlledOutcome::ProposalExhausted;
}
template<class S,class Best,class Sink> void controlledExpansion(LOCAL const ControlledSettings &c,LOCAL ControlledState &s,
        LOCAL S &current,LOCAL Best &best,bool recovery,LOCAL Sink &sink) {
    uint32_t applied=0;
    if(uint64_t(current.m)+1>c.ceiling) { if(!controlledIncrement(s,s.blocked)) return; s.lastOutcome=ControlledOutcome::RankBlocked; }
    else {
        uint32_t count=recovery?1+randomWord(&s.rng)%2:1;
        for(uint32_t p=0;p<count;++p) {
            if(uint64_t(current.m)+1>c.ceiling) { if(!controlledIncrement(s,s.blocked)) return; s.lastOutcome=ControlledOutcome::RankBlocked; break; }
            uint32_t op=randomWord(&s.rng)%3;
            if(controlledPrimitive(c,s,current,op)==ControlledOutcome::Applied) {
                ++applied; controlledObserve(c,s,current,best,ControlledOperation::Expansion,sink);
            }
            if(controlledTerminal(s)) return;
        }
    }
    controlledRedraw(c,s);
    if(recovery && !applied) { s.pendingRestart=1; s.lastOutcome=ControlledOutcome::RestartRequested; }
}
template<class S,class Best,class Sink> void controlledStep(LOCAL const ControlledSettings &c,LOCAL ControlledState &s,
        LOCAL S &current,LOCAL Best &best,LOCAL Sink &sink) {
    if(controlledTerminal(s)) return;
    if(!s.initialized || candidateOverflow(current)) { controlledFail(s,candidateOverflow(current)?ControlledOutcome::CapacityError:ControlledOutcome::InvalidState); return; }
    if(controlledBudget(c,s)) { controlledFail(s,ControlledOutcome::BudgetExhausted); return; }
    if(s.pendingRestart) return;
    ++s.controls; ++s.flips;
    if(s.countdown) --s.countdown;
    if(!controlledIncrement(s,s.stagnation) || !controlledIncrement(s,s.attempted[0])) return;
    int before=current.m;
    bool success=current.tryFlip(s.rng,true);
    if(candidateOverflow(current)) { controlledFail(s,ControlledOutcome::CapacityError); return; }
    s.lastOutcome=success?ControlledOutcome::Applied:ControlledOutcome::UnsuccessfulFlip;
    if(success) {
        if(!controlledIncrement(s,s.applied[0])) return;
        if(!controlledAdd(s,s.removed[0],uint64_t(before-current.m)) ||
           !controlledAdd(s,s.removed[2],uint64_t(before-current.m))) return;
        controlledObserve(c,s,current,best,ControlledOperation::Flip,sink);
    }
    if(controlledTerminal(s)) return;
    if(controlledBudget(c,s)) { controlledFail(s,ControlledOutcome::BudgetExhausted); return; }
    if(success && randomWord(&s.rng)<=c.reductionQ) {
        if(!controlledIncrement(s,s.attempted[1])) return;
        before=current.m;
        bool reduced=current.tryReduce();
        if(candidateOverflow(current)) { controlledFail(s,ControlledOutcome::CapacityError); return; }
        if(reduced) {
            if(!controlledIncrement(s,s.applied[1])) return;
            if(!controlledAdd(s,s.removed[1],uint64_t(before-current.m)) ||
               !controlledAdd(s,s.removed[2],uint64_t(before-current.m))) return;
            controlledObserve(c,s,current,best,ControlledOperation::Reduction,sink);
        }
        if(controlledTerminal(s)) return;
    }
    if(!success || !s.countdown) controlledExpansion(c,s,current,best,!success,sink);
    if(controlledTerminal(s)) return;
    if(s.stagnation>=c.stagnationLimit && !s.pendingRestart) { s.pendingRestart=1; s.lastOutcome=ControlledOutcome::RestartRequested; }
}
inline bool controlledCommit(LOCAL ControlledState &s,bool verified,bool durable) {
    if(!verified || !durable || s.terminal==ControlledOutcome::CapacityError || s.terminal==ControlledOutcome::InvalidState) return false;
    s.mandatoryCommitted=1;
    if(s.terminal==ControlledOutcome::TargetPending || s.terminal==ControlledOutcome::ExistingTargetPending) s.terminal=ControlledOutcome::TargetMet;
    return true;
}
inline bool controlledBoundary(LOCAL ControlledState &s) {
    if(!s.mandatoryCommitted || s.terminal==ControlledOutcome::CapacityError || s.terminal==ControlledOutcome::InvalidState) return false;
    s.mandatoryValid=0; s.optionalCount=0; s.optionalEncounters=0; s.optionalDrops=0; return true;
}
template<class S> bool controlledInstallRestart(LOCAL const ControlledSettings &c,LOCAL ControlledState &s,
        LOCAL const S &parent,LOCAL S &current,LOCAL S &best,bool parentVerified) {
    if(!s.pendingRestart || controlledTerminal(s) || !s.mandatoryCommitted || !parentVerified ||
       parent.m<1 || parent.m>MAX_RANK || candidateOverflow(parent)) return false;
    for(int p=0;p<3;++p) if(parent.n[p]!=current.n[p] || parent.nn[p]!=current.nn[p]) return false;
    if(controlledTarget(c,parent.m)) { controlledFail(s,ControlledOutcome::ExistingTargetPending); return true; }
    if(controlledBudget(c,s)) { controlledFail(s,ControlledOutcome::BudgetExhausted); return true; }
    ++s.controls; current=parent; best=parent; s.stagnation=0; s.pendingRestart=0;
    controlledRedraw(c,s); return true;
}
inline bool controlledChargeStage(LOCAL const ControlledSettings &c,LOCAL ControlledState &s) {
    if(s.controls>=c.controlBudget) return false;
    ++s.controls; return true;
}

static_assert(sizeof(ControlledSettings)==96,"controlled settings layout changed");
static_assert(sizeof(ControlledState)==200,"controlled state layout changed");
