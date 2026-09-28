#pragma once

// Included after CircuitStage and its expression helpers.
#include "../metal/two_aux_constructor.h"

namespace fgm {
struct TwoAuxValidation {
    CircuitStage stage;
    uint64_t negativeRuleChecks=0;
    int usedHelpers=0;
    int liveCount=0;
};

// Check a device trace against trusted host-prepared rules. In particular, a
// negative trace must already be closed; this scan never constructs a result.
inline TwoAuxValidation validateTwoAuxWitness(const fgm_constructor::TwoAuxProblem &p,
                                              const fgm_constructor::TwoAuxTask &task,
                                              const fgm_constructor::TwoAuxWitness &w) {
    using namespace fgm_constructor;
    if(p.inputs<1 || p.inputs>MaxInputs || p.directions<p.inputs || p.directions>32 ||
       p.outputs<0 || p.outputs>MaxTargets || p.operationCount<0 || p.operationCount>MaxCandidates ||
       task.operationCount<0 || task.operationCount>TwoAuxMaxExtraRules)
        throw std::runtime_error("invalid two-auxiliary problem bounds");
    const int directions=p.directions+2;
    if((w.status!=0 && w.status!=1) || w.count<0 || w.count>TwoAuxMaxGates ||
       w.sweeps<0 || w.sweeps>TwoAuxMaxGates+1 ||
       w.ruleChecks>uint64_t(TwoAuxMaxRules)*(TwoAuxMaxGates+1) ||
       w.ruleChecks<uint64_t(w.count) || (w.available>>directions))
        throw std::runtime_error("invalid two-auxiliary witness bounds");
    auto equalGate=[](const Gate &a,const Gate &b) {
        return a.out==b.out && a.left==b.left && a.right==b.right &&
               a.leftSign==b.leftSign && a.rightSign==b.rightSign;
    };
    auto eachRule=[&](auto visit) {
        for(int i=0;i<p.operationCount;++i) visit(p.operations[i]);
        for(int i=0;i<task.operationCount;++i) visit(task.operations[i]);
        for(const auto &g:task.creations) visit(g);
    };
    // The rule arrays are host-owned, but check indices before any shift or
    // access so a malformed test/task fails at the validation boundary.
    eachRule([&](const Gate &g) {
        if(g.out<p.inputs || g.out>=directions || g.left<0 || g.left>=directions ||
           g.right<0 || g.right>=directions || g.left==g.right ||
           (g.leftSign!=1 && g.leftSign!=-1) || (g.rightSign!=1 && g.rightSign!=-1))
            throw std::runtime_error("invalid two-auxiliary rule bounds");
    });
    Matrix vectors(directions,std::vector<int64_t>(p.inputs));
    for(int i=0;i<p.directions;++i)
        for(int c=0;c<p.inputs;++c) vectors[i][c]=p.vectors[i][c];
    for(int h=0;h<2;++h)
        for(int c=0;c<p.inputs;++c) vectors[p.directions+h][c]=task.helpers[h][c];
    uint64_t available=(uint64_t(1)<<p.inputs)-1;
    for(int i=0;i<w.count;++i) {
        const auto &g=w.gates[i];
        if(g.out<p.inputs || g.out>=directions || g.left<0 || g.left>=directions ||
           g.right<0 || g.right>=directions || g.left==g.right ||
           (g.leftSign!=1 && g.leftSign!=-1) || (g.rightSign!=1 && g.rightSign!=-1))
            throw std::runtime_error("invalid two-auxiliary gate bounds");
        if((available&(uint64_t(1)<<g.out)) || !(available&(uint64_t(1)<<g.left)) ||
           !(available&(uint64_t(1)<<g.right)))
            throw std::runtime_error("unreachable two-auxiliary dependency");
        bool permitted=false;
        eachRule([&](const Gate &rule) { permitted |= equalGate(g,rule); });
        if(!permitted || (g.out>=p.directions && !equalGate(g,task.creations[g.out-p.directions])))
            throw std::runtime_error("two-auxiliary creation route or rule changed");
        // Prepared vectors are int32, so a signed pair fits int64 even for
        // malformed vector values. Compare before narrowing any result.
        for(int c=0;c<p.inputs;++c)
            if(vectors[g.out][c]!=int64_t(g.leftSign)*vectors[g.left][c]+
                                 int64_t(g.rightSign)*vectors[g.right][c])
                throw std::runtime_error("incorrect two-auxiliary gate arithmetic");
        available|=uint64_t(1)<<g.out;
    }
    const uint64_t required=(uint64_t(1)<<p.directions)-1;
    const bool complete=(available&required)==required;
    if(available!=w.available || (w.status==1)!=complete)
        throw std::runtime_error("two-auxiliary completion mismatch");
    TwoAuxValidation result;
    if(!complete) {
        eachRule([&](const Gate &g) {
            ++result.negativeRuleChecks;
            if((available&(uint64_t(1)<<g.left)) && (available&(uint64_t(1)<<g.right)) &&
               !(available&(uint64_t(1)<<g.out)))
                throw std::runtime_error("unsuccessful two-auxiliary witness is not a fixed point");
        });
        return result;
    }

    // Successful closure may have emitted unused helpers. Trim the actual
    // trace backwards, then assign wires in its original topological order.
    uint64_t live=required;
    for(int i=w.count-1;i>=0;--i) {
        const auto &g=w.gates[i];
        if(live&(uint64_t(1)<<g.out)) live|=(uint64_t(1)<<g.left)|(uint64_t(1)<<g.right);
    }
    int wires[TwoAuxMaxDirections]{};
    for(int i=0;i<p.inputs;++i) wires[i]=i+1;
    for(int i=0;i<w.count;++i) {
        const auto &g=w.gates[i];
        if(!(live&(uint64_t(1)<<g.out))) continue;
        if(!wires[g.left] || !wires[g.right]) throw std::runtime_error("invalid two-auxiliary liveness");
        result.stage.fresh.array.push_back(circuitExpression({g.leftSign*wires[g.left],g.rightSign*wires[g.right]}));
        wires[g.out]=p.inputs+(++result.liveCount);
        result.usedHelpers+=g.out>=p.directions;
    }
    for(int i=0;i<p.outputs;++i) {
        const int d=p.outputDirection[i], sign=p.outputSign[i];
        if(d==-1 && sign==0) result.stage.outputs.array.push_back(circuitExpression({}));
        else {
            if(d<0 || d>=p.directions || !wires[d] || (sign!=1 && sign!=-1))
                throw std::runtime_error("invalid two-auxiliary output mapping");
            result.stage.outputs.array.push_back(circuitExpression({sign*wires[d]}));
        }
    }
    if(result.liveCount!=p.directions-p.inputs+result.usedHelpers)
        throw std::runtime_error("two-auxiliary operation count mismatch");
    return result;
}
}
