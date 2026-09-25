#ifndef METAL_F2
// Direct-workflow-only counters; legacy dispatches allocate and update none.
kernel void directMutationKernel(device SchemeInteger *schemes [[buffer(0)]],device RandomState *states [[buffer(1)]],
    device uint64_t *counters [[buffer(2)]],constant int &count [[buffer(3)]],constant int &maximum [[buffer(4)]],
    device int *errors [[buffer(30)]],uint index [[thread_position_in_grid]]) {
    if(index==0 || index>=uint(count)) return;
    SchemeInteger scheme; loadObject(scheme,schemes);
    RandomState rng; loadObject(rng,states+index);
    uint32_t requested=randomWord(&rng)%uint32_t(maximum+1);
    scheme.copyTo(scheme);
    for(uint32_t attempt=0;attempt<requested;++attempt) {
        ++counters[index*2];
        if(!scheme.tryFlip(rng)) break;
        ++counters[index*2+1];
        if(candidateOverflow(scheme)) break;
    }
    storeObject(schemes+index,scheme); storeObject(states+index,rng);
    errors[index]=candidateOverflow(scheme)?4:!scheme.validate();
}
// Retain existing reducer modes. Each triplet now uses its assigned mutated
// scheme, whereas the legacy fixed-scheme kernel intentionally reads lane zero.
kernel void runDirectReducersKernel(device ReducerU *reducersU [[buffer(0)]],device ReducerV *reducersV [[buffer(1)]],
    device ReducerW *reducersW [[buffer(2)]],device SchemeInteger *schemes [[buffer(3)]],device RandomState *states [[buffer(4)]],
    constant int &count [[buffer(5)]],constant int &schemesCount [[buffer(6)]],device int *errors [[buffer(30)]],uint index [[thread_position_in_grid]]) {
    if(index>=uint(count)) return;
    SchemeInteger scheme; loadObject(scheme,schemes+index%uint(schemesCount));
    RandomState rng; loadObject(rng,states+index);
    device ReducerU &u=reducersU[index]; device ReducerV &v=reducersV[index]; device ReducerW &w=reducersW[index];
    int modes[]={GREEDY_INTERSECTIONS_MODE,GREEDY_INTERSECTIONS_MODE,GREEDY_INTERSECTIONS_MODE,GREEDY_INTERSECTIONS_MODE,
        GREEDY_INTERSECTIONS_MODE,GREEDY_INTERSECTIONS_MODE,GREEDY_INTERSECTIONS_MODE,GREEDY_INTERSECTIONS_MODE,
        GREEDY_ALTERNATIVE_MODE,GREEDY_ALTERNATIVE_MODE,GREEDY_ALTERNATIVE_MODE,GREEDY_ALTERNATIVE_MODE,
        GREEDY_RANDOM_MODE,GREEDY_RANDOM_MODE,MIX_MODE};
    u.setMode(index==0?0:modes[randomWord(&rng)%15]);
    v.setMode(index==0?0:modes[randomWord(&rng)%15]);
    w.setMode(index==0?0:modes[randomWord(&rng)%15]);
    copySchemeToReducers(u,v,w,scheme);
    u.reduce(rng); v.reduce(rng); w.reduce(rng);
    storeObject(states+index,rng);
    errors[index]=candidateOverflow(scheme)?4:(!u.isValid() || !v.isValid() || !w.isValid())?2:0;
}
#endif
