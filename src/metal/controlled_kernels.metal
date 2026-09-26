// The legacy kernels above provide the ordered load/store helpers.
template<class S> struct ControlledDeviceSink {
    device S *captures;
    device ControlledCaptureMeta *metadata;
    void capture(thread const S &scheme, ControlledOperation operation, uint64_t control,
                 bool mandatory, bool optional, uint64_t index) thread {
        if (mandatory) {
            storeObject(captures, scheme);
            metadata[0] = {control, uint32_t(operation), 0};
        }
        if (optional) {
            storeObject(captures + index + 1, scheme);
            metadata[index + 1] = {control, uint32_t(operation), 0};
        }
    }
};

// Best state only needs its rank while a dispatch runs. Store improvements
// directly to the dedicated buffer instead of doubling private scheme storage.
template<class S> struct ControlledDeviceBest {
    device S *destination;
    int m;
    void operator=(thread const S &source) thread {storeObject(destination,source);m=source.m;}
};

template<class S> void controlledGeneralBody(device S *current, device S *best,
    device ControlledState *states, device S *captures, device ControlledCaptureMeta *metadata,
    constant ControlledSettings &settings, uint64_t count, uint64_t steps,
    device int *errors, uint index, uint command) {
    if (uint64_t(index) >= count) return;
    S local;
    ControlledState state;
    loadObject(local, current + index);
    ControlledDeviceBest<S> localBest{best+index,best[index].m};
    loadObject(state, states + index);
    ControlledSettings config = settings;
    const auto base = uint64_t(index) * (config.optionalQuota + 1);
    ControlledDeviceSink<S> sink{captures + base, metadata + base};
    if (command) controlledPrimitive(config, state, local, command - 1);
    else for (uint64_t step = 0; step < steps; ++step) {
        if (controlledTerminal(state) || state.pendingRestart) break;
        controlledStep(config, state, local, localBest, sink);
    }
    storeObject(current + index, local);
    storeObject(states + index, state);
    errors[index] = state.terminal == ControlledOutcome::CapacityError ? 4 :
                    state.terminal == ControlledOutcome::InvalidState ? 5 : 0;
}

kernel void controlledGeneralKernel(device Scheme *current [[buffer(0)]],
    device Scheme *best [[buffer(1)]], device ControlledState *states [[buffer(2)]],
    device Scheme *captures [[buffer(3)]], device ControlledCaptureMeta *metadata [[buffer(4)]],
    constant ControlledSettings &settings [[buffer(5)]], constant uint64_t &count [[buffer(6)]],
    constant uint64_t &steps [[buffer(7)]], device int *errors [[buffer(30)]], uint index [[thread_position_in_grid]]) {
    controlledGeneralBody(current, best, states, captures, metadata, settings, count, steps, errors, index, 0);
}

#ifdef METAL_TESTING
kernel void controlledTestKernel(device Scheme *current [[buffer(0)]],
    device Scheme *best [[buffer(1)]], device ControlledState *states [[buffer(2)]],
    device Scheme *captures [[buffer(3)]], device ControlledCaptureMeta *metadata [[buffer(4)]],
    constant ControlledSettings &settings [[buffer(5)]], constant uint64_t &count [[buffer(6)]],
    constant uint64_t &steps [[buffer(7)]], constant uint &command [[buffer(8)]],
    device int *errors [[buffer(30)]], uint index [[thread_position_in_grid]]) {
    controlledGeneralBody(current, best, states, captures, metadata, settings, count, steps, errors, index, command);
}
#endif

#if !defined(METAL_F2)
void controlledPackedBody(device SchemeInteger *current,device SchemeInteger *best,
    device ControlledState *states,device SchemeInteger *captures,device ControlledCaptureMeta *metadata,
    constant ControlledSettings &settings,uint64_t count,uint64_t steps,
    device CompactStoredAddition *terms,device uint32_t *pairs,device int *scratch,
    device CompactStoredAddition *bestTerms,device uint32_t *bestPairs,
    device int *errors,uint index,uint command,uint mode) {
    if(uint64_t(index)>=count) return;
    if(settings.alternatives!=mode) { errors[index]=5; return; }
    ControlledPackedScheme local,localBest;
    local.bind(terms,pairs,scratch,index); localBest.bind(bestTerms,bestPairs,nullptr,index);
    uint32_t failure=local.load(current[index]);
    if(!failure) failure=localBest.load(best[index]);
    if(failure) { errors[index]=failure; return; }
    ControlledState state; loadObject(state,states+index);
    ControlledSettings config=settings;
    uint64_t base=uint64_t(index)*(config.optionalQuota+1);
    ControlledPackedSink sink{captures+base,metadata+base};
    if(command) controlledPrimitive(config,state,local,command-1);
    else for(uint64_t step=0;step<steps;++step) {
        if(controlledTerminal(state) || state.pendingRestart) break;
        controlledStep(config,state,local,localBest,sink);
    }
    local.store(current[index]); localBest.store(best[index]); storeObject(states+index,state);
    errors[index]=state.terminal==ControlledOutcome::CapacityError?4:state.terminal==ControlledOutcome::InvalidState?5:0;
}
#define CONTROLLED_PACKED_ARGUMENTS \
    device SchemeInteger *current [[buffer(0)]],device SchemeInteger *best [[buffer(1)]], \
    device ControlledState *states [[buffer(2)]],device SchemeInteger *captures [[buffer(3)]], \
    device ControlledCaptureMeta *metadata [[buffer(4)]],constant ControlledSettings &settings [[buffer(5)]], \
    constant uint64_t &count [[buffer(6)]],constant uint64_t &steps [[buffer(7)]], \
    device CompactStoredAddition *terms [[buffer(8)]],device uint32_t *pairs [[buffer(9)]], \
    device int *scratch [[buffer(10)]],device CompactStoredAddition *bestTerms [[buffer(11)]], \
    device uint32_t *bestPairs [[buffer(12)]],device int *errors [[buffer(30)]],uint index [[thread_position_in_grid]]
#define CONTROLLED_PACKED_CALL(mode,command) \
    controlledPackedBody(current,best,states,captures,metadata,settings,count,steps,terms,pairs,scratch,bestTerms,bestPairs,errors,index,command,mode)
kernel void controlledPackedReductionKernel(CONTROLLED_PACKED_ARGUMENTS) { CONTROLLED_PACKED_CALL(0,0); }
kernel void controlledPackedAlternativesKernel(CONTROLLED_PACKED_ARGUMENTS) { CONTROLLED_PACKED_CALL(1,0); }
#ifdef METAL_TESTING
kernel void controlledPackedTestKernel(CONTROLLED_PACKED_ARGUMENTS,constant uint &command [[buffer(13)]]) {
    CONTROLLED_PACKED_CALL(settings.alternatives,command);
}
#endif
#undef CONTROLLED_PACKED_CALL
#undef CONTROLLED_PACKED_ARGUMENTS
#endif
