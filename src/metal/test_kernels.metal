kernel void layoutKernel(device uint *output [[buffer(0)]]) {
    output[0] = sizeof(Addition);
    output[1] = sizeof(Scheme);
    output[2] = sizeof(ReducerU);
    output[3] = sizeof(ReducerW);
    output[4] = alignof(Addition);
    output[5] = __builtin_offsetof(Addition, values);
    output[6] = __builtin_offsetof(Addition, signs);
    output[7] = __builtin_offsetof(Addition, valid);
    output[8] = alignof(Scheme);
    output[9] = __builtin_offsetof(Scheme, uvw);
    output[10] = __builtin_offsetof(Scheme, flips);
    output[11] = sizeof(FlipSet);
    output[12] = alignof(FlipSet);
    output[13] = __builtin_offsetof(FlipSet, size);
    output[14] = __builtin_offsetof(FlipSet, overflow);
    output[15] = __builtin_offsetof(FlipSet, pairs);
}

kernel void arithmeticKernel(device int *output [[buffer(0)]], uint idx [[thread_position_in_grid]]) {
    int a = int(idx % 3) - 1, b = int((idx / 3) % 3) - 1;
    int bit = idx / 9;
    Addition left(64), right(64);
    left.set(bit, a); right.set(bit, b);
    Addition sum = left + right, difference = left - right;
    output[4*idx] = sum.valid;
    output[4*idx+1] = sum[bit];
    output[4*idx+2] = difference.valid;
    output[4*idx+3] = difference[bit];
}

kernel void transformationKernel(device Scheme *schemes [[buffer(0)]], device RandomState *states [[buffer(1)]], device Scheme *other [[buffer(2)]], constant int &operation [[buffer(3)]], device int *errors [[buffer(30)]], uint idx [[thread_position_in_grid]]) {
    Scheme scheme;
    RandomState state;
    loadObject(scheme, schemes + idx);
    loadObject(state, states + idx);
    switch (operation) {
        case 0: for (int i = 0; i < 32; i++) scheme.tryFlip(state); break;
        case 1: scheme.tryPlus(state); break;
        case 2: scheme.tryExpand(2, state); break;
        case 3: scheme.tryReduce(); break;
        case 4: scheme.tryProject(state); break;
        case 5: scheme.tryExtend(state); break;
        case 6: scheme.tryProduct(state); break;
        case 7: scheme.swapBasis(state); break;
        case 8: scheme.swapSize(state); break;
        case 9: scheme.tryMerge(other[0], state); break;
        case 10: scheme.tryProduct(other[0]); break;
#ifdef METAL_F2
        case 11: scheme.sandwiching(state); break;
        case 12: scheme.tryReduceGauss(state); break;
#endif
    }
    storeObject(schemes + idx, scheme);
    storeObject(states + idx, state);
    errors[idx] = candidateOverflow(scheme) ? 4 : !scheme.validate();
}

kernel void reducerTestKernel(device ReducerU *reducers [[buffer(0)]], device RandomState *states [[buffer(1)]], device int *errors [[buffer(30)]], uint idx [[thread_position_in_grid]]) {
    RandomState state;
    loadObject(state, states + idx);
    reducers[idx].reduce(state);
    storeObject(states + idx, state);
    errors[idx] = !reducers[idx].isValid();
}

kernel void randomAdditionKernel(device Addition *output [[buffer(0)]], device RandomState *states [[buffer(1)]], uint idx [[thread_position_in_grid]]) {
    Addition addition(idx + 1);
    RandomState state;
    loadObject(state, states + idx);
    addition.random(state);
    storeObject(output + idx, addition);
    storeObject(states + idx, state);
}

kernel void capacityTestKernel(device ReducerU *reducers [[buffer(0)]], device int *output [[buffer(1)]]) {
    int values[33];
    for (int i = 0; i < 33; i++) values[i] = 1;
    reducers[0].clear();
    bool accepted = reducers[0].addExpression(values, 33);
    output[0] = !accepted && !reducers[0].isValid();
}

// Exercise the boundary and a transient overflow erased from the visible list.
kernel void candidateCapacityKernel(device uint *output [[buffer(0)]],
                                    device uint *storage [[buffer(1)]]) {
    FlipSet set;
    for (uint i = 0; i < MAX_PAIRS; i++) set.add(0, i);
    output[0] = set.size == MAX_PAIRS && !set.overflow;
    set.add(1, 2);
    set.remove(0, 0);
    set.clear();
    set.add(2, 3);
    output[1] = set.size == 1 && set.overflow;
#ifndef METAL_F2
    CompactFlipSet compact;
    compact.size = 0;
    compact.overflow = 0;
    compact.pairs = storage;
    for (uint i = 0; i < MAX_PAIRS; i++) compact.add(0, i);
    output[2] = compact.size == MAX_PAIRS && !compact.overflow;
    compact.add(1, 2);
    compact.remove(0, 0);
    compact.clear();
    compact.add(2, 3);
    output[3] = compact.size == 1 && compact.overflow;
#endif
}
