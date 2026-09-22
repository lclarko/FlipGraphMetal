kernel void layoutKernel(device uint *output [[buffer(0)]]) {
    output[0] = sizeof(Addition);
    output[1] = sizeof(Scheme);
    output[2] = sizeof(ReducerU);
    output[3] = sizeof(ReducerW);
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
    errors[idx] = !scheme.validate();
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
