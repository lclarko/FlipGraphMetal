kernel void profileCopyKernel(device const Scheme *input [[buffer(0)]], device Scheme *output [[buffer(1)]], constant int &count [[buffer(2)]], uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(count)) return;
    Scheme scheme;
    loadObject(scheme, input + idx);
    storeObject(output + idx, scheme);
}

kernel void profileValidateKernel(device const Scheme *input [[buffer(0)]], constant int &count [[buffer(1)]], device int *errors [[buffer(30)]], uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(count)) return;
    Scheme scheme;
    loadObject(scheme, input + idx);
    errors[idx] = !scheme.validate();
}

kernel void profilePermutationKernel(device const Scheme *input [[buffer(0)]], device RandomState *states [[buffer(1)]], device uint *output [[buffer(2)]], constant int &count [[buffer(3)]], constant int &iterations [[buffer(4)]], uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(count)) return;
    RandomState state;
    loadObject(state, states + idx);
    int size = input[idx].flips[0].size + input[idx].flips[1].size + input[idx].flips[2].size;
    int indices[MAX_PAIRS * 3];
    uint sum = 0;
    for (int iteration = 0; iteration < iterations; iteration++) {
        randomPermutation(indices, size, state);
        for (int i = 0; i < size; i++) sum = (sum ^ uint(indices[i])) * 16777619u;
    }
    output[idx] = sum;
    storeObject(states + idx, state);
}

kernel void profileSelectionKernel(device const Scheme *input [[buffer(0)]], device RandomState *states [[buffer(1)]], device uint *output [[buffer(2)]], constant int &count [[buffer(3)]], constant int &iterations [[buffer(4)]], uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(count)) return;
    Scheme scheme;
    RandomState state;
    loadObject(scheme, input + idx);
    loadObject(state, states + idx);
    uint sum = 0;
    for (int iteration = 0; iteration < iterations; iteration++)
        sum = (sum ^ profileSelect(scheme, state)) * 16777619u;
    output[idx] = sum;
    storeObject(states + idx, state);
}
