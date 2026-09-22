template <class T> void loadObject(thread T &target, device const T *source) {
    thread uchar *dst = reinterpret_cast<thread uchar *>(&target);
    device const uchar *src = reinterpret_cast<device const uchar *>(source);
    for (size_t i = 0; i < sizeof(T); i++) dst[i] = src[i];
}
template <class T> void storeObject(device T *target, thread const T &source) {
    device uchar *dst = reinterpret_cast<device uchar *>(target);
    thread const uchar *src = reinterpret_cast<thread const uchar *>(&source);
    for (size_t i = 0; i < sizeof(T); i++) dst[i] = src[i];
}
void loadObject(thread SchemeInteger &target, device const SchemeInteger *source) {
    target.m = source->m;
    for (int p = 0; p < 3; p++) {
        target.n[p] = source->n[p];
        target.nn[p] = source->nn[p];
        for (int r = 0; r < target.m; r++) {
            target.uvw[p][r].n = source->uvw[p][r].n;
            target.uvw[p][r].values = source->uvw[p][r].values;
            target.uvw[p][r].signs = source->uvw[p][r].signs;
            target.uvw[p][r].valid = source->uvw[p][r].valid;
        }
        target.flips[p].size = source->flips[p].size;
        for (size_t i = 0; i < target.flips[p].size; i++)
            target.flips[p].pairs[i] = source->flips[p].pairs[i];
    }
}

void storeObject(device SchemeInteger *target, thread const SchemeInteger &source) {
    target->m = source.m;
    for (int p = 0; p < 3; p++) {
        target->n[p] = source.n[p];
        target->nn[p] = source.nn[p];
        for (int r = 0; r < source.m; r++) {
            target->uvw[p][r].n = source.uvw[p][r].n;
            target->uvw[p][r].values = source.uvw[p][r].values;
            target->uvw[p][r].signs = source.uvw[p][r].signs;
            target->uvw[p][r].valid = source.uvw[p][r].valid;
        }
        target->flips[p].size = source.flips[p].size;
        for (size_t i = 0; i < source.flips[p].size; i++)
            target->flips[p].pairs[i] = source.flips[p].pairs[i];
    }
}

#ifdef METAL_F2
using Scheme = SchemeZ2;
#else
using Scheme = SchemeInteger;
#endif
using ReducerU = AdditionsReducer<350, 250, 32, 2016>;
using ReducerV = ReducerU;
using ReducerW = AdditionsReducer<64, 500, 175, 61075>;

kernel void initializeRandomStatesKernel(device RandomState *states [[buffer(0)]], constant int &schemesCount [[buffer(1)]], constant int &seed [[buffer(2)]], device int *errors [[buffer(30)]], uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(schemesCount)) return;
    states[idx].value = uint(seed) ^ (0x9e3779b9u * (idx + 1));
    if (states[idx].value == 0) states[idx].value = 1;

}

kernel void initializeNaiveKernel(device Scheme *schemes [[buffer(0)]], constant int &schemesCount [[buffer(1)]], constant int &n1 [[buffer(2)]], constant int &n2 [[buffer(3)]], constant int &n3 [[buffer(4)]], device int *errors [[buffer(30)]], uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(schemesCount)) return;
    Scheme scheme;
    scheme.initializeNaive(n1,n2,n3);
    storeObject(schemes + idx, scheme);
    errors[idx] = !scheme.validate();
}

kernel void initializeCopyKernel(device Scheme *schemes [[buffer(0)]], constant int &schemesCount [[buffer(1)]], constant int &count [[buffer(2)]], device int *errors [[buffer(30)]], uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(schemesCount) || idx < uint(count)) return;
    Scheme scheme;
    loadObject(scheme, schemes + idx % count);
    scheme.copyTo(scheme);
    storeObject(schemes + idx, scheme);
    errors[idx] = !scheme.validate();
}

kernel void initializeSchemesKernel(device Scheme *schemes [[buffer(0)]], device Scheme *schemesBest [[buffer(1)]], device int *bestRanks [[buffer(2)]], device int *flips [[buffer(3)]], constant int &n1 [[buffer(4)]], constant int &n2 [[buffer(5)]], constant int &n3 [[buffer(6)]], constant int &schemesCount [[buffer(7)]], device int *errors [[buffer(30)]], uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(schemesCount)) return;
    Scheme scheme;
    loadObject(scheme, schemes + idx);
    storeObject(schemesBest + idx, scheme);
    bestRanks[idx] = n1*n2*n3;
    flips[idx] = 0;
    errors[idx] = !scheme.validate();
}

kernel void randomWalkKernel(device Scheme *schemes [[buffer(0)]], device Scheme *schemesBest [[buffer(1)]], device int *bestRanks [[buffer(2)]], device int *flips [[buffer(3)]], device RandomState *states [[buffer(4)]], constant int &schemesCount [[buffer(5)]], constant int &maxIterations [[buffer(6)]], constant int &plusIterations [[buffer(7)]], constant float &reduceProbability [[buffer(8)]], constant float &expandProbability [[buffer(9)]], constant float &sandwichingProbability [[buffer(10)]], constant float &basisProbability [[buffer(11)]], constant bool &randomIterations [[buffer(12)]], device int *errors [[buffer(30)]], uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(schemesCount)) return;
    Scheme scheme;
    RandomState state;
    loadObject(scheme, schemes + idx);
    int savedRank = schemesBest[idx].m;
    loadObject(state, states + idx);
    int flipsCount = flips[idx];
    int bestRank = bestRanks[idx];
    int iterations = randomIterations ? randint(1, maxIterations, state) : maxIterations;

    for (int iteration = 0; iteration < iterations; iteration++) {
        int rank = scheme.m;

        if (!scheme.tryFlip(state)) {
            scheme.tryExpand(randint(1, 2, state), state);
            continue;
        }

        if (scheme.m < rank)
            flipsCount = 0;

        flipsCount++;

        if (scheme.m < bestRank || (scheme.m == bestRank && randomUniform(&state) < 0.01f)) {
            bestRank = scheme.m;
            storeObject(schemesBest + idx, scheme);
            savedRank = scheme.m;
        }

        if (randomUniform(&state) * maxIterations < reduceProbability)
            scheme.tryReduce();

        if (randomUniform(&state) * maxIterations < expandProbability && scheme.m <= savedRank + 2) {
            if (scheme.tryExpand(randint(1, 2, state), state))
                flipsCount = 0;
        }

#ifdef METAL_F2
        if (randomUniform(&state) * maxIterations < sandwichingProbability) scheme.sandwiching(state);
#else
        randomUniform(&state);
#endif

        if (randomUniform(&state) * maxIterations < basisProbability)
            scheme.swapBasis(state);

        if (flipsCount >= plusIterations && scheme.tryPlus(state))
            flipsCount = 0;
    }

    flips[idx] = flipsCount;
    bestRanks[idx] = bestRank;

    storeObject(schemes + idx, scheme);
    storeObject(states + idx, state);
    errors[idx] = !scheme.validate();
}

kernel void resizeKernel(device Scheme *schemes [[buffer(0)]], device Scheme *schemesBest [[buffer(1)]], constant int &schemesCount [[buffer(2)]], device RandomState *states [[buffer(3)]], constant float &resizeProbability [[buffer(4)]], device int *errors [[buffer(30)]], uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(schemesCount)) return;
    Scheme scheme;
    RandomState state;
    loadObject(state, states + idx);
    if (randomUniform(&state) > resizeProbability) {
        storeObject(states + idx, state);
        return;
    }
    loadObject(scheme, schemes + idx);
    if (randomUniform(&state) < 0.5f) scheme.swapSize(state);
    bool merged = false;
    for (int i = 0; i < 3 && !merged; i++) {
        int index = randomWord(&state) % schemesCount;
        merged |= scheme.tryMerge(schemesBest[index], state);
    }
    if (!merged) {
        float p = randomUniform(&state);
        if (p < 0.05f) scheme.tryProject(state);
        else if (p < 0.55f) {
            int index = randomWord(&state) % schemesCount;
            scheme.tryProduct(schemesBest[index]);
        }
        else if (p < 0.85f) scheme.tryProduct(state);
        else scheme.tryExtend(state);
    }
    storeObject(schemes + idx, scheme);
    storeObject(states + idx, state);
    errors[idx] = !scheme.validate();
}

kernel void initializeMinimizerKernel(device Scheme *schemes [[buffer(0)]], device Scheme *schemesBest [[buffer(1)]], device int *bestComplexities [[buffer(2)]], device RandomState *states [[buffer(3)]], constant int &schemesCount [[buffer(4)]], constant int &initialCount [[buffer(5)]], constant int &complexity [[buffer(6)]], constant int &seed [[buffer(7)]], device int *errors [[buffer(30)]], uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(schemesCount)) return;
    Scheme scheme;
    loadObject(scheme, schemes + idx % initialCount);
    scheme.copyTo(scheme);
    if (idx >= uint(initialCount)) storeObject(schemes + idx, scheme);
    storeObject(schemesBest + idx, scheme);
    bestComplexities[idx] = complexity;
    states[idx].value = uint(seed) ^ (0x9e3779b9u * (idx + 1));
    if (states[idx].value == 0) states[idx].value = 1;
    errors[idx] = !scheme.validate();
}

kernel void minimizeKernel(device Scheme *schemes [[buffer(0)]], device Scheme *schemesBest [[buffer(1)]], device int *bestComplexities [[buffer(2)]], device RandomState *states [[buffer(3)]], constant int &schemesCount [[buffer(4)]], constant int &iterations [[buffer(5)]], device int *errors [[buffer(30)]], uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(schemesCount)) return;
    Scheme scheme;
    RandomState state;
    loadObject(scheme, schemes + idx);
    loadObject(state, states + idx);
    int bestComplexity = bestComplexities[idx];
    for (int iteration = 0; iteration < iterations; iteration++) {
        if (!scheme.tryFlip(state, false)) break;
        int complexity = scheme.getComplexity();
        if (complexity < bestComplexity) {
            bestComplexity = complexity;
            storeObject(schemesBest + idx, scheme);
        }
    }
    bestComplexities[idx] = bestComplexity;
    storeObject(schemes + idx, scheme);
    storeObject(states + idx, state);
    errors[idx] = !scheme.validate();
}

void copySchemeToReducers(device ReducerU &u, device ReducerV &v, device ReducerW &w, thread const SchemeInteger &scheme) {
    int values[MAX_RANK];
    u.clear(); v.clear(); w.clear();
    for (int index = 0; index < scheme.m; index++) {
        for (int i = 0; i < scheme.nn[0]; i++) values[i] = scheme.uvw[0][index][i];
        u.addExpression(values, scheme.nn[0]);
        for (int i = 0; i < scheme.nn[1]; i++) values[i] = scheme.uvw[1][index][i];
        v.addExpression(values, scheme.nn[1]);
    }
    for (int i = 0; i < scheme.nn[2]; i++) {
        for (int index = 0; index < scheme.m; index++) values[index] = scheme.uvw[2][index][i];
        w.addExpression(values, scheme.m);
    }
}

kernel void initializeReducersKernel(device ReducerU *reducersU [[buffer(0)]], device ReducerV *reducersV [[buffer(1)]], device ReducerW *reducersW [[buffer(2)]], device SchemeInteger *schemes [[buffer(3)]], device RandomState *states [[buffer(4)]], constant int &count [[buffer(5)]], constant int &schemesCount [[buffer(6)]], constant int &seed [[buffer(7)]], device int *errors [[buffer(30)]], uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(count)) return;
    SchemeInteger scheme;
    loadObject(scheme, schemes);
    if (idx == 0) {
        copySchemeToReducers(reducersU[count],reducersV[count],reducersW[count],scheme);
    } else if (idx < uint(schemesCount)) {
        scheme.copyTo(scheme);
        storeObject(schemes + idx, scheme);
    }
    states[idx].value = uint(seed) ^ (0x9e3779b9u * (idx + 1));
    if (states[idx].value == 0) states[idx].value = 1;
    errors[idx] = !scheme.validate();
}

kernel void flipSchemesKernel(device SchemeInteger *schemes [[buffer(0)]], device RandomState *states [[buffer(1)]], constant int &schemesCount [[buffer(2)]], constant int &maxFlips [[buffer(3)]], device int *errors [[buffer(30)]], uint idx [[thread_position_in_grid]]) {
    if (idx == 0 || idx >= uint(schemesCount)) return;
    SchemeInteger scheme;
    RandomState state;
    loadObject(scheme, schemes);
    loadObject(state, states + idx);
    int flips = randomWord(&state) % (maxFlips + 1);
    scheme.copyTo(scheme);
    for (int i = 0; i < flips; i++) if (!scheme.tryFlip(state)) break;
    storeObject(schemes + idx, scheme);
    storeObject(states + idx, state);
    errors[idx] = !scheme.validate();
}

kernel void runReducersKernel(device ReducerU *reducersU [[buffer(0)]], device ReducerV *reducersV [[buffer(1)]], device ReducerW *reducersW [[buffer(2)]], device SchemeInteger *schemes [[buffer(3)]], device RandomState *states [[buffer(4)]], constant int &count [[buffer(5)]], constant int &schemesCount [[buffer(6)]], constant bool &independent [[buffer(7)]], device int *errors [[buffer(30)]], uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(count)) return;
    SchemeInteger scheme;
    RandomState state;
    device ReducerU &u = reducersU[idx];
    device ReducerV &v = reducersV[idx];
    device ReducerW &w = reducersW[idx];
    device ReducerU &uBest = reducersU[count];
    device ReducerV &vBest = reducersV[count];
    device ReducerW &wBest = reducersW[count];
    loadObject(scheme, schemes + idx % schemesCount);
    loadObject(state, states + idx);
    copySchemeToReducers(u,v,w,scheme);
    int modes[] = {
        GREEDY_INTERSECTIONS_MODE, GREEDY_INTERSECTIONS_MODE, GREEDY_INTERSECTIONS_MODE, GREEDY_INTERSECTIONS_MODE, GREEDY_INTERSECTIONS_MODE, GREEDY_INTERSECTIONS_MODE, GREEDY_INTERSECTIONS_MODE, GREEDY_INTERSECTIONS_MODE,
        GREEDY_ALTERNATIVE_MODE, GREEDY_ALTERNATIVE_MODE, GREEDY_ALTERNATIVE_MODE, GREEDY_ALTERNATIVE_MODE,
        GREEDY_RANDOM_MODE, GREEDY_RANDOM_MODE,
        MIX_MODE
    };

    int modesCount = sizeof(modes) / sizeof(modes[0]);

    u.setMode(idx == 0 ? 0 : modes[randomWord(&state) % modesCount]);
    v.setMode(idx == 0 ? 0 : modes[randomWord(&state) % modesCount]);
    w.setMode(idx == 0 ? 0 : modes[randomWord(&state) % modesCount]);

    if (independent) {
        if (randomUniform(&state) < 0.3f && uBest.getFreshVars() > 0)
            u.partialInitialize(uBest, 1 + randomWord(&state) % uBest.getFreshVars());

        if (randomUniform(&state) < 0.3f && vBest.getFreshVars() > 0)
            v.partialInitialize(vBest, 1 + randomWord(&state) % vBest.getFreshVars());

        if (randomUniform(&state) < 0.3f && wBest.getFreshVars() > 0)
            w.partialInitialize(wBest, 1 + randomWord(&state) % wBest.getFreshVars());
    }
    else if (randomUniform(&state) < 0.3f) {
        if (uBest.getFreshVars() > 0)
            u.partialInitialize(uBest, 1 + randomWord(&state) % uBest.getFreshVars());

        if (vBest.getFreshVars() > 0)
            v.partialInitialize(vBest, 1 + randomWord(&state) % vBest.getFreshVars());

        if (wBest.getFreshVars() > 0)
            w.partialInitialize(wBest, 1 + randomWord(&state) % wBest.getFreshVars());
    }

    u.reduce(state);
    v.reduce(state);
    w.reduce(state);
    storeObject(states + idx, state);
    errors[idx] = (!u.isValid() || !v.isValid() || !w.isValid()) ? 2 : 0;
}

#if defined(__METAL_VERSION__) && !defined(METAL_F2)
kernel void randomWalkCompactKernel(device Scheme *schemes [[buffer(0)]], device Scheme *schemesBest [[buffer(1)]], device int *bestRanks [[buffer(2)]], device int *flips [[buffer(3)]], device RandomState *states [[buffer(4)]], constant int &schemesCount [[buffer(5)]], constant int &maxIterations [[buffer(6)]], constant int &plusIterations [[buffer(7)]], constant float &reduceProbability [[buffer(8)]], constant float &expandProbability [[buffer(9)]], constant float &sandwichingProbability [[buffer(10)]], constant float &basisProbability [[buffer(11)]], constant bool &randomIterations [[buffer(12)]], device int *scratch [[buffer(13)]], device CompactStoredAddition *terms [[buffer(14)]], device uint32_t *pairs [[buffer(15)]], device int *errors [[buffer(30)]], uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(schemesCount)) return;
    if (maxIterations != 1000 || plusIterations != 1000000000 || randomIterations ||
        reduceProbability != 0 || expandProbability != 0 || sandwichingProbability != 0 || basisProbability != 0 ||
        schemes[idx].n[0] != 3 || schemes[idx].n[1] != 3 || schemes[idx].n[2] != 3 ||
        flips[idx] >= plusIterations - maxIterations) { errors[idx] = 2; return; }
    if (schemes[idx].m < 1 || schemes[idx].m > MAX_RANK) { errors[idx] = 3; return; }
    for (int p = 0; p < 3; p++) {
        if (schemes[idx].nn[p] != 9 || schemes[idx].flips[p].size > MAX_PAIRS) { errors[idx] = 3; return; }
        for (int r = 0; r < schemes[idx].m; r++) {
            device const Addition &term = schemes[idx].uvw[p][r];
            if (term.n != 9 || (term.values & ~T(511)) || (term.signs & ~T(511))) { errors[idx] = 3; return; }
        }
    }
    CompactScheme scheme;
    RandomState state;
    for (int i = 0; i < 3; i++) { scheme.n[i] = schemes[idx].n[i]; scheme.nn[i] = schemes[idx].nn[i]; }
    scheme.m = schemes[idx].m;
    uint tile = idx / 32, lane = idx % 32;
    scheme.scratch = scratch + tile * (MAX_PAIRS * 3 * 32) + lane;
    scheme.terms = terms + tile * (MAX_RANK * 3 * 32) + lane;
    for (int p = 0; p < 3; p++) {
        scheme.flips[p].pairs = pairs + tile * (MAX_PAIRS * 3 * 32) + p * MAX_PAIRS * 32 + lane;
        scheme.flips[p].size = schemes[idx].flips[p].size;
        for (size_t pair = 0; pair < scheme.flips[p].size; pair++)
            scheme.flips[p].pairs[pair * 32] = schemes[idx].flips[p].pairs[pair];
        for (int r = 0; r < scheme.m; r++) {
            CompactAddition packedInput(9);
            packedInput.values = ushort(schemes[idx].uvw[p][r].values);
            packedInput.signs = ushort(schemes[idx].uvw[p][r].signs);
            packedInput.valid = schemes[idx].uvw[p][r].valid;
            scheme.term(p, r) = packedInput;
        }
    }
    int savedRank = schemesBest[idx].m;
    loadObject(state, states + idx);
    int flipsCount = flips[idx];
    int bestRank = bestRanks[idx];
    int iterations = randomIterations ? randint(1, maxIterations, state) : maxIterations;

    for (int iteration = 0; iteration < iterations; iteration++) {
        int rank = scheme.m;

        if (!scheme.tryFlip(state)) {
            scheme.tryExpand(randint(1, 2, state), state);
            continue;
        }

        if (scheme.m < rank)
            flipsCount = 0;

        flipsCount++;

        if (scheme.m < bestRank || (scheme.m == bestRank && randomUniform(&state) < 0.01f)) {
            bestRank = scheme.m;
            schemesBest[idx].m = scheme.m;
            for (int p = 0; p < 3; p++) {
                schemesBest[idx].n[p] = scheme.n[p];
                schemesBest[idx].nn[p] = scheme.nn[p];
                for (int r = 0; r < scheme.m; r++) {
                    CompactAddition packedOutput = scheme.term(p, r).read();
                    schemesBest[idx].uvw[p][r].n = packedOutput.n;
                    schemesBest[idx].uvw[p][r].values = packedOutput.values;
                    schemesBest[idx].uvw[p][r].signs = packedOutput.signs;
                    schemesBest[idx].uvw[p][r].valid = packedOutput.valid;
                }
                schemesBest[idx].flips[p].size = scheme.flips[p].size;
                for (size_t pair = 0; pair < scheme.flips[p].size; pair++)
                    schemesBest[idx].flips[p].pairs[pair] = scheme.flips[p].pairs[pair * 32];
            }
            savedRank = scheme.m;
        }

        if (randomUniform(&state) * maxIterations < reduceProbability)
            scheme.tryReduce();

        if (randomUniform(&state) * maxIterations < expandProbability && scheme.m <= savedRank + 2) {
            if (scheme.tryExpand(randint(1, 2, state), state))
                flipsCount = 0;
        }

#ifdef METAL_F2
        if (randomUniform(&state) * maxIterations < sandwichingProbability) scheme.sandwiching(state);
#else
        randomUniform(&state);
#endif

        randomUniform(&state);

        if (flipsCount >= plusIterations && scheme.tryPlus(state))
            flipsCount = 0;
    }

    flips[idx] = flipsCount;
    bestRanks[idx] = bestRank;

    schemes[idx].m = scheme.m;
    for (int p = 0; p < 3; p++) {
        schemes[idx].n[p] = scheme.n[p];
        schemes[idx].nn[p] = scheme.nn[p];
        for (int r = 0; r < scheme.m; r++) {
            CompactAddition packedOutput = scheme.term(p, r).read();
            schemes[idx].uvw[p][r].n = packedOutput.n;
            schemes[idx].uvw[p][r].values = packedOutput.values;
            schemes[idx].uvw[p][r].signs = packedOutput.signs;
            schemes[idx].uvw[p][r].valid = packedOutput.valid;
        }
        schemes[idx].flips[p].size = scheme.flips[p].size;
        for (size_t pair = 0; pair < scheme.flips[p].size; pair++)
            schemes[idx].flips[p].pairs[pair] = scheme.flips[p].pairs[pair * 32];
    }

    storeObject(states + idx, state);
    errors[idx] = !scheme.validate();
}

#endif
