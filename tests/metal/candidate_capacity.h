#pragma once

inline void capacityRequire(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Action> void expectCapacityError(Action action) {
    bool rejected = false;
    try { action(); }
    catch (const std::runtime_error &error) {
        rejected = std::string(error.what()).find("flip candidate capacity exceeded") != std::string::npos;
        if (!rejected) throw;
    }
    capacityRequire(rejected, "expected explicit flip candidate capacity rejection");
}

inline void checkCandidateCapacityAndLayout() {
    using U = AdditionsReducer<350,250,32,2016>;
    using W = AdditionsReducer<64,500,175,61075>;
    const unsigned expected[] = {
        sizeof(Addition), sizeof(Scheme), sizeof(U), sizeof(W),
        alignof(Addition), __builtin_offsetof(Addition, values),
        __builtin_offsetof(Addition, signs), __builtin_offsetof(Addition, valid),
        alignof(Scheme), __builtin_offsetof(Scheme, uvw), __builtin_offsetof(Scheme, flips),
        sizeof(FlipSet), alignof(FlipSet), __builtin_offsetof(FlipSet, size),
        __builtin_offsetof(FlipSet, overflow), __builtin_offsetof(FlipSet, pairs)
    };
    unsigned *output, *storage;
    metalAllocate(&output, sizeof(expected));
    metalAllocate(&storage, MAX_PAIRS * 32 * sizeof(unsigned));
    metalDispatch("layoutKernel", 1, 1, output);
    for (size_t i = 0; i < sizeof(expected)/sizeof(*expected); i++)
        capacityRequire(output[i] == expected[i], "CPU/Metal size, alignment or member offset mismatch");
    metalDispatch("candidateCapacityKernel", 1, 1, output, storage);
#ifdef METAL_F2
    const int results = 2;
#else
    const int results = 4;
#endif
    for (int i = 0; i < results; i++)
        capacityRequire(output[i] == 1, "GPU candidate boundary or persistent overflow mismatch");
    metalFree(output); metalFree(storage);

    FlipSet list;
    for (unsigned i = 0; i < MAX_PAIRS; i++) list.add(0, i);
    capacityRequire(list.size == MAX_PAIRS && !list.overflow, "500 candidates must fit");
    list.add(1, 2);
    list.remove(0, 0); list.clear(); list.add(2, 3);
    capacityRequire(list.size == 1 && list.overflow, "removal/rebuild lost overflow");
    Scheme rejected, copied;
    rejected.initializeNaive(6, 6, 6);
    capacityRequire(candidateOverflow(rejected), "naive 6x6 must exceed candidate capacity");
    capacityRequire(rejected.validate(), "capacity failure must be distinct from tensor failure");
    std::ostringstream encoded;
    encoded << "6 6 6 216\n";
    for (int p = 0; p < 3; p++)
        for (int r = 0; r < rejected.m; r++)
            for (int c = 0; c < rejected.nn[p]; c++) {
#ifdef METAL_F2
                encoded << ((rejected.uvw[p][r] >> c) & 1) << ' ';
#else
                encoded << rejected.uvw[p][r][c] << ' ';
#endif
            }
    std::istringstream input(encoded.str());
    Scheme loaded;
    expectCapacityError([&] { loaded.read(input, false); });
    for (const char *bad : {"8 8 8 351", "9 8 8 1", "3 3 3 1 2", "3 3"}) {
        Scheme malformed;
        std::istringstream text(bad);
        capacityRequire(!malformed.read(text), "malformed loaded scheme accepted");
    }
    rejected.copyTo(copied);
    capacityRequire(candidateOverflow(copied), "copy lost overflow");
    // A subsequent smaller rebuild cannot make this walk's earlier truncation safe.
    rejected.initializeNaive(3, 3, 3);
    capacityRequire(candidateOverflow(rejected), "rebuild lost overflow");

    Scheme *gpu, *best;
    RandomState *states;
    int *ranks, *counts;
    metalAllocate(&gpu, sizeof(Scheme)); metalAllocate(&best, sizeof(Scheme));
    metalAllocate(&states, sizeof(RandomState));
    metalAllocate(&ranks, sizeof(int)); metalAllocate(&counts, sizeof(int));
    expectCapacityError([&] { metalDispatch("initializeNaiveKernel", 1, 1, gpu, 1, 6, 6, 6); });
    // Test host/device flag transfer and rejection before a truncated walk can be accepted.
    *gpu = rejected; *best = rejected; *states = {7}; *ranks = 27; *counts = 0;
    expectCapacityError([&] { metalDispatch("randomWalkKernel", 1, 1, gpu, best, ranks, counts,
        states, 1, 1, 1000000000, 0.0f, 0.0f, 0.0f, 0.0f, false); });
#ifndef METAL_F2
    expectCapacityError([&] { metalDispatch("randomWalkCompactKernel", 1, 32, gpu, best, ranks, counts,
        states, 1, 1000, 1000000000, 0.0f, 0.0f, 0.0f, 0.0f, false); });
#endif
    Scheme fresh;
    fresh.initializeNaive(3, 3, 3);
    *gpu = fresh;
    metalDispatch("initializeSchemesKernel", 1, 1, gpu, best, ranks, counts, 8, 8, 8, 1);
    capacityRequire(*ranks == 27, "loaded header must determine the initial rank bound");
    metalFree(gpu); metalFree(best); metalFree(states); metalFree(ranks); metalFree(counts);
    std::cout << "PASS: candidate capacity, persistent overflow, dispatch rejection and member layouts\n";
}
