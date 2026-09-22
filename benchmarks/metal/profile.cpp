#include <algorithm>
#include <chrono>
#include <filesystem>
#include <numeric>
#include <vector>
#include <omp.h>
#include <notify.h>
#include <unistd.h>
#include <cerrno>
#include "core.h"
#include "addition.h"
#include "flip_set.h"
#include "scheme_integer.h"
#include "scheme_z2.h"
#include "pairs_counter.h"
#include "additions_reducer.h"
#include "runtime.h"
using uint = unsigned int;
using uchar = unsigned char;
#define thread
#define device
#define kernel
#include <kernels.metal>
#include <profile_trace.h>
#include <profile_record.h>
#undef thread
#undef device
#undef kernel

using Clock = std::chrono::steady_clock;

double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

int number(const char *text) {
    size_t length;
    int value = std::stoi(text, &length);
    if (text[length]) throw std::runtime_error("invalid integer argument");
    return value;
}

int main(int argc, char **argv) {
    try {
        if (argc < 5 || argc > 8) throw std::runtime_error("usage: profile schemes cpu-threads seed output-directory [matched|phases|repeat|replay|permutation|selected|dispatch|population|capture] [block-size] [fixture]");
        std::string mode = argc >= 6 ? argv[5] : "matched";
        int blockSize = argc >= 7 ? number(argv[6]) : 32;
        if (blockSize < 1 || blockSize > 1024) throw std::runtime_error("expected block size 1..1024; the pipeline may impose a smaller limit");
        bool dispatch = mode == "dispatch";
        bool population = mode == "population";
        bool capture = mode == "capture";
        if (dispatch && (argc < 7 || blockSize == 32))
            throw std::runtime_error("dispatch comparison requires an explicit block size different from 32");
        bool replay = mode == "replay";
        const char *variantKernel = nullptr;
        if (replay) variantKernel = "profileReplayKernel";
        else if (mode == "permutation") variantKernel = "profileNarrowKernel";
        else if (mode == "selected") variantKernel = "profileSelectedKernel";
        bool phases = mode == "phases", repeat = mode == "repeat" || variantKernel || dispatch || population || capture;
        if (mode != "matched" && !phases && !repeat) throw std::runtime_error("unknown profiling mode");
        int count = number(argv[1]), threads = number(argv[2]), seed = number(argv[3]);
        if (count < 1 || count > 4096 || threads < 1 || threads > 8 || seed < 1)
            throw std::runtime_error("expected 1..4096 schemes, 1..8 CPU threads and positive seed");
        if (population && (count < 512 || (count & (count - 1))))
            throw std::runtime_error("population comparison requires a power of two from 512 through 4096");
        std::filesystem::path output(argv[4]);
        if (std::filesystem::exists(output) && !std::filesystem::is_empty(output))
            throw std::runtime_error("output directory must be empty");
        std::filesystem::create_directories(output);
        std::vector<Scheme> cpu(count), cpuBest(count);
        std::vector<RandomState> cpuStates(count);
        std::vector<int> cpuRanks(count), cpuFlips(count), cpuErrors(count);
        Scheme *gpu, *gpuBest;
        RandomState *gpuStates;
        int *gpuRanks, *gpuFlips;
        metalAllocate(&gpu, count * sizeof(Scheme));
        metalAllocate(&gpuBest, count * sizeof(Scheme));
        metalAllocate(&gpuStates, count * sizeof(RandomState));
        metalAllocate(&gpuRanks, count * sizeof(int));
        metalAllocate(&gpuFlips, count * sizeof(int));
        Scheme initial;
        initial.initializeNaive(3, 3, 3);
        if (argc == 8) {
            std::ifstream fixture(argv[7]);
            if (!fixture || !initial.read(fixture) || initial.n[0] != 3 || initial.n[1] != 3 || initial.n[2] != 3)
                throw std::runtime_error("expected a valid signed 3x3 input fixture");
            fixture >> std::ws;
            if (!fixture.eof()) throw std::runtime_error("unexpected content after fixture scheme");
        }
        for (int i = 0; i < count; i++) {
            cpu[i] = initial;
            cpuBest[i] = cpu[i];
            cpuRanks[i] = cpu[i].m;
            cpuStates[i].value = uint(seed) ^ (0x9e3779b9u * (i + 1));
            if (!cpuStates[i].value) cpuStates[i].value = 1;
            gpu[i] = cpu[i]; gpuBest[i] = cpuBest[i];
            gpuStates[i] = cpuStates[i]; gpuRanks[i] = cpuRanks[i]; gpuFlips[i] = 0;
        }
        std::cout << "CONFIG " << count << ' ' << threads << ' ' << seed << " SCHEME_BYTES " << sizeof(Scheme) << " BLOCK_SIZE " << blockSize << " INITIAL_RANK " << initial.m << '\n';
        int iterations = 1000, plusIterations = 1000000000;
        float probability = 0;
        bool randomIterations = false;
        auto check = [&](int checkedCount) {
            for (int i = 0; i < checkedCount; i++) {
                if (cpuErrors[i]) throw std::runtime_error("CPU tensor validation failed");
                if (cpuStates[i].value != gpuStates[i].value || cpuRanks[i] != gpuRanks[i] || cpuFlips[i] != gpuFlips[i])
                    throw std::runtime_error("CPU/Metal walk counters or random state differ");
                for (int which = 0; which < 2; which++) {
                    const Scheme &a = which ? cpuBest[i] : cpu[i];
                    const Scheme &b = which ? gpuBest[i] : gpu[i];
                    if (a.m != b.m) throw std::runtime_error("CPU/Metal rank differs");
                    for (int p = 0; p < 3; p++) {
                        if (a.n[p] != b.n[p] || a.nn[p] != b.nn[p] || a.flips[p].size != b.flips[p].size)
                            throw std::runtime_error("CPU/Metal shape differs");
                        for (int r = 0; r < a.m; r++)
                            if (a.uvw[p][r] != b.uvw[p][r] || a.uvw[p][r].valid != b.uvw[p][r].valid || a.uvw[p][r].n != b.uvw[p][r].n)
                                throw std::runtime_error("CPU/Metal coefficients differ");
                        for (size_t j = 0; j < a.flips[p].size; j++)
                            if (a.flips[p].pairs[j] != b.flips[p].pairs[j])
                                throw std::runtime_error("CPU/Metal candidate ordering differs");
                    }
                }
            }
        };
        for (int round = 0; round < 6; round++) {
            auto start = Clock::now();
            #pragma omp parallel for num_threads(threads)
            for (int i = 0; i < count; i++)
                randomWalkKernel(cpu.data(), cpuBest.data(), cpuRanks.data(), cpuFlips.data(), cpuStates.data(), count, iterations, plusIterations, probability, probability, probability, probability, randomIterations, cpuErrors.data(), i);
            double cpuTime = seconds(start);
            if (repeat) {
                for (int error : cpuErrors)
                    if (error) throw std::runtime_error("CPU tensor validation failed during preparation");
                std::cout << "PREPARE " << round << " CPU " << cpuTime << '\n';
                continue;
            }
            start = Clock::now();
            metalDispatch("randomWalkKernel", count, blockSize, gpu, gpuBest, gpuRanks, gpuFlips, gpuStates, count, iterations, plusIterations, probability, probability, probability, probability, randomIterations);
            double gpuTime = seconds(start);
            check(count);
            std::cout << "ROUND " << round << " CPU " << cpuTime << " METAL_WALL " << gpuTime << " MATCH\n";
        }
        if (repeat) {
            auto input = cpu, inputBest = cpuBest;
            auto inputStates = cpuStates;
            auto inputRanks = cpuRanks, inputFlips = cpuFlips;
            ProfileDecision *decisions = nullptr;
            if (replay) metalAllocate(&decisions, count * iterations * sizeof(ProfileDecision));
            #pragma omp parallel for num_threads(threads)
            for (int i = 0; i < count; i++) {
                if (replay)
                    profileRecordKernel(cpu.data(), cpuBest.data(), cpuRanks.data(), cpuFlips.data(), cpuStates.data(), count, iterations, plusIterations, probability, probability, probability, probability, randomIterations, decisions, cpuErrors.data(), i);
                else
                    randomWalkKernel(cpu.data(), cpuBest.data(), cpuRanks.data(), cpuFlips.data(), cpuStates.data(), count, iterations, plusIterations, probability, probability, probability, probability, randomIterations, cpuErrors.data(), i);
            }
            if (replay) {
                size_t selected = 0;
                for (int i = 0; i < count * iterations; i++) selected += decisions[i].operation != 0xffffffffu;
                std::cout << "RECORDED " << selected << " SELECTED " << count * iterations - selected << " RECOVERIES\n";
            }
            std::vector<int> populations;
            if (population)
                for (int size = 512; size <= count; size *= 2) populations.push_back(size);
            int variants = population ? int(populations.size()) : variantKernel || dispatch ? 2 : 1;
            int notification = -1;
            if (capture) {
                std::copy(input.begin(), input.end(), gpu);
                std::copy(inputBest.begin(), inputBest.end(), gpuBest);
                std::copy(inputStates.begin(), inputStates.end(), gpuStates);
                std::copy(inputRanks.begin(), inputRanks.end(), gpuRanks);
                std::copy(inputFlips.begin(), inputFlips.end(), gpuFlips);
                metalDispatch("randomWalkKernel", count, blockSize, gpu, gpuBest, gpuRanks, gpuFlips, gpuStates, count, iterations, plusIterations, probability, probability, probability, probability, randomIterations);
                check(count);
                std::string name = "org.flipgraphgpu.capture." + std::to_string(getpid());
                int descriptor = -1;
                if (notify_register_file_descriptor(name.c_str(), &descriptor, 0, &notification) != NOTIFY_STATUS_OK)
                    throw std::runtime_error("cannot register capture notification");
                std::cout << "CAPTURE_READY " << getpid() << ' ' << name << std::endl;
                uint32_t delivered;
                ssize_t received;
                do { received = read(descriptor, &delivered, sizeof(delivered)); } while (received < 0 && errno == EINTR);
                close(descriptor);
                notify_cancel(notification);
                if (received != sizeof(delivered)) throw std::runtime_error("capture notification was not delivered");
            }
            for (int sample = 0; sample < (capture ? 3 : 13); sample++) {
                for (int variant = 0; variant < variants; variant++) {
                    std::copy(input.begin(), input.end(), gpu);
                    std::copy(inputBest.begin(), inputBest.end(), gpuBest);
                    std::copy(inputStates.begin(), inputStates.end(), gpuStates);
                    std::copy(inputRanks.begin(), inputRanks.end(), gpuRanks);
                    std::copy(inputFlips.begin(), inputFlips.end(), gpuFlips);
                    bool useVariant = variantKernel && (variant + sample) % 2;
                    bool useReplay = replay && useVariant;
                    const char *name = useVariant ? variantKernel : "randomWalkKernel";
                    int dispatchBlockSize = dispatch && (variant + sample) % 2 == 0 ? 32 : blockSize;
                    int dispatchCount = population ? populations[(variant + (sample == 12 ? 0 : sample)) % populations.size()] : count;
                    auto start = Clock::now();
                    if (useReplay)
                        metalDispatch(name, dispatchCount, dispatchBlockSize, gpu, gpuBest, gpuRanks, gpuFlips, gpuStates, dispatchCount, iterations, plusIterations, probability, probability, probability, probability, randomIterations, decisions);
                    else
                        metalDispatch(name, dispatchCount, dispatchBlockSize, gpu, gpuBest, gpuRanks, gpuFlips, gpuStates, dispatchCount, iterations, plusIterations, probability, probability, probability, probability, randomIterations);
                    double gpuTime = seconds(start);
                    check(dispatchCount);
                    std::cout << (population ? "POPULATION " : dispatch ? "DISPATCH " : "REPEAT ") << sample << ' ' << name;
                    if (population) std::cout << " SCHEMES " << dispatchCount;
                    if (dispatch) std::cout << " BLOCK_SIZE " << dispatchBlockSize;
                    std::cout << " METAL_WALL " << gpuTime << " MATCH\n";
                }
            }
            if (capture) {
                std::cout << "CAPTURE_DONE" << std::endl;
                std::string command;
                if (!std::getline(std::cin, command) || command != "export")
                    throw std::runtime_error("capture export was not released");
            }
            if (decisions) metalFree(decisions);
        }
        for (int i = 0; i < count; i++) gpuBest[i].save((output / (std::to_string(i) + ".json")).string());
        if (phases) {
            uint *checksums;
            metalAllocate(&checksums, count * sizeof(uint));
            std::vector<int> candidates(count);
            for (int i = 0; i < count; i++)
                candidates[i] = cpu[i].flips[0].size + cpu[i].flips[1].size + cpu[i].flips[2].size;
            std::cout << "CANDIDATES " << *std::min_element(candidates.begin(), candidates.end()) << ' '
                      << std::accumulate(candidates.begin(), candidates.end(), 0.0) / count << ' '
                      << *std::max_element(candidates.begin(), candidates.end()) << '\n';
            for (int repeat = 0; repeat < 6; repeat++) {
                std::copy(cpuStates.begin(), cpuStates.end(), gpuStates);
                metalDispatch("profileCopyKernel", count, blockSize, gpu, gpuBest, count);
                metalDispatch("profileValidateKernel", count, blockSize, gpu, count);
                metalDispatch("profilePermutationKernel", count, blockSize, gpu, gpuStates, checksums, count, iterations);
                std::copy(cpuStates.begin(), cpuStates.end(), gpuStates);
                metalDispatch("profileSelectionKernel", count, blockSize, gpu, gpuStates, checksums, count, iterations);
            }
            auto input = cpu, inputBest = cpuBest;
            auto inputStates = cpuStates;
            auto inputRanks = cpuRanks, inputFlips = cpuFlips;
            std::vector<uint> trace(2*count);
            #pragma omp parallel for num_threads(threads)
            for (int i = 0; i < count; i++)
                profileTraceKernel(cpu.data(), cpuBest.data(), cpuRanks.data(), cpuFlips.data(), cpuStates.data(), count, iterations, plusIterations, probability, probability, probability, probability, randomIterations, trace.data(), cpuErrors.data(), i);
            double totalPairs = 0;
            uint maxPairs = 0;
            for (int i = 0; i < count; i++) {
                totalPairs += trace[2*i];
                maxPairs = std::max(maxPairs, trace[2*i+1]);
            }
            std::cout << "WALK_CANDIDATES " << totalPairs / count / iterations << ' ' << maxPairs << '\n';
            for (int repeat = 0; repeat < 6; repeat++) {
                for (int variant = 0; variant < 2; variant++) {
                    std::copy(input.begin(), input.end(), gpu);
                    std::copy(inputBest.begin(), inputBest.end(), gpuBest);
                    std::copy(inputStates.begin(), inputStates.end(), gpuStates);
                    std::copy(inputRanks.begin(), inputRanks.end(), gpuRanks);
                    std::copy(inputFlips.begin(), inputFlips.end(), gpuFlips);
                    const char *name = (variant + repeat) % 2 ? "profileWalkKernel" : "randomWalkKernel";
                    std::cout << "FROZEN " << repeat << ' ' << name << '\n';
                    metalDispatch(name, count, blockSize, gpu, gpuBest, gpuRanks, gpuFlips, gpuStates, count, iterations, plusIterations, probability, probability, probability, probability, randomIterations);
                    metalDispatch("profileValidateKernel", count, blockSize, gpu, count);
                    check(count);
                }
            }
            metalFree(checksums);
        }
        metalFree(gpu); metalFree(gpuBest); metalFree(gpuStates); metalFree(gpuRanks); metalFree(gpuFlips);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
