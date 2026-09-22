import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

REDUCTION_COMMIT = 'a3ee98a5e47ad747307610bce7d3f47afd9cf0d3'
SELECTION_COMMIT = '987562e99375999b82edae8d4bb307c7c19fc63b'


def function_end(text, start):
    begin = text.index('{', start)
    depth, end = 1, begin + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return end


def replace_once(text, original, replacement):
    if text.count(original) != 1:
        raise ValueError('source no longer matches: ' + original)
    return text.replace(original, replacement)


def transform(source, lazy, sign, upstream):
    if lazy:
        begin = source.index('bool SchemeInteger::tryFlip(')
        end = function_end(source, begin)
        flip = source[begin:end]
        flip = replace_once(flip, '    randomPermutation(indices, size, state);',
                            '    for (int index = 0; index < size; index++) indices[index] = index;')
        flip = replace_once(flip, '        int index = indices[p];',
                            '        int index = algorithmCandidate(indices, p, size, randomWord(&state));')
        source = source[:begin] + flip + source[end:]
    if sign:
        source = replace_once(source, 'private:\n', 'private:\n    bool checkFlipReduce(int i, int j, int index1, int index2, int sign) LOCAL_METHOD;\n')
        begin = upstream.index('bool TernaryScheme<T>::checkFlipReduce(')
        helper = upstream[begin:function_end(upstream, begin)]
        helper = helper.replace('TernaryScheme<T>::', 'SchemeInteger::').replace('int sign) {', 'int sign) LOCAL_METHOD {')
        begin = source.index('void SchemeInteger::flip(')
        end = function_end(source, begin)
        flip = source[begin:end]
        loop = flip.index('    bool useMasks =')
        flip = flip[:loop] + '''    for (int index = 0; index < m; index++) {
        if (index != index1) {
            int cmp = uvw[j][index].compare(uvw[j][index1]);
            if (checkReduce && cmp != 0 && checkFlipReduce(i, k, index, index1, cmp)) return;
            if (cmp == 1) flips[j].add(index1, index);
        }
        if (index != index2) {
            int cmp = uvw[k][index].compare(uvw[k][index2]);
            if (checkReduce && cmp != 0 && checkFlipReduce(i, j, index, index2, cmp)) return;
            if (cmp == 1) flips[k].add(index2, index);
        }
    }
}'''
        source = source[:begin] + flip + source[end:] + '\n' + helper + '\n'
    return source


CANDIDATE = '''
inline int algorithmCandidate(LOCAL int *indices, int p, int size, uint32_t word) {
    int q = p + word % (size - p);
    int index = indices[q];
    indices[q] = indices[p];
    indices[p] = index;
    return index;
}
'''


def quality_kernel(kernels):
    begin = kernels.index('kernel void randomWalkKernel(')
    end = function_end(kernels, begin)
    walk = kernels[begin:end].replace('randomWalkKernel(', 'qualityWalkKernel(', 1)
    walk = replace_once(walk, 'device int *errors [[buffer(30)]]',
                        'device uint32_t *metrics [[buffer(13)]], device int *errors [[buffer(30)]]')
    walk = replace_once(walk, '    for (int iteration = 0; iteration < iterations; iteration++) {',
                        '    uint32_t successful = 0, reductions = 0, removed = 0;\n    for (int iteration = 0; iteration < iterations; iteration++) {')
    change = 'if (scheme.m < rank) { reductions++; removed += rank - scheme.m; }'
    walk = replace_once(walk, '            continue;', '            ' + change + '\n            continue;')
    walk = replace_once(walk, '        flipsCount++;', '        flipsCount++;\n        successful++;')
    walk = replace_once(walk, '            flipsCount = 0;\n    }', '            flipsCount = 0;\n        ' + change + '\n    }')
    walk = replace_once(walk, '    errors[idx] = !scheme.validate();',
                        '    metrics[3 * idx] = successful;\n    metrics[3 * idx + 1] = reductions;\n    metrics[3 * idx + 2] = removed;\n    errors[idx] = !scheme.validate();')
    return walk


QUALITY = r'''#include <algorithm>
#include <chrono>
#include <filesystem>
#include <vector>
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
#include "kernels.metal"
#undef thread
#undef device
#undef kernel

int integer(const char *text) {
    size_t end;
    int result = std::stoi(text, &end);
    if (text[end]) throw std::runtime_error("invalid integer");
    return result;
}

void compare(const Scheme &a, const Scheme &b) {
    if (a.m != b.m) throw std::runtime_error("rank mismatch");
    for (int p = 0; p < 3; p++) {
        if (a.n[p] != b.n[p] || a.nn[p] != b.nn[p] || a.flips[p].size != b.flips[p].size)
            throw std::runtime_error("shape mismatch");
        for (int r = 0; r < a.m; r++) {
            const Addition &x = a.uvw[p][r], &y = b.uvw[p][r];
            if (x.n != y.n || x.values != y.values || x.signs != y.signs || x.valid != y.valid)
                throw std::runtime_error("coefficient mismatch");
        }
        for (size_t r = 0; r < a.flips[p].size; r++)
            if (a.flips[p].pairs[r] != b.flips[p].pairs[r]) throw std::runtime_error("pair ordering mismatch");
    }
}

int main(int argc, char **argv) {
    try {
        if (argc != 7) throw std::runtime_error("usage: quality count seed rounds work|dispatch|wall budget-ms output");
        int count = integer(argv[1]), seed = integer(argv[2]), rounds = integer(argv[3]), budget = integer(argv[5]);
        std::string mode = argv[4];
        if (mode != "work" && mode != "dispatch" && mode != "wall") throw std::runtime_error("unknown budget mode");
        if ((mode == "work") != (budget == 0)) throw std::runtime_error("work requires zero budget; dispatch/wall require positive budget");
        if (count < 1 || count > 2048 || seed < 1 || rounds < 1 || rounds > 33 || budget < 0 || budget > 20000)
            throw std::runtime_error("expected count1..2048, positive seed, rounds1..33 and budget0..20000ms");
        std::filesystem::path output(argv[6]);
        if (std::filesystem::exists(output)) throw std::runtime_error("output must not exist");
        std::filesystem::create_directories(output);
        std::vector<Scheme> cpu(count), best(count);
        std::vector<RandomState> states(count);
        std::vector<int> ranks(count, 27), flips(count), errors(count);
        std::vector<uint32_t> metrics(3 * count);
        Scheme *gpu, *gpuBest;
        RandomState *gpuStates;
        int *gpuRanks, *gpuFlips;
        uint32_t *gpuMetrics;
        metalAllocate(&gpu, count * sizeof(Scheme));
        metalAllocate(&gpuBest, count * sizeof(Scheme));
        metalAllocate(&gpuStates, count * sizeof(RandomState));
        metalAllocate(&gpuRanks, count * sizeof(int));
        metalAllocate(&gpuFlips, count * sizeof(int));
        metalAllocate(&gpuMetrics, 3 * count * sizeof(uint32_t));
        for (int i = 0; i < count; i++) {
            cpu[i].initializeNaive(3, 3, 3);
            best[i] = cpu[i];
            states[i].value = uint(seed) ^ (0x9e3779b9u * (i + 1));
            if (!states[i].value) states[i].value = 1;
            gpu[i] = cpu[i]; gpuBest[i] = best[i]; gpuStates[i] = states[i];
            gpuRanks[i] = 27; gpuFlips[i] = 0;
        }
        int iterations = 1000, plusIterations = 1000000000;
        float probability = 0;
        bool randomIterations = false;
        auto dispatch = [&] {
            metalDispatch("qualityWalkKernel", count, 32, gpu, gpuBest, gpuRanks, gpuFlips, gpuStates, count,
                          1000, 1000000000, 0.0f, 0.0f, 0.0f, 0.0f, false, gpuMetrics);
        };
        auto cpuRound = [&] {
            for (int i = 0; i < count; i++)
                qualityWalkKernel(cpu.data(), best.data(), ranks.data(), flips.data(), states.data(), count,
                    iterations, plusIterations, probability, probability, probability, probability, randomIterations, metrics.data(), errors.data(), i);
            for (int error : errors) if (error) throw std::runtime_error("CPU tensor validation failed during replay");
        };
        auto checkState = [&] {
            for (int i = 0; i < count; i++) {
                if (errors[i] || states[i].value != gpuStates[i].value || ranks[i] != gpuRanks[i] || flips[i] != gpuFlips[i])
                    throw std::runtime_error("CPU/Metal state or validity mismatch");
                compare(cpu[i], gpu[i]); compare(best[i], gpuBest[i]);
            }
        };
        auto warmupStart = std::chrono::steady_clock::now();
        dispatch();
        double warmupSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - warmupStart).count();
        for (int i = 0; i < count; i++) {
            gpu[i] = cpu[i]; gpuBest[i] = best[i]; gpuStates[i] = states[i];
            gpuRanks[i] = ranks[i]; gpuFlips[i] = flips[i];
        }
        std::vector<uint64_t> gpuTotals(3 * count), cpuTotals(3 * count);
        double dispatchSeconds = 0, maxRoundSeconds = 0;
        int completed = 0;
        auto wallStart = std::chrono::steady_clock::now();
        for (int round = 0; round < rounds; round++) {
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - wallStart).count();
            if ((mode == "dispatch" && dispatchSeconds * 1000 >= budget) || (mode == "wall" && elapsed * 1000 >= budget)) break;
            auto roundStart = std::chrono::steady_clock::now();
            if (mode != "wall") cpuRound();
            auto start = std::chrono::steady_clock::now();
            dispatch();
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            dispatchSeconds += seconds;
            for (int i = 0; i < 3 * count; i++) {
                gpuTotals[i] += gpuMetrics[i];
                if (mode != "wall") {
                    if (metrics[i] != gpuMetrics[i]) throw std::runtime_error("metric mismatch");
                    cpuTotals[i] += metrics[i];
                }
            }
            if (mode != "wall") checkState();
            maxRoundSeconds = std::max(maxRoundSeconds, std::chrono::duration<double>(std::chrono::steady_clock::now() - roundStart).count());
            completed++;
            std::cout << "ROUND " << round << " METAL_WALL " << seconds << (mode == "wall" ? " REPLAY_PENDING" : " MATCH") << std::endl;
        }
        double wallSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - wallStart).count();
        double replaySeconds = 0;
        if (mode == "wall") {
            auto replayStart = std::chrono::steady_clock::now();
            for (int round = 0; round < completed; round++) {
                cpuRound();
                for (int i = 0; i < 3 * count; i++) cpuTotals[i] += metrics[i];
            }
            checkState();
            if (cpuTotals != gpuTotals) throw std::runtime_error("replayed quality metric mismatch");
            replaySeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - replayStart).count();
            std::cout << "REPLAY " << completed << " MATCH" << std::endl;
        }
        uint64_t successful = 0, reductions = 0, removed = 0;
        for (int i = 0; i < count; i++) {
            successful += gpuTotals[3 * i]; reductions += gpuTotals[3 * i + 1]; removed += gpuTotals[3 * i + 2];
        }
        bool budgetReached = mode != "work" && (mode == "wall" ? wallSeconds : dispatchSeconds) * 1000 >= budget;
        const char *stopReason = mode == "work" ? "fixed_work" : budgetReached ? "budget_reached" : "round_cap";
        for (int i = 0; i < count; i++) gpuBest[i].save((output / (std::to_string(i) + ".json")).string());
        std::cout << "QUALITY ROUNDS " << completed << " SUCCESSFUL_FLIPS " << successful << " RANK_DECREASE_EVENTS "
                  << reductions << " TERMS_REMOVED " << removed << " BEST_RANK " << *std::min_element(ranks.begin(), ranks.end())
                  << " GPU_DISPATCH_SECONDS " << dispatchSeconds << " WALL_SECONDS " << wallSeconds
                  << " MAX_ROUND_SECONDS " << maxRoundSeconds << " BUDGET_MODE " << mode
                  << " WARMUP_SECONDS " << warmupSeconds << " CPU_REPLAY_SECONDS " << replaySeconds
                  << " STOP_REASON " << stopReason << " BUDGET_REACHED " << budgetReached
                  << " OVERSHOOT_SECONDS " << (mode == "work" ? 0.0 : std::max(0.0, (mode == "wall" ? wallSeconds : dispatchSeconds) - budget / 1000.0)) << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
'''


def generate(source, output, cpu_source):
    upstream = subprocess.run(['git', '-C', str(cpu_source), 'show', REDUCTION_COMMIT + ':src/schemes/ternary_scheme.hpp'],
                              check=True, capture_output=True, text=True).stdout
    license_text = subprocess.run(['git', '-C', str(cpu_source), 'show', 'HEAD:LICENSE'],
                                  check=True, capture_output=True, text=True).stdout
    output.mkdir(parents=True, exist_ok=False)
    commands = []
    for name, lazy, sign in [('original', False, False), ('lazy', True, False), ('sign', False, True), ('both', True, True)]:
        directory = output / name
        shutil.copytree(source, directory)
        scheme = directory / 'scheme_integer.h'
        scheme.write_text(transform(scheme.read_text(), lazy, sign, upstream))
        if lazy:
            core = directory / 'core.h'
            core.write_text(core.read_text() + CANDIDATE)
        kernels = directory / 'kernels.metal'
        text = kernels.read_text()
        kernels.write_text(text + '\n' + quality_kernel(text))
        (directory / 'quality.cpp').write_text(QUALITY)
        (directory / 'UPSTREAM_LICENSE').write_text(license_text)
        commands.extend([
            ['xcrun', 'clang++', '-std=c++17', '-O3', '-ffp-contract=off', '-Wno-unknown-attributes', '-I' + str(directory),
             '-c', str(directory / 'quality.cpp'), '-o', str(directory / 'quality.o')],
            ['xcrun', 'clang++', '-std=c++17', '-O2', '-fobjc-arc', '-ffp-contract=off', '-framework', 'Foundation', '-framework', 'Metal',
             '-DMETAL_SOURCE_DIR="' + str(directory) + '"', str(directory / 'quality.o'), str(directory / 'runtime.mm'), '-o', str(directory / 'quality')]])
    manifest = {'source': str(source), 'inputs': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()},
                'reduction_commit': REDUCTION_COMMIT, 'selection_commit': SELECTION_COMMIT,
                'attribution': 'Andrew Perminov, ternary_flip_graph, MIT; sign-aware helper adapted from the pinned reduction commit',
                'selection_rng': 'xorshift32 word modulo remaining suffix length; modulo bias retained, one word per candidate including final candidate',
                'metrics': {'successful_flips': 'tryFlip returned true', 'rank_decrease_events': 'iterations whose final rank is lower than initial rank',
                            'terms_removed': 'sum of positive iteration rank decreases'},
                'budget': 'all modes warm one dispatch then reset initial state; work requires zero budget; dispatch bounds accumulated host metalDispatch time; wall measures GPU-only search plus metric collection/logging, then replays CPU outside the budget; checked between rounds with reported overshoot; round_cap without budget_reached is not a completed budget comparison',
                'commands': commands}
    (output / 'algorithms.json').write_text(json.dumps(manifest, indent=2) + '\n')


def verify_exports(directory):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tests/metal'))
    from verify import verify
    ranks, unique = [], set()
    for path in sorted(directory.glob('*.json')):
        data = json.loads(path.read_text())
        if data.get('n') != [3, 3, 3] or data.get('z2') is not False:
            raise ValueError('expected signed 3x3 scheme: ' + str(path))
        verify(data)
        ranks.append(data['m'])
        terms = sorted(tuple(value for key in 'uvw' for value in data[key][r]) for r in range(data['m']))
        unique.add(hashlib.sha256(json.dumps(terms).encode()).hexdigest())
    if not ranks:
        raise ValueError('no exported schemes')
    return {'verified_exports': len(ranks), 'best_rank': min(ranks), 'ranks': ranks,
            'distinct_term_order_normalized_schemes': len(unique)}


def build(output):
    from application import build_identity, digest, source_identity, write_json
    output = output.resolve(strict=True)
    manifest_path = output / 'algorithms.json'
    manifest_hash = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    environment = {key: os.environ[key] for key in
                   ('PATH', 'HOME', 'TMPDIR', 'DEVELOPER_DIR', 'LANG', 'LC_ALL', 'USER', 'LOGNAME') if key in os.environ}
    for index, name in enumerate(('original', 'lazy', 'sign', 'both')):
        directory = output / name
        commands = manifest['commands'][2 * index:2 * index + 2]
        if len(commands) != 2:
            raise ValueError('missing algorithm build commands')
        identity = {'version': 1, 'source_root': str(directory), 'source_files': source_identity(directory),
                    'commands': commands, 'source_revision': 'algorithms.json sha256 ' + manifest_hash,
                    'algorithms_manifest_sha256': manifest_hash, 'metal_source_dir': str(directory),
                    'binary': str(directory / 'quality')}
        write_json(directory / 'build-attempt.json', identity)
        for step, command in enumerate(commands):
            with (directory / f'build-{step}.log').open('wb') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=45, env=environment)
            if digest(manifest_path) != manifest_hash or source_identity(directory) != identity['source_files']:
                raise ValueError('algorithm sources changed during compilation')
        identity['binary_sha256'] = digest(directory / 'quality')
        write_json(directory / 'build.json', identity)
        build_identity('candidate', directory / 'quality', directory, directory / 'build.json')


def main():
    parser = argparse.ArgumentParser(description='Generate controlled signed flip algorithm experiments')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--cpu-source', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-output', type=Path)
    parser.add_argument('--build', action='store_true', help='Compile generated variants and record verified build manifests')
    args = parser.parse_args()
    if args.verify_output:
        if args.source or args.cpu_source or args.output or args.build:
            parser.error('--verify-output cannot be combined with generation arguments')
        print(json.dumps(verify_exports(args.verify_output.resolve()), indent=2))
    else:
        if not args.source or not args.output or not args.cpu_source:
            parser.error('generation requires --source, --cpu-source and --output')
        generate(args.source.resolve(), args.output.resolve(), args.cpu_source.resolve())
        if args.build:
            build(args.output.resolve())


if __name__ == '__main__':
    main()
