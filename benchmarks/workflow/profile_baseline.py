"""Prepare a separate baseline diagnostic snapshot without building or running it.

Only a hashed source archive from pinned baseline freeze is accepted. Tracked
shared-buffer bytes describe application-owned MTLBuffer lengths, not driver
allocations or physical residency. Phase receipts require later behavioral
qualification before they may support performance conclusions.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

from baseline import BASELINE, archive_files, extract_source

RUNTIME = 'src/metal/runtime.mm'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('instrumentation anchor missing or ambiguous: '+repr(old[:80]))
    return source.replace(old, new, 1)


def instrument(source):
    source = replace_once(source, '#include <limits>\n', '#include <limits>\n#include <chrono>\n#include <iomanip>\n')
    source = replace_once(source, '    MetalRuntime() {\n', '''    // Logical shared-buffer lengths only; excludes driver and residency costs.
    size_t profileTransient = 0;
    size_t profilePeak = 0;
    size_t profileBytes() {
        size_t bytes = profileTransient;
        for (const auto &entry : buffers) bytes += entry.second.length;
        for (auto buffer : compactBuffers) if (buffer) bytes += buffer.length;
        return bytes;
    }
    void profileObserve(size_t pending = 0) {
        size_t bytes = profileBytes() + pending;
        if (bytes > profilePeak) profilePeak = bytes;
    }

    MetalRuntime() {
''')
    source = replace_once(source, '    r.buffers[buffer.contents] = buffer;\n', '    r.buffers[buffer.contents] = buffer;\n    r.profileObserve();\n')
    source = replace_once(source, '        auto &pipeline = r.pipelines[name];\n', '''        struct TransientReset {
            MetalRuntime &runtime;
            ~TransientReset() { runtime.profileTransient = 0; }
        } transientReset{r};
        auto &pipeline = r.pipelines[name];
''')
    source = replace_once(source, '    @autoreleasepool {\n', '''    const auto profileStart = std::chrono::steady_clock::now();
    @autoreleasepool {
''')
    source = replace_once(source, '        std::memset(errors.contents, 0, errors.length);\n', '''        r.profileTransient = errors.length;
        r.profileObserve();
        std::memset(errors.contents, 0, errors.length);
''')
    source = replace_once(source, '''                if (!r.compactBuffers[buffer] || r.compactBuffers[buffer].length < length)
                    r.compactBuffers[buffer] = [r.device newBufferWithLength:length options:MTLResourceStorageModeShared];
''', '''                if (!r.compactBuffers[buffer] || r.compactBuffers[buffer].length < length) {
                    id<MTLBuffer> replacement = [r.device newBufferWithLength:length options:MTLResourceStorageModeShared];
                    // Include the brief logical overlap with the previous buffer.
                    if (replacement) r.profileObserve(replacement.length);
                    r.compactBuffers[buffer] = replacement;
                }
''')
    source = replace_once(source, '''        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
''', '''        [encoder endEncoding];
        const auto profileEncoded = std::chrono::steady_clock::now();
        [command commit];
        [command waitUntilCompleted];
        const auto profileCompleted = std::chrono::steady_clock::now();
''')
    source = replace_once(source, '''                  << (command.GPUEndTime - command.GPUStartTime) * 1000 << " ms GPU" << std::endl;
''', '''                  << (command.GPUEndTime - command.GPUStartTime) * 1000 << " ms GPU" << std::endl;
        const auto profileReported = std::chrono::steady_clock::now();
        auto profileSeconds = [](auto a, auto b) {
            return std::chrono::duration<double>(b - a).count();
        };
        const auto oldPrecision = std::cout.precision();
        std::cout << std::setprecision(17)
                  << "FGM_PROFILE_V1 kernel=" << name
                  << " setup_seconds=" << profileSeconds(profileStart, profileEncoded)
                  << " commit_wait_seconds=" << profileSeconds(profileEncoded, profileCompleted)
                  << " validation_report_seconds=" << profileSeconds(profileCompleted, profileReported)
                  << " tracked_shared_live_bytes=" << r.profileBytes()
                  << " tracked_shared_peak_bytes=" << r.profilePeak
                  << std::endl;
        std::cout.precision(oldPrecision);
''')
    return source


def prepare(baseline, output):
    baseline, output = Path(baseline), Path(output)
    if output.resolve().is_relative_to(baseline.resolve()):
        raise ValueError('diagnostic output must be outside baseline preservation directory')
    if output.exists() or output.is_symlink():
        raise ValueError('output must be new')
    freeze_path, archive_path = baseline/'freeze.json', baseline/'source.tar'
    if freeze_path.is_symlink() or archive_path.is_symlink():
        raise ValueError('baseline inputs must not be symlinks')
    freeze_bytes = freeze_path.read_bytes()
    freeze = json.loads(freeze_bytes)
    data = archive_path.read_bytes()
    if freeze.get('commit') != BASELINE or freeze.get('archive_sha256') != sha(data):
        raise ValueError('baseline pin or archive hash mismatch')
    files = archive_files(data)
    original = files[RUNTIME][0]
    changed = instrument(original.decode('utf-8')).encode('utf-8')
    patch = ''.join(difflib.unified_diff(original.decode().splitlines(True), changed.decode().splitlines(True),
                                       fromfile='baseline/'+RUNTIME, tofile='diagnostic/'+RUNTIME)).encode()
    original_inventory = {name: sha(value[0]) for name,value in sorted(files.items())}
    files[RUNTIME] = (changed, files[RUNTIME][1])
    # Check all prerequisites before creating the new directory. Never copy builds.
    output.mkdir(parents=True, exist_ok=False)
    source = output/'source'
    extract_source(files, source)
    (output/'runtime.patch').write_bytes(patch)
    receipt = dict(schema='fgm-baseline-profile-v1', baseline_commit=BASELINE,
                   baseline_archive_sha256=sha(data), freeze_sha256=sha(freeze_bytes),
                   generator_sha256=sha(Path(__file__).read_bytes()),
                   archive_helper_sha256=sha(Path(__file__).with_name('baseline.py').read_bytes()),
                   patch_sha256=sha(patch),
                   original_files=original_inventory,
                   diagnostic_files={name:sha(value[0]) for name,value in sorted(files.items())},
                   build_cwd=str(source.resolve()),
                   build_commands=[['make','METAL_LIBRARY_MODE=metallib','build/metal/'+program]
                                   for program in ('flip_graph','flip_graph_f2','complexity_minimizer',
                                                   'complexity_minimizer_f2','additions_reducer')],
                   build_status='NOT RUN', behavior_status='NOT VERIFIED',
                   phase_scope={'setup':'runtime acquisition, pipeline setup, buffer setup and encoding',
                                'commit_wait':'command submission and blocking completion wait; overlaps GPU duration',
                                'validation_report':'status validation and existing dispatch report',
                                'excluded':'application host phases outside metalLaunch and profile-line output'},
                   memory_scope='Logical live and process peak tracked shared MTLBuffer lengths, including compact and error buffers; excludes driver allocations, autorelease retention and physical residency',
                   qualification='Compare exports and required complete trajectories against uninstrumented baseline before interpreting diagnostic timings; never substitute for production timing')
    payload = json.dumps(receipt, sort_keys=True, indent=2).encode()+b'\n'
    (output/'profile.json').write_bytes(payload)
    (output/'profile.sha256').write_text(sha(payload)+'\n')
    # Detect changes during preparation. Retain output as incomplete evidence on error.
    if archive_path.read_bytes() != data or freeze_path.read_bytes() != freeze_bytes:
        raise ValueError('baseline inputs changed during preparation')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True, help='directory containing freeze.json and source.tar')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.baseline, args.output)
    print('Prepared diagnostic source:', result['build_cwd'])
    print('Build and behavioral qualification: NOT RUN')


if __name__ == '__main__':
    main()
