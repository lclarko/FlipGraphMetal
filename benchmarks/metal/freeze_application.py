"""Freeze and build the current Metal application for an immutable comparison."""

import argparse
from pathlib import Path
import shutil
import subprocess

from application import build_identity, digest, source_identity, write_json
from guard import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Metal source directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(strict=True), args.output.resolve()
    if output == source or output.is_relative_to(source):
        parser.error('output must be outside the source directory')
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'source'
    shader = snapshot / 'src/metal'
    shutil.copytree(source, shader)
    root = Path(__file__).resolve().parents[2]
    # The Metal entry point shares the original host argument parser.
    shared = snapshot / 'src/entities'
    shared.mkdir()
    for name in ('arg_parser.cu', 'arg_parser.cuh'):
        shutil.copy2(root / 'src/entities' / name, shared / name)
    binary = output / 'flip_graph'
    command = ['xcrun', 'clang++', '-mmacosx-version-min=15.0', '-std=c++17', '-O2',
               '-fobjc-arc', '-ffp-contract=off', '-framework', 'Foundation', '-framework', 'Metal',
               '-DMETAL_SOURCE_DIR="' + str(shader) + '"', '-DMETAL_PROGRAM=1',
               str(shader / 'main.cpp'), str(shader / 'runtime.mm'), '-o', str(binary)]
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    manifest = dict(version=1, source_revision=revision + '+source-snapshot',
                    source_root=str(snapshot), source_files=source_identity(snapshot),
                    metal_source_dir=str(shader), binary=str(binary), commands=[command],
                    input_source=str(source), generator_sha256=digest(Path(__file__)), complete=False)
    path = output / 'build.json'
    write_json(path, manifest)
    record = run(command, output / 'compile')
    if not record['complete']:
        raise SystemExit('application build incomplete; see ' + str(output / 'compile'))
    manifest.update(binary_sha256=digest(binary), complete=True)
    write_json(path, manifest)
    build_identity('candidate', binary, snapshot, path)
    print('Frozen application:', binary)


if __name__ == '__main__':
    main()
