"""Freeze one project's Metal application and its matching host parser."""

import argparse
from pathlib import Path
import shutil
import subprocess

from application import build_identity, digest, source_identity, write_json
from guard import run


def parser_files(project):
    layouts = [('src/common', 'arg_parser.cpp', 'arg_parser.h'),
               ('src/entities', 'arg_parser.cu', 'arg_parser.cuh')]
    found = [layout for layout in layouts if any((project / layout[0] / name).exists() for name in layout[1:])]
    if len(found) != 1:
        raise ValueError('selected project must contain exactly one parser layout')
    folder, implementation, header = found[0]
    files = [project / folder / name for name in (implementation, header)]
    if any(not p.is_file() or p.is_symlink() for p in files):
        raise ValueError('selected project parser is missing or is a symlink')
    entry = project / 'src/metal/main.cpp'
    expected = '../' + Path(folder).name + '/' + implementation
    if ('"' + expected + '"') not in entry.read_text():
        raise ValueError('Metal entry point does not include the selected project parser')
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    project, output = args.project_root.resolve(strict=True), args.output.resolve()
    source = project / 'src/metal'
    try:
        shared = parser_files(project)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if not source.is_dir() or source.is_symlink() or any(p.is_symlink() for p in source.rglob('*')):
        parser.error('selected Metal source must be a directory without symlinks')
    if output == source or output.is_relative_to(source) or any(output.is_relative_to(p.parent) for p in shared):
        parser.error('output must be outside the source directories')
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'source'
    shader = snapshot / 'src/metal'
    shutil.copytree(source, shader)
    for path in shared:
        target = snapshot / path.relative_to(project)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    binary = output / 'flip_graph'
    command = ['xcrun', 'clang++', '-mmacosx-version-min=15.0', '-std=c++17', '-O2',
               '-fobjc-arc', '-ffp-contract=off', '-framework', 'Foundation', '-framework', 'Metal',
               '-DMETAL_SOURCE_DIR="' + str(shader) + '"', '-DMETAL_PROGRAM=1',
               str(shader / 'main.cpp'), str(shader / 'runtime.mm'), '-o', str(binary)]
    # A source-only snapshot need not have Git metadata. File hashes are authoritative.
    revision = 'source-snapshot'
    if (project / '.git').exists():
        revision = subprocess.check_output(['git', '-C', str(project), 'rev-parse', 'HEAD'], text=True).strip() + '+source-snapshot'
    manifest = dict(version=1, source_revision=revision,
                    source_root=str(snapshot), source_files=source_identity(snapshot),
                    metal_source_dir=str(shader), binary=str(binary), commands=[command],
                    input_project=str(project), input_source=str(source),
                    input_parser={str(p.relative_to(project)): digest(p) for p in shared},
                    generator_sha256=digest(Path(__file__)), complete=False)
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
