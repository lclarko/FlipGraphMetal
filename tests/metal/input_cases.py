"""Exercise file-loaded dimensions with a generated Strassen tensor cube."""
import json
from pathlib import Path
from verify import verify
from smoke import dispatch_evidence


def tensor_cube():
    u = [[1,0,0,1], [0,0,1,1], [1,0,0,0], [0,0,0,1],
         [1,1,0,0], [-1,0,1,0], [0,1,0,-1]]
    v = [[1,0,0,1], [1,0,0,0], [0,1,0,-1], [-1,0,1,0],
         [0,0,0,1], [1,1,0,0], [0,0,1,1]]
    w = [[1,0,0,1], [0,1,0,-1], [0,0,1,1], [1,1,0,0],
         [-1,0,1,0], [0,0,0,1], [1,0,0,0]]
    data = dict(n=[2,2,2], m=7, z2=False, u=u, v=v, w=w)
    base = data
    for width in (2, 4):
        result = dict(n=[width*2]*3, m=data['m']*7, z2=False)
        for key in 'uvw':
            result[key] = []
            for outer in data[key]:
                for inner in base[key]:
                    row = [0]*(width*2)**2
                    for i in range(width):
                        for j in range(width):
                            for k in range(2):
                                for l in range(2):
                                    row[(2*i+k)*(2*width)+2*j+l] = outer[i*width+j]*inner[2*k+l]
                    result[key].append(row)
        data = result
    return data


def run_input_cases(binary, attempt, guarded):
    data = tensor_cube()
    output = attempt / 'file-inputs'
    output.mkdir()
    checks = []
    for z2, program in ((False, 'flip_graph'), (True, 'flip_graph_f2')):
        fixture = {**data, 'z2': z2}
        if z2:
            for key in 'uvw':
                fixture[key] = [[v % 2 for v in row] for row in data[key]]
        # Exact integer reconstruction is independent of shared C++/Metal code.
        checked = verify(fixture)
        path = output / (program + '.txt')
        lines = ['1', '8 8 8 343']
        lines += [' '.join(map(str, row)) for key in 'uvw' for row in fixture[key]]
        path.write_text('\n'.join(lines) + '\n')
        command = [str(binary / program), '-n1', '8', '-n2', '8', '-n3', '8',
                   '--input-path', str(path), '--schemes', '1', '--block-size', '1',
                   '--max-iterations', '1', '--resize-probability', '0',
                   '--expand-probability', '0', '--rounds', '1', '--seed', '7',
                   '--path', str(output / (program + '-exports'))]
        log = guarded('input-' + program, command)
        exports = {file.name: verify(json.loads(file.read_text()))
                   for file in sorted((output / (program + '-exports')).glob('*.json'))}
        checks.append(dict(program=program, verification=checked, exports=exports,
                           **dispatch_evidence(log)))
    (output / 'verification.json').write_text(json.dumps(checks, indent=2) + '\n')
