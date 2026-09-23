#!/usr/bin/env python3
"""Construct signed 4x4 fixtures from scalar multiplication and Strassen's formulas.

No downloaded implementation or fixture is used. W indexes C transposed:
U[i*4+j], V[j*4+k], W[k*4+i]. Raw files contain one scheme, without a count.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/metal"))
from verify import verify


def naive():
    data = {"n": [4, 4, 4], "m": 64, "z2": False, "u": [], "v": [], "w": []}
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for key, index in zip("uvw", (4*i+j, 4*j+k, 4*k+i)):
                    data[key].append([int(x == index) for x in range(16)])
    return data


def strassen():
    # A=(a,b;c,d), B=(e,f;g,h). Seven products in their canonical order.
    u = [[1,0,0,1], [0,0,1,1], [1,0,0,0], [0,0,0,1],
         [1,1,0,0], [-1,0,1,0], [0,1,0,-1]]
    v = [[1,0,0,1], [1,0,0,0], [0,1,0,-1], [-1,0,1,0],
         [0,0,0,1], [1,1,0,0], [0,0,1,1]]
    # Contributions to C11,C12,C21,C22, before converting to W order.
    output = [[1,0,0,1], [0,0,1,-1], [0,1,0,1], [1,0,1,0],
              [-1,1,0,0], [0,0,0,1], [1,0,0,0]]
    w = [[row[0], row[2], row[1], row[3]] for row in output]
    base = {"n": [2,2,2], "m": 7, "z2": False, "u": u, "v": v, "w": w}
    verify(base)
    data = {"n": [4,4,4], "m": 49, "z2": False}
    for key in "uvw":
        rows = data[key] = []
        for outer in base[key]:
            for inner in base[key]:
                row = [0]*16
                for a in range(2):
                    for b in range(2):
                        for c in range(2):
                            for d in range(2):
                                row[(2*a+c)*4 + 2*b+d] = outer[a*2+b]*inner[c*2+d]
                rows.append(row)
    return data


def encode(data):
    verify(data)
    lines = [" ".join(map(str, data["n"] + [data["m"]]))]
    for key in "uvw":
        lines.extend(" ".join(map(str, row)) for row in data[key])
    return ("\n".join(lines) + "\n").encode("ascii")


def artifacts():
    result = {}
    for name, construct in (("naive_4x4", naive), ("strassen_4x4", strassen)):
        data = construct()
        checked = verify(data)
        raw = encode(data)
        result[name + ".txt"] = raw
        provenance = {
            "version": 1, "fixture": name + ".txt", "sha256": hashlib.sha256(raw).hexdigest(),
            "dimensions": data["n"], "rank": data["m"], "coefficient_domain": "signed integers {-1,0,1}",
            "construction": "scalar schoolbook multiplication" if name.startswith("naive") else "tensor square of the canonical seven-product 2x2 Strassen formulas",
            "source": "generated from mathematical definitions; no external code or fixture copied",
            "mathematical_reference": None if name.startswith("naive") else "Volker Strassen, Gaussian elimination is not optimal, Numerische Mathematik 13 (1969), 354-356, doi:10.1007/BF02165411",
            "generator": "benchmarks/metal/make_4x4_fixtures.py",
            "format": "one raw scheme without a scheme-count prefix; U[i*4+j], V[j*4+k], W[k*4+i]",
            "verification": {"method": "independent exact-integer tensor reconstruction using tests/metal/verify.py", "equations": checked["equations"]},
        }
        result[name + ".provenance.json"] = (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode()
    return result


def write_fixtures(output):
    content = artifacts()
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in content.items():
        with (output / name).open("xb") as stream:
            stream.write(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory; existing directories are refused")
    args = parser.parse_args()
    write_fixtures(args.output)


if __name__ == "__main__":
    main()
