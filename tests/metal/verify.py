#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    require(type(value) is int, "expected an integer")
    return value


def reconstruct(expressions, fresh, variables, modulo):
    forms = [[int(i == j) for i in range(variables)] for j in range(variables)]

    def linear_form(expression):
        require(isinstance(expression, list), "expression must be a list")
        result = [0] * variables
        for term in expression:
            require(isinstance(term, dict) and set(term) == {"index", "value"}, "invalid circuit term")
            index, value = integer(term["index"]), integer(term["value"])
            require(0 <= index < len(forms), "circuit has a forward or out-of-range reference")
            require(value in (-1, 1), "invalid circuit sign")
            for i, coefficient in enumerate(forms[index]):
                result[i] += value * coefficient
        return [v % 2 for v in result] if modulo else result

    for expression in fresh:
        require(len(expression) == 2, "fresh variable must contain exactly one binary operation")
        forms.append(linear_form(expression))
    outputs = [linear_form(expression) for expression in expressions]
    count = len(fresh) + sum(max(0, len(expression) - 1) for expression in expressions)
    return outputs, count


def verify(data, reference=None):
    require(isinstance(data, dict), "scheme must be an object")
    n = data["n"]
    require(isinstance(n, list) and len(n) == 3, "expected three dimensions")
    a, b, c = [integer(v) for v in n]
    m = integer(data["m"])
    require(min(a, b, c, m) > 0, "dimensions and rank must be positive")
    require(type(data["z2"]) is bool, "z2 must declare the coefficient domain")
    modulo = data["z2"]
    lengths = [a * b, b * c, c * a]
    circuit = "u_fresh" in data
    matrices = []
    operations = 0
    for p, key in enumerate("uvw"):
        if circuit:
            forms, count = reconstruct(data[key], data[key + "_fresh"], lengths[p] if p < 2 else m, modulo)
            operations += count
            require(len(forms) == (m if p < 2 else lengths[p]), "incorrect circuit output count")
            matrix = forms if p < 2 else [list(row) for row in zip(*forms)]
        else:
            matrix = data[key]
        require(isinstance(matrix, list) and len(matrix) == m, "incorrect factor rank")
        for row in matrix:
            require(isinstance(row, list) and len(row) == lengths[p], "incorrect factor width")
            for value in row:
                integer(value)
                require(value in ((0, 1) if modulo else (-1, 0, 1)), "coefficient outside declared domain")
        matrices.append(matrix)
    if reference is not None:
        verify(reference)
        require(reference["n"] == n and reference["m"] == m and reference["z2"] == modulo, "reference scheme metadata mismatch")
        require(matrices == [reference[key] for key in "uvw"], "circuit forms differ from reference factors")
    u, v, w = matrices
    for i in range(a * b):
        for j in range(b * c):
            for k in range(c * a):
                expected = int(i % b == j // c and i // b == k % a and j % c == k // a)
                actual = sum(u[r][i] * v[r][j] * w[r][k] for r in range(m))
                if modulo:
                    actual %= 2
                require(actual == expected, f"tensor equation ({i},{j},{k}): got {actual}, expected {expected}")
    naive = sum(max(0, sum(value != 0 for value in row) - 1) for row in u + v)
    naive += sum(max(0, sum(row[k] != 0 for row in w) - 1) for k in range(c * a))
    if circuit:
        require(integer(data["complexity"]["reduced"]) == operations, "incorrect circuit operation count")
        require(integer(data["complexity"]["naive"]) == naive, "incorrect naive operation count")
    elif "complexity" in data:
        require(integer(data["complexity"]) == naive, "incorrect complexity")
    return {"domain": "F2" if modulo else "ZT", "rank": m, "equations": a*b*b*c*c*a, "additions": operations if circuit else naive}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text()) if args.reference else None
    for path in args.files:
        print(path, "PASS", verify(json.loads(path.read_text()), reference))


if __name__ == "__main__":
    main()
