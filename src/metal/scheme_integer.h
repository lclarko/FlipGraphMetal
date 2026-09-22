#pragma once

struct SchemeInteger {
    int n[3];
    int nn[3];
    int m;
    Addition uvw[3][MAX_RANK];
    FlipSet flips[3];

    bool validate() const LOCAL_METHOD;
    void initializeNaive(int n1, int n2, int n3) LOCAL_METHOD;
    void copyTo(LOCAL SchemeInteger &target, bool withFlips = true) const LOCAL_METHOD;
#ifndef __METAL_VERSION__
    bool read(std::istream &is, bool checkValidity = true);
#endif
#ifndef __METAL_VERSION__
    bool read(std::istream &is, int n1, int n2, int n3, int m, bool checkValidity = true);
#endif

    int getComplexity() const LOCAL_METHOD;
    int getMaxRealVariables(int position) const LOCAL_METHOD;
    int getMaxSubexpressions(int position) const LOCAL_METHOD;

    bool isValidProject(int i, int minN = MIN_PROJECT_N) const LOCAL_METHOD;
    bool isValidExtension(int i, int maxN = MAX_EXTENSION_N) const LOCAL_METHOD;
    bool isValidProduct(int i, int maxN = MAX_EXTENSION_N) const LOCAL_METHOD;
    bool isValidMerge(int i, LOCAL const SchemeInteger &scheme) const LOCAL_METHOD;

    bool tryFlip(LOCAL RandomState &state, bool checkReduce = true) LOCAL_METHOD;
    bool tryPlus(LOCAL RandomState &state) LOCAL_METHOD;
    bool trySplit(LOCAL RandomState &state) LOCAL_METHOD;
    bool trySplitExisted(LOCAL RandomState &state) LOCAL_METHOD;
    bool tryExpand(int count, LOCAL RandomState &state) LOCAL_METHOD;
    bool tryReduce() LOCAL_METHOD;
    bool tryProject(LOCAL RandomState &state, int minN = MIN_PROJECT_N) LOCAL_METHOD;
    bool tryExtend(LOCAL RandomState &state, int maxN = MAX_EXTENSION_N) LOCAL_METHOD;
    bool tryProduct(LOCAL RandomState &state, int maxN = MAX_EXTENSION_N) LOCAL_METHOD;
    bool tryMerge(GLOBAL const SchemeInteger &scheme, LOCAL RandomState &state) LOCAL_METHOD;
    bool tryProduct(GLOBAL const SchemeInteger &scheme) LOCAL_METHOD;
    void swapBasis(LOCAL RandomState &state) LOCAL_METHOD;
    void swapSize(LOCAL RandomState &state) LOCAL_METHOD;

    void merge(GLOBAL const SchemeInteger &scheme, int p) LOCAL_METHOD;
    void project(int p, int q) LOCAL_METHOD;
    void extend(int p) LOCAL_METHOD;
    void product(int p) LOCAL_METHOD;

#ifndef __METAL_VERSION__
    void save(const std::string &path);
#endif
#ifndef __METAL_VERSION__
    void show() const;
#endif
private:
    bool validateEquation(int i, int j, int k) const LOCAL_METHOD;
    void checkSubexpression(int v1, int v2, LOCAL bool &pos, LOCAL bool &neg) const LOCAL_METHOD;

    void initFlips() LOCAL_METHOD;
    void removeZeroes() LOCAL_METHOD;
    void removeAt(int startIndex) LOCAL_METHOD;
    void addTriplet(int i, int j, int k, LOCAL const Addition &u, LOCAL const Addition &v, LOCAL const Addition &w) LOCAL_METHOD;
    void excludeColumn(int matrix, int column) LOCAL_METHOD;
    void excludeRow(int matrix, int row) LOCAL_METHOD;
    void addColumn(int matrix) LOCAL_METHOD;
    void addRow(int matrix) LOCAL_METHOD;
    bool fixSigns() LOCAL_METHOD;

    void flip(int i, int j, int k, int index1, int index2, bool checkReduce) LOCAL_METHOD;
    void plus(int i, int j, int k, int index1, int index2, int variant) LOCAL_METHOD;
    void split(int i, int j, int k, int index, LOCAL const Addition & addition) LOCAL_METHOD;
    void reduceAdd(int i, int index1, int index2) LOCAL_METHOD;
    void reduceSub(int i, int index1, int index2) LOCAL_METHOD;
    void product(GLOBAL const SchemeInteger &scheme) LOCAL_METHOD;
    void swapBasisRows(int i1, int i2) LOCAL_METHOD;
    void swapBasisColumns(int j1, int j2) LOCAL_METHOD;
    void swapSize(int p1, int p2) LOCAL_METHOD;

#ifndef __METAL_VERSION__
    void saveMatrix(std::ofstream &f, std::string name, int m, LOCAL const Addition *additions) const;
#endif
#ifndef __METAL_VERSION__
    void showTensor(LOCAL const Addition &addition, int n1, int n2, std::string name, bool transpose) const;
#endif
};

bool SchemeInteger::validateEquation(int i, int j, int k) const LOCAL_METHOD {
    int i1 = i / n[1];
    int i2 = i % n[1];
    int j1 = j / n[2];
    int j2 = j % n[2];
    int k1 = k / n[0];
    int k2 = k % n[0];

    int target = (i2 == j1) && (i1 == k2) && (j2 == k1);
    int equation = 0;

    for (int index = 0; index < m; index++)
        equation += uvw[0][index][i] * uvw[1][index][j] * uvw[2][index][k];

    return equation == target;
}

bool SchemeInteger::validate() const LOCAL_METHOD {
    if (m < 1 || m > MAX_RANK) return false;
    for (int i = 0; i < 3; i++)
        if (n[i] < 1 || n[i] > 16 || nn[i] != n[i] * n[(i + 1) % 3] || nn[i] > 64) return false;
    bool valid = true;

    for (int i = 0; i < nn[0] && valid; i++)
        for (int j = 0; j < nn[1] && valid; j++)
            for (int k = 0; k < nn[2] && valid; k++)
                valid &= validateEquation(i, j, k);

    for (int index = 0;index < m; index++) {
        for (int i = 0; i < 3; i++) {
            if (!uvw[i][index].valid || (nn[i] < 64 && (uvw[i][index].values >> nn[i]) != 0)) {

                return false;
            }
        }
    }

    return valid;
}

void SchemeInteger::initializeNaive(int n1, int n2, int n3) LOCAL_METHOD {
    n[0] = n1;
    n[1] = n2;
    n[2] = n3;

    nn[0] = n1 * n2;
    nn[1] = n2 * n3;
    nn[2] = n3 * n1;

    m = n1 * n2 * n3;

    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n3; j++) {
            for (int k = 0; k < n2; k++) {
                int index = (i * n3 + j) * n2 + k;
                uvw[0][index] = Addition(n1 * n2, i * n2 + k);
                uvw[1][index] = Addition(n2 * n3, k * n3 + j);
                uvw[2][index] = Addition(n3 * n1, j * n1 + i);
            }
        }
    }

    initFlips();
}

void SchemeInteger::copyTo(LOCAL SchemeInteger &target, bool withFlips) const LOCAL_METHOD {
    target.m = m;

    for (int i = 0; i < 3; i++) {
        target.n[i] = n[i];
        target.nn[i] = nn[i];

        for (int index = 0; index < m; index++)
            target.uvw[i][index] = uvw[i][index];
    }

    if (withFlips)
        target.initFlips();
}

#ifndef __METAL_VERSION__
bool SchemeInteger::read(std::istream &is, bool checkValidity) {
    if (!(is >> n[0] >> n[1] >> n[2] >> m)) return false;

    if (n[0] < 1 || n[1] < 1 || n[2] < 1 || n[0] > 16 || n[1] > 16 || n[2] > 16 || m < 1 || n[0] * n[1] > MAX_MATRIX_ELEMENTS || n[1] * n[2] > MAX_MATRIX_ELEMENTS || n[2] * n[0] > MAX_MATRIX_ELEMENTS || m > MAX_RANK) {

        return false;
    }

    int value;

    for (int i = 0; i < 3; i++) {
        nn[i] = n[i] * n[(i + 1) % 3];

        for (int index = 0; index < m; index++) {
            uvw[i][index] = Addition(nn[i]);

            for (int j = 0; j < nn[i]; j++) {
                if (!(is >> value) || value < -1 || value > 1) return false;
                uvw[i][index].set(j, value);
            }
        }
    }

    fixSigns();
    initFlips();
    return !checkValidity || validate();
}
#endif

#ifndef __METAL_VERSION__
bool SchemeInteger::read(std::istream &is, int n1, int n2, int n3, int m, bool checkValidity) {
    if (n1 < 1 || n2 < 1 || n3 < 1 || n1 > 16 || n2 > 16 || n3 > 16 || m < 1 || n1 * n2 > MAX_MATRIX_ELEMENTS || n2 * n3 > MAX_MATRIX_ELEMENTS || n3 * n1 > MAX_MATRIX_ELEMENTS || m > MAX_RANK) {

        return false;
    }

    this->n[0] = n1;
    this->n[1] = n2;
    this->n[2] = n3;
    this->m = m;
    int value;

    for (int i = 0; i < 3; i++) {
        nn[i] = n[i] * n[(i + 1) % 3];

        for (int index = 0; index < m; index++) {
            uvw[i][index] = Addition(nn[i]);

            for (int j = 0; j < nn[i]; j++) {
                if (!(is >> value) || value < -1 || value > 1) return false;
                uvw[i][index].set(j, value);
            }
        }
    }

    fixSigns();
    initFlips();
    return !checkValidity || validate();
}
#endif

int SchemeInteger::getComplexity() const LOCAL_METHOD {
    int count = 0;
    for (int index = 0; index < m; index++) {
        for (int i = 0; i < 3; i++) {
            count += uvw[i][index].nonZeroCount();
            if (i < 2 && uvw[i][index]) count--;
        }
    }
    return count - nn[2];
}

int SchemeInteger::getMaxRealVariables(int position) const LOCAL_METHOD {
    int maxVariables = 0;

    if (position == 2) {
        for (int i = 0; i < nn[position]; i++) {
            int variables = 0;

            for (int index = 0; index < m; index++)
                if (uvw[position][index][i])
                    variables++;

            if (variables > maxVariables)
                maxVariables = variables;
        }
    }
    else {
        for (int index = 0; index < m; index++) {
            int variables = 0;

            for (int i = 0; i < nn[position]; i++)
                if (uvw[position][index][i])
                    variables++;

            if (variables > maxVariables)
                maxVariables = variables;
        }
    }

    return maxVariables;
}

int SchemeInteger::getMaxSubexpressions(int position) const LOCAL_METHOD {
    int maxSubexpressions = 0;

    if (position == 2) {
        for (int index1 = 0; index1 < m; index1++) {
            for (int index2 = index1 + 1; index2 < m; index2++) {
                bool pos = false;
                bool neg = false;

                for (int i = 0; i < nn[position] && (!pos || !neg); i++)
                    checkSubexpression(uvw[position][index1][i], uvw[position][index2][i], pos, neg);

                maxSubexpressions += int(pos) + int(neg);
            }
        }
    }
    else {
        for (int i1 = 0; i1 < nn[position]; i1++) {
            for (int i2 = i1 + 1; i2 < nn[position]; i2++) {
                bool pos = false;
                bool neg = false;

                for (int index = 0; index < m && (!pos || !neg); index++)
                    checkSubexpression(uvw[position][index][i1], uvw[position][index][i2], pos, neg);

                maxSubexpressions += int(pos) + int(neg);
            }
        }
    }

    return maxSubexpressions;
}

void SchemeInteger::checkSubexpression(int v1, int v2, LOCAL bool &pos, LOCAL bool &neg) const LOCAL_METHOD {
    if (!v1 || !v2)
        return;

    if (v1 < 0) {
        v1 = -v1;
        v2 = -v2;
    }

    if (v2 == 1)
        pos = true;

    if (v2 == -1)
        neg = true;
}

void SchemeInteger::initFlips() LOCAL_METHOD {
    for (int i = 0; i < 3; i++) {
        flips[i].clear();

        for (int index1 = 0; index1 < m; index1++)
            for (int index2 = index1 + 1; index2 < m; index2++)
                if (uvw[i][index1] == uvw[i][index2])
                    flips[i].add(index1, index2);
    }
}

void SchemeInteger::removeZeroes() LOCAL_METHOD {
    for (int index = 0; index < m; index++) {
        if (uvw[0][index] && uvw[1][index] && uvw[2][index])
            continue;

        m--;
        uvw[0][index] = uvw[0][m];
        uvw[1][index] = uvw[1][m];
        uvw[2][index] = uvw[2][m];
        index--;
    }
}

void SchemeInteger::removeAt(int index) LOCAL_METHOD {
    m--;

    if (index == m)
        return;

    uvw[0][index] = uvw[0][m];
    uvw[1][index] = uvw[1][m];
    uvw[2][index] = uvw[2][m];
}

void SchemeInteger::addTriplet(int i, int j, int k, LOCAL const Addition &u, LOCAL const Addition &v, LOCAL const Addition &w) LOCAL_METHOD {
    u.copyTo(uvw[i][m]);
    v.copyTo(uvw[j][m]);
    w.copyTo(uvw[k][m]);
    m++;
}

void SchemeInteger::excludeColumn(int matrix, int column) LOCAL_METHOD {
    int n1 = n[matrix];
    int n2 = n[(matrix + 1) % 3];
    int oldColumns[MAX_MATRIX_ELEMENTS];
    int size = 0;

    for (int j = 0; j < n2; j++)
        if (j != column)
            oldColumns[size++] = j;

    for (int index = 0; index < m; index++) {
        Addition value(n1 * (n2 - 1));

        for (int i = 0; i < n1; i++)
            for (int j = 0; j < n2 - 1; j++)
                value.set(i * (n2 - 1) + j, uvw[matrix][index][i * n2 + oldColumns[j]]);

        uvw[matrix][index] = value;
    }
}

void SchemeInteger::excludeRow(int matrix, int row) LOCAL_METHOD {
    int n1 = n[matrix];
    int n2 = n[(matrix + 1) % 3];
    int oldRows[MAX_MATRIX_ELEMENTS];
    int size = 0;

    for (int i = 0; i < n1; i++)
        if (i != row)
            oldRows[size++] = i;

    for (int index = 0; index < m; index++) {
        Addition value((n1 - 1) * n2);

        for (int i = 0; i < n1 - 1; i++)
            for (int j = 0; j < n2; j++)
                value.set(i * n2 + j, uvw[matrix][index][oldRows[i] * n2 + j]);

        uvw[matrix][index] = value;
    }
}

void SchemeInteger::addColumn(int matrix) LOCAL_METHOD {
    int n1 = n[matrix];
    int n2 = n[(matrix + 1) % 3];

    for (int index = 0; index < m; index++) {
        Addition value(n1 * (n2 + 1));

        for (int i = 0; i < n1; i++)
            for (int j = 0; j < n2; j++)
                value.set(i * (n2 + 1) + j, uvw[matrix][index][i * n2 + j]);

        uvw[matrix][index] = value;
    }
}

void SchemeInteger::addRow(int matrix) LOCAL_METHOD {
    int n1 = n[matrix];
    int n2 = n[(matrix + 1) % 3];

    for (int index = 0; index < m; index++) {
        Addition value((n1 + 1) * n2);

        for (int i = 0; i < n1; i++)
            for (int j = 0; j < n2; j++)
                value.set(i * n2 + j, uvw[matrix][index][i * n2 + j]);

        uvw[matrix][index] = value;
    }
}

bool SchemeInteger::isValidProject(int i, int minN) const LOCAL_METHOD {
    return n[i] - 1 >= minN && n[(i + 1) % 3] >= minN && n[(i + 2) % 3] >= minN;
}

bool SchemeInteger::isValidExtension(int i, int maxN) const LOCAL_METHOD {
    int newN[3] = {n[0], n[1], n[2]};
    newN[i] += 1;

    if (m + newN[(i + 1) % 3] * newN[(i + 2) % 3] > MAX_RANK)
        return false;

    for (int p = 0; p < 3; p++) {
        if (newN[p] * newN[(p + 1) % 3] > MAX_MATRIX_ELEMENTS)
            return false;

        if (newN[p] > maxN)
            return false;
    }

    return true;
}

bool SchemeInteger::isValidProduct(int i, int maxN) const LOCAL_METHOD {
    int newN[3] = {n[0], n[1], n[2]};
    newN[i] *= 2;

    if (m * 2 > MAX_RANK)
        return false;

    for (int p = 0; p < 3; p++) {
        if (newN[p] * newN[(p + 1) % 3] > MAX_MATRIX_ELEMENTS)
            return false;

        if (newN[p] > maxN)
            return false;
    }

    return true;
}

bool SchemeInteger::isValidMerge(int i, LOCAL const SchemeInteger &scheme) const LOCAL_METHOD {
    if (m + scheme.m > MAX_RANK)
        return false;

    int j = (i + 1) % 3;
    int k = (i + 2) % 3;

    int eq2 = n[j] == scheme.n[j];
    int eq3 = n[k] == scheme.n[k];

    int n1 = n[i] + scheme.n[i];

    return n1 <= MAX_EXTENSION_N && n1 * n[j] <= MAX_MATRIX_ELEMENTS && n1 * n[k] <= MAX_MATRIX_ELEMENTS && eq2 && eq3;
}

bool SchemeInteger::fixSigns() LOCAL_METHOD {
    bool changed = false;

    for (int index = 0; index < m; index++) {
        bool i = uvw[0][index].positiveFirstNonZero();
        bool j = uvw[1][index].positiveFirstNonZero();

        if (i && j)
            continue;

        if (!i && !j) {
            uvw[0][index].inverse();
            uvw[1][index].inverse();
        }
        else if (!i) {
            uvw[0][index].inverse();
            uvw[2][index].inverse();
        }
        else {
            uvw[1][index].inverse();
            uvw[2][index].inverse();
        }

        changed = true;
    }

    return changed;
}

void SchemeInteger::flip(int i, int j, int k, int index1, int index2, bool checkReduce) LOCAL_METHOD {
    uvw[j][index1] += uvw[j][index2];
    uvw[k][index2] -= uvw[k][index1];

    flips[j].remove(index1);
    flips[k].remove(index2);

    if (!uvw[j][index1] || !uvw[k][index2]) {
        removeZeroes();
        initFlips();

        // Successful reductions lower rank; the bound avoids a Metal loop stall.
        for (int remaining = m; remaining > 0; remaining--)
            if (!tryReduce()) break;
        return;
    }

    bool useMasks = m <= 32;
    uint32_t matchesJ = 0, matchesK = 0;
    if (useMasks) {
        for (int index = 0; index < m; index++) {
            matchesJ |= uint32_t(uvw[j][index] == uvw[j][index1]) << index;
            matchesK |= uint32_t(uvw[k][index] == uvw[k][index2]) << index;
        }
        matchesJ &= ~(1u << index1);
        matchesK &= ~(1u << index2);
    }
    uint32_t matches = matchesJ | matchesK;
    for (int index = 0; index < m; index++) {
        if (useMasks) {
            if (!matches) break;
#ifdef __METAL_VERSION__
            index = ctz(matches);
#else
            index = __builtin_ctz(matches);
#endif
            matches &= matches - 1;
        }
        if (useMasks ? (matchesJ & (1u << index)) != 0 : index != index1 && uvw[j][index] == uvw[j][index1]) {
            if (checkReduce) {
                int cmpI = uvw[i][index].compare(uvw[i][index1]);
                if (cmpI == 1 && uvw[k][index].limitSum(uvw[k][index1], k != 2)) {
                    reduceAdd(k, index, index1);
                    return;
                }

                if (i == 2 && cmpI == -1 && uvw[k][index].limitSub(uvw[k][index1], true)) {
                    reduceSub(k, index, index1);
                    return;
                }

                int cmpK = uvw[k][index].compare(uvw[k][index1]);
                if (cmpK == 1 && uvw[i][index].limitSum(uvw[i][index1], i != 2)) {
                    reduceAdd(i, index, index1);
                    return;
                }

                if (k == 2 && cmpK == -1 && uvw[i][index].limitSub(uvw[i][index1], true)) {
                    reduceSub(i, index, index1);
                    return;
                }
            }

            flips[j].add(index1, index);
        }

        if (useMasks ? (matchesK & (1u << index)) != 0 : index != index2 && uvw[k][index] == uvw[k][index2]) {
            if (checkReduce) {
                int cmpI = uvw[i][index].compare(uvw[i][index2]);
                if (cmpI == 1 && uvw[j][index].limitSum(uvw[j][index2], j != 2)) {
                    reduceAdd(j, index, index2);
                    return;
                }

                if (i == 2 && cmpI == -1 && uvw[j][index].limitSub(uvw[j][index2], true)) {
                    reduceSub(j, index, index2);
                    return;
                }

                int cmpJ = uvw[j][index].compare(uvw[j][index2]);
                if (cmpJ == 1 && uvw[i][index].limitSum(uvw[i][index2], i != 2)) {
                    reduceAdd(i, index, index2);
                    return;
                }

                if (j == 2 && cmpJ == -1 && uvw[i][index].limitSub(uvw[i][index2], true)) {
                    reduceSub(i, index, index2);
                    return;
                }
            }

            flips[k].add(index2, index);
        }
    }
}

void SchemeInteger::plus(int i, int j, int k, int index1, int index2, int variant) LOCAL_METHOD {
    const Addition a1 = uvw[i][index1];
    const Addition b1 = uvw[j][index1];
    const Addition c1 = uvw[k][index1];

    const Addition a2 = uvw[i][index2];
    const Addition b2 = uvw[j][index2];
    const Addition c2 = uvw[k][index2];

    const Addition aAdd = a1 + a2;
    const Addition bAdd = b1 + b2;
    const Addition cAdd = c1 + c2;

    const Addition aSub = a2 - a1;
    const Addition bSub = b2 - b1;
    const Addition cSub = c2 - c1;

    if (variant == 0 && aSub.limit(i != 2) && bAdd.limit(j != 2) && cSub.limit(k != 2)) {
        uvw[j][index1] = bAdd;
        uvw[i][index2] = aSub;
        addTriplet(i, j, k, a1, b2, cSub);
    }
    else if (variant == 1 && aSub.limit(i != 2) && bSub.limit(j != 2) && cAdd.limit(k != 2)) {
        uvw[k][index1] = cAdd;
        uvw[j][index2] = bSub;
        addTriplet(i, j, k, aSub, b1, c2);
    }
    else if (aAdd.limit(i != 2) && bSub.limit(j != 2) && cSub.limit(k != 2)) {
        uvw[i][index1] = aAdd;
        uvw[k][index2] = cSub;
        addTriplet(i, j, k, a2, bSub, c1);
    }

    removeZeroes();
    fixSigns();
    initFlips();
}

void SchemeInteger::split(int i, int j, int k, int index, LOCAL const Addition & addition) LOCAL_METHOD {
    addTriplet(i, j, k, uvw[i][index] - addition, uvw[j][index], uvw[k][index]);
    uvw[i][index] = addition;

    fixSigns();
    initFlips();
}

void SchemeInteger::reduceAdd(int i, int index1, int index2) LOCAL_METHOD {
    uvw[i][index1] += uvw[i][index2];
    bool isZero = !uvw[i][index1];

    removeAt(index2);

    if (isZero)
        removeZeroes();

    initFlips();
}

void SchemeInteger::reduceSub(int i, int index1, int index2) LOCAL_METHOD {
    uvw[i][index1] -= uvw[i][index2];
    bool isZero = !uvw[i][index1];

    removeAt(index2);

    if (isZero)
        removeZeroes();

    initFlips();
}

void SchemeInteger::project(int p, int q) LOCAL_METHOD {
    excludeRow(p, q);
    excludeColumn((p + 2) % 3, q);
    n[p]--;

    for (int i = 0; i < 3; i++)
        nn[i] = n[i] * n[(i + 1) % 3];

    removeZeroes();
    fixSigns();
    initFlips();
}

void SchemeInteger::extend(int p) LOCAL_METHOD {
    if (p == 0) {
        addRow(0);
        addColumn(2);

        for (int i = 0; i < n[2]; i++)
            for (int j = 0; j < n[1]; j++)
                addTriplet(0, 1, 2, Addition((n[0] + 1) * n[1], n[0] * n[1] + j), Addition(n[1] * n[2], j * n[2] + i), Addition(n[2] * (n[0] + 1), i * (n[0] + 1) + n[0]));
    }
    else if (p == 1) {
        addRow(1);
        addColumn(0);

        for (int i = 0; i < n[0]; i++)
            for (int j = 0; j < n[2]; j++)
                addTriplet(0, 1, 2, Addition(n[0] * (n[1] + 1), i * (n[1] + 1) + n[1]), Addition((n[1] + 1) * n[2], n[1] * n[2] + j), Addition(n[2] * n[0], j * n[0] + i));
    }
    else {
        addRow(2);
        addColumn(1);

        for (int i = 0; i < n[0]; i++)
            for (int j = 0; j < n[1]; j++)
                addTriplet(0, 1, 2, Addition(n[0] * n[1], i * n[1] + j), Addition(n[1] * (n[2] + 1), j * (n[2] + 1) + n[2]), Addition((n[2] + 1) * n[0], n[2] * n[0] + i));
    }

    n[p]++;

    for (int i = 0; i < 3; i++)
        nn[i] = n[i] * n[(i + 1) % 3];

    initFlips();
}

void SchemeInteger::product(int p) LOCAL_METHOD {
    int nNew[3] = {n[0], n[1], n[2]};
    int nnNew[3];
    int d[3];
    int mNew = m * 2;

    nNew[p] *= 2;

    for (int i = 0; i < 3; i++) {
        nnNew[i] = nNew[i] * nNew[(i + 1) % 3];
        d[i] = p == i ? n[i] : 0;
    }

    for (int index = 0; index < m; index++) {
        Addition u1(nnNew[0]), u2(nnNew[0]);
        Addition v1(nnNew[1]), v2(nnNew[1]);
        Addition w1(nnNew[2]), w2(nnNew[2]);

        for (int i = 0; i < n[0]; i++) {
            for (int j = 0; j < n[1]; j++) {
                int uij = uvw[0][index][i * n[1] + j];
                u1.set(i * nNew[1] + j, uij);
                u2.set((i + d[0]) * nNew[1] + j + d[1], uij);
            }
        }

        for (int i = 0; i < n[1]; i++) {
            for (int j = 0; j < n[2]; j++) {
                int vij = uvw[1][index][i * n[2] + j];
                v1.set(i * nNew[2] + j, vij);
                v2.set((i + d[1]) * nNew[2] + j + d[2], vij);
            }
        }

        for (int i = 0; i < n[2]; i++) {
            for (int j = 0; j < n[0]; j++) {
                int wij = uvw[2][index][i * n[0] + j];
                w1.set(i * nNew[0] + j, wij);
                w2.set((i + d[2]) * nNew[0] + j + d[0], wij);
            }
        }

        uvw[0][index] = u1;
        uvw[1][index] = v1;
        uvw[2][index] = w1;

        uvw[0][index + m] = u2;
        uvw[1][index + m] = v2;
        uvw[2][index + m] = w2;
    }

    for (int i = 0; i < 3; i++) {
        n[i] = nNew[i];
        nn[i] = nnNew[i];
    }

    m = mNew;
    initFlips();
}

void SchemeInteger::merge(GLOBAL const SchemeInteger &scheme, int p) LOCAL_METHOD {
    int nNew[3];
    int nnNew[3];
    int d[3];

    for (int i = 0; i < 3; i++) {
        nNew[i] = i == p ? n[i] + scheme.n[i] : n[i];
        d[i] = i == p ? n[i] : 0;
    }

    for (int i = 0; i < 3; i++)
        nnNew[i] = nNew[i] * nNew[(i + 1) % 3];

    for (int index = 0; index < m; index++) {
        Addition u(nnNew[0]);
        Addition v(nnNew[1]);
        Addition w(nnNew[2]);

        for (int i = 0; i < n[0]; i++)
            for (int j = 0; j < n[1]; j++)
                u.set(i * nNew[1] + j, uvw[0][index][i * n[1] + j]);

        for (int i = 0; i < n[1]; i++)
            for (int j = 0; j < n[2]; j++)
                v.set(i * nNew[2] + j, uvw[1][index][i * n[2] + j]);

        for (int i = 0; i < n[2]; i++)
            for (int j = 0; j < n[0]; j++)
                w.set(i * nNew[0] + j, uvw[2][index][i * n[0] + j]);

        uvw[0][index] = u;
        uvw[1][index] = v;
        uvw[2][index] = w;
    }

    for (int index = 0; index < scheme.m; index++) {
        Addition u(nnNew[0]);
        Addition v(nnNew[1]);
        Addition w(nnNew[2]);

        for (int i = 0; i < scheme.n[0]; i++)
            for (int j = 0; j < scheme.n[1]; j++)
                u.set((i + d[0]) * nNew[1] + j + d[1], scheme.uvw[0][index][i * scheme.n[1] + j]);

        for (int i = 0; i < scheme.n[1]; i++)
            for (int j = 0; j < scheme.n[2]; j++)
                v.set((i + d[1]) * nNew[2] + j + d[2], scheme.uvw[1][index][i * scheme.n[2] + j]);

        for (int i = 0; i < scheme.n[2]; i++)
            for (int j = 0; j < scheme.n[0]; j++)
                w.set((i + d[2]) * nNew[0] + j + d[0], scheme.uvw[2][index][i * scheme.n[0] + j]);

        uvw[0][m + index] = u;
        uvw[1][m + index] = v;
        uvw[2][m + index] = w;
    }

    for (int i = 0; i < 3; i++) {
        n[i] = nNew[i];
        nn[i] = nnNew[i];
    }

    m += scheme.m;

    initFlips();
}

void SchemeInteger::product(GLOBAL const SchemeInteger &scheme2) LOCAL_METHOD {
    int oldN[3] = {n[0], n[1], n[2]};
    int oldNN[3] = {nn[0], nn[1], nn[2]};
    int oldM = m;
    for (int p = 0; p < 3; p++) n[p] *= scheme2.n[p];
    for (int p = 0; p < 3; p++) nn[p] = n[p] * n[(p + 1) % 3];
    m *= scheme2.m;
    for (int index1 = oldM - 1; index1 >= 0; index1--) {
        Addition source[3] = {uvw[0][index1], uvw[1][index1], uvw[2][index1]};
        for (int index2 = 0; index2 < scheme2.m; index2++) {
            int index = index1 * scheme2.m + index2;
            for (int p = 0; p < 3; p++) {
                int p1 = (p + 1) % 3;
                uvw[p][index] = Addition(nn[p]);
                for (int i = 0; i < oldNN[p]; i++) {
                    for (int j = 0; j < scheme2.nn[p]; j++) {
                        int row = (i / oldN[p1]) * scheme2.n[p] + j / scheme2.n[p1];
                        int col = (i % oldN[p1]) * scheme2.n[p1] + j % scheme2.n[p1];
                        uvw[p][index].set(row * n[p1] + col, source[p][i] * scheme2.uvw[p][index2][j]);
                    }
                }
            }
        }
    }
    initFlips();
}

void SchemeInteger::swapBasisRows(int i1, int i2) LOCAL_METHOD {
    int rows[MAX_MATRIX_ELEMENTS];

    for (int row = 0; row < n[0]; row++)
        rows[row] = row;

    rows[i1] = i2;
    rows[i2] = i1;

    for (int index = 0; index < m; index++) {
        Addition u(n[0] * n[1]);
        Addition w(n[2] * n[0]);

        for (int i = 0; i < n[0]; i++)
            for (int j = 0; j < n[1]; j++)
                u.set(i * n[1] + j, uvw[0][index][rows[i] * n[1] + j]);

        for (int i = 0; i < n[2]; i++)
            for (int j = 0; j < n[0]; j++)
                w.set(i * n[0] + j, uvw[2][index][i * n[0] + rows[j]]);

        uvw[0][index] = u;
        uvw[2][index] = w;
    }
}

void SchemeInteger::swapBasisColumns(int j1, int j2) LOCAL_METHOD {
    int columns[MAX_MATRIX_ELEMENTS];

    for (int column = 0; column < n[2]; column++)
        columns[column] = column;

    columns[j1] = j2;
    columns[j2] = j1;

    for (int index = 0; index < m; index++) {
        Addition v(n[1] * n[2]);
        Addition w(n[2] * n[0]);

        for (int i = 0; i < n[1]; i++)
            for (int j = 0; j < n[2]; j++)
                v.set(i * n[2] + j, uvw[1][index][i * n[2] + columns[j]]);

        for (int i = 0; i < n[2]; i++)
            for (int j = 0; j < n[0]; j++)
                w.set(i * n[0] + j, uvw[2][index][columns[i] * n[0] + j]);

        uvw[1][index] = v;
        uvw[2][index] = w;
    }
}

void SchemeInteger::swapSize(int p1, int p2) LOCAL_METHOD {
    if (p1 == p2)
        return;

    if (p1 > p2) {
        int tmp = p1;
        p1 = p2;
        p2 = tmp;
    }

    int indices[3];
    int nNew[3];

    if (p1 == 0 && p2 == 1) {
        indices[0] = 0;
        indices[1] = 2;
        indices[2] = 1;
    }
    else if (p1 == 0 && p2 == 2) {
        indices[0] = 1;
        indices[1] = 0;
        indices[2] = 2;
    }
    else {
        indices[0] = 2;
        indices[1] = 1;
        indices[2] = 0;
    }

    for (int i = 0; i < 3; i++)
        nNew[i] = n[(indices[i] + 1) % 3];

    for (int index = 0; index < m; index++) {
        Addition u(nNew[0] * nNew[1]);
        Addition v(nNew[1] * nNew[2]);
        Addition w(nNew[2] * nNew[0]);

        for (int i = 0; i < nNew[0]; i++)
            for (int j = 0; j < nNew[1]; j++)
                u.set(i * nNew[1] + j, uvw[indices[0]][index][j * nNew[0] + i]);

        for (int i = 0; i < nNew[1]; i++)
            for (int j = 0; j < nNew[2]; j++)
                v.set(i * nNew[2] + j, uvw[indices[1]][index][j * nNew[1] + i]);

        for (int i = 0; i < nNew[2]; i++)
            for (int j = 0; j < nNew[0]; j++)
                w.set(i * nNew[0] + j, uvw[indices[2]][index][j * nNew[2] + i]);

        uvw[0][index] = u;
        uvw[1][index] = v;
        uvw[2][index] = w;
    }

    for (int i = 0; i < 3; i++) {
        n[i] = nNew[i];
        nn[i] = nNew[i] * nNew[(i + 1) % 3];
    }

    fixSigns();
    initFlips();
}

bool SchemeInteger::tryFlip(LOCAL RandomState &state, bool checkReduce) LOCAL_METHOD {
    int size = flips[0].size + flips[1].size + flips[2].size;
    int indices[MAX_PAIRS * 3];

    randomPermutation(indices, size, state);

    int i = 0, j = 0, k = 0, index1 = 0, index2 = 0;
    int p = 0;
    for (; p < size; p++) {
        int index = indices[p];

        if (index < flips[0].size) {
            i = 0;
            j = 1;
            k = 2;
            index1 = flips[0].index1(index);
            index2 = flips[0].index2(index);
        }
        else if (index < flips[0].size + flips[1].size) {
            i = 1;
            j = 0;
            k = 2;
            index1 = flips[1].index1(index - flips[0].size);
            index2 = flips[1].index2(index - flips[0].size);
        }
        else {
            i = 2;
            j = 0;
            k = 1;
            index1 = flips[2].index1(index - flips[0].size - flips[1].size);
            index2 = flips[2].index2(index - flips[0].size - flips[1].size);
        }

        if (randomWord(&state) % 2) {
            int tmp = j;
            j = k;
            k = tmp;
        }

        if (randomWord(&state) % 2) {
            int tmp = index1;
            index1 = index2;
            index2 = tmp;
        }

        if (uvw[j][index1].limitSum(uvw[j][index2], j != 2) && uvw[k][index2].limitSub(uvw[k][index1])) {
            if (k != 2 && !uvw[k][index2].positiveFirstNonZeroSub(uvw[k][index1])) {
                int tmp = index1;
                index1 = index2;
                index2 = tmp;
            }
            break;
        }

        if (uvw[k][index1].limitSum(uvw[k][index2], k != 2) && uvw[j][index2].limitSub(uvw[j][index1])) {
            if (j != 2 && !uvw[j][index2].positiveFirstNonZeroSub(uvw[j][index1])) {
                int tmp = index1;
                index1 = index2;
                index2 = tmp;
            }
            int tmp = j;
            j = k;
            k = tmp;
            break;
        }
    }

    if (p == size) return false;
    flip(i, j, k, index1, index2, checkReduce);
    return true;
}

bool SchemeInteger::tryPlus(LOCAL RandomState &state) LOCAL_METHOD {
    if (m >= MAX_RANK || m >= n[0] * n[1] * n[2])
        return false;

    bool candidate = false;
    for (int a = 0; a < m && !candidate; a++)
        for (int b = a + 1; b < m && !candidate; b++)
            candidate = uvw[0][a] != uvw[0][b] && uvw[1][a] != uvw[1][b] && uvw[2][a] != uvw[2][b];
    if (!candidate) return false;

    int index1 = randomWord(&state) % m;
    int index2 = randomWord(&state) % m;

    while (index1 == index2 || uvw[0][index1] == uvw[0][index2] || uvw[1][index1] == uvw[1][index2] || uvw[2][index1] == uvw[2][index2]) {
        index1 = randomWord(&state) % m;
        index2 = randomWord(&state) % m;
    }

    int permutation[3];
    randomPermutation(permutation, 3, state);
    int i = permutation[0];
    int j = permutation[1];
    int k = permutation[2];

    int variant = randomWord(&state) % 3;
    plus(i, j, k, index1, index2, variant);
    return true;
}

bool SchemeInteger::trySplit(LOCAL RandomState &state) LOCAL_METHOD {
    if (m >= MAX_RANK || m >= n[0] * n[1] * n[2])
        return false;

    int index = randomWord(&state) % m;

    int permutation[3];
    randomPermutation(permutation, 3, state);
    int i = permutation[0];
    int j = permutation[1];
    int k = permutation[2];

    Addition a(nn[i]);
    a.random(state);

    if (!uvw[i][index].limitSub(a, i != 2))
        return false;

    split(i, j, k, index, a);
    return true;
}

bool SchemeInteger::trySplitExisted(LOCAL RandomState &state) LOCAL_METHOD {
    if (m >= MAX_RANK || m >= n[0] * n[1] * n[2])
        return false;

    bool candidate = false;
    for (int p = 0; p < 3 && !candidate; p++)
        for (int a = 0; a < m && !candidate; a++)
            for (int b = a + 1; b < m && !candidate; b++)
                candidate = uvw[p][a] != uvw[p][b];
    if (!candidate) return false;

    int index1, index2;
    int i;

    do {
        index1 = randomWord(&state) % m;
        index2 = randomWord(&state) % m;
        i = randomWord(&state) % 3;
    } while (index1 == index2 || uvw[i][index1] == uvw[i][index2]);

    if (!uvw[i][index1].limitSub(uvw[i][index2], i != 2))
        return false;

    int j = (i + 1) % 3;
    int k = (i + 2) % 3;

    split(i, j, k, index1, uvw[i][index2]);
    return true;
}

bool SchemeInteger::tryExpand(int count, LOCAL RandomState &state) LOCAL_METHOD {
    bool result = false;

    for (int i = 0; i < count && m < MAX_RANK; i++) {
        int v = randomWord(&state) % 3;

        if (v == 0)
            result |= tryPlus(state);
        else if (v == 1)
            result |= trySplit(state);
        else
            result |= trySplitExisted(state);
    }

    return result;
}

bool SchemeInteger::tryReduce() LOCAL_METHOD {
    for (size_t i = 0; i < flips[0].size; i++) {
        int index1 = flips[0].index1(i);
        int index2 = flips[0].index2(i);

        if (uvw[1][index1] == uvw[1][index2] && uvw[2][index1].limitSum(uvw[2][index2], false)) {
            reduceAdd(2, index1, index2);
            return true;
        }

        int cmp2 = uvw[2][index1].compare(uvw[2][index2]);
        if (cmp2 == 1 && uvw[1][index1].limitSum(uvw[1][index2], true)) {
            reduceAdd(1, index1, index2);
            return true;
        }

        if (cmp2 == -1 && uvw[1][index1].limitSub(uvw[1][index2], true)) {
            reduceSub(1, index1, index2);
            return true;
        }
    }

    for (size_t i = 0; i < flips[1].size; i++) {
        int index1 = flips[1].index1(i);
        int index2 = flips[1].index2(i);
        int cmp2 = uvw[2][index1].compare(uvw[2][index2]);

        if (cmp2 == 1 && uvw[0][index1].limitSum(uvw[0][index2], true)) {
            reduceAdd(0, index1, index2);
            return true;
        }

        if (cmp2 == -1 && uvw[0][index1].limitSub(uvw[0][index2], true)) {
            reduceSub(0, index1, index2);
            return true;
        }
    }

    return false;
}

bool SchemeInteger::tryProject(LOCAL RandomState &state, int minN) LOCAL_METHOD {
    int indices[3];
    int size = 0;

    #pragma unroll
    for (int i = 0; i < 3; i++)
        if (isValidProject(i, minN))
            indices[size++] = i;

    if (!size)
        return false;

    int p = indices[randomWord(&state) % size];
    int q = randomWord(&state) % n[p];
    project(p, q);

    for (int remaining = m; remaining > 0; remaining--)
        if (!tryReduce()) break;

    return true;
}

bool SchemeInteger::tryExtend(LOCAL RandomState &state, int maxN) LOCAL_METHOD {
    int indices[3];
    int size = 0;

    #pragma unroll
    for (int i = 0; i < 3; i++)
        if (isValidExtension(i, maxN))
            indices[size++] = i;

    if (!size)
        return false;

    int p = indices[randomWord(&state) % size];
    extend(p);

    for (int remaining = m; remaining > 0; remaining--)
        if (!tryReduce()) break;

    return true;
}

bool SchemeInteger::tryProduct(LOCAL RandomState &state, int maxN) LOCAL_METHOD {
    int indices[3];
    int size = 0;

    #pragma unroll
    for (int i = 0; i < 3; i++)
        if (isValidProduct(i, maxN))
            indices[size++] = i;

    if (!size)
        return false;

    int p = indices[randomWord(&state) % size];
    product(p);
    return true;
}

bool SchemeInteger::tryMerge(GLOBAL const SchemeInteger &scheme, LOCAL RandomState &state) LOCAL_METHOD {
    if (m + scheme.m > MAX_RANK)
        return false;

    int eq1 = n[0] == scheme.n[0];
    int eq2 = n[1] == scheme.n[1];
    int eq3 = n[2] == scheme.n[2];

    int n1 = n[0] + scheme.n[0];
    int n2 = n[1] + scheme.n[1];
    int n3 = n[2] + scheme.n[2];

    int p[3];
    int size = 0;

    if (n1 <= MAX_EXTENSION_N && n1 * n[1] <= MAX_MATRIX_ELEMENTS && n1 * n[2] <= MAX_MATRIX_ELEMENTS && eq2 && eq3)
        p[size++] = 0;

    if (eq1 && n2 <= MAX_EXTENSION_N && n[0] * n2 <= MAX_MATRIX_ELEMENTS && n[2] * n2 <= MAX_MATRIX_ELEMENTS && eq3)
        p[size++] = 1;

    if (eq1 && eq2 && n3 <= MAX_EXTENSION_N && n[0] * n3 <= MAX_MATRIX_ELEMENTS && n[1] * n3 <= MAX_MATRIX_ELEMENTS)
        p[size++] = 2;

    if (size == 0)
        return false;

    merge(scheme, p[randomWord(&state) % size]);
    return true;
}

bool SchemeInteger::tryProduct(GLOBAL const SchemeInteger &scheme) LOCAL_METHOD {
    if (m * scheme.m > MAX_RANK)
        return false;

    int sizes[3];

    for (int i = 0; i < 3; i++)
        sizes[i] = n[i] * scheme.n[i];

    for (int i = 0; i < 3; i++) {
        if (sizes[i] > MAX_EXTENSION_N)
            return false;

        if (sizes[i] * sizes[(i + 1) % 3] > MAX_MATRIX_ELEMENTS)
            return false;
    }

    product(scheme);
    return true;
}

void SchemeInteger::swapBasis(LOCAL RandomState &state) LOCAL_METHOD {
    if (randomWord(&state) % 2) {
        int i1 = randomWord(&state) % n[0];
        int i2 = randomWord(&state) % n[0];
        swapBasisRows(i1, i2);
    }
    else {
        int j1 = randomWord(&state) % n[2];
        int j2 = randomWord(&state) % n[2];
        swapBasisColumns(j1, j2);
    }

    if (fixSigns())
        initFlips();
}

void SchemeInteger::swapSize(LOCAL RandomState &state) LOCAL_METHOD {
    int p1, p2;

    do {
        p1 = randomWord(&state) % 3;
        p2 = randomWord(&state) % 3;
    } while (p1 == p2);

    swapSize(p1, p2);
}

#ifndef __METAL_VERSION__
void SchemeInteger::saveMatrix(std::ofstream &f, std::string name, int m, LOCAL const Addition *additions) const {
    f << "    \"" << name << "\": [" << std::endl;

    for (int index = 0; index < m; index++)
        f << "        [" << additions[index] << "]" << (index < m - 1 ? "," : "") << std::endl;

    f << "    ]";
}
#endif

#ifndef __METAL_VERSION__
void SchemeInteger::save(const std::string &path) {
    if (!validate()) throw std::runtime_error("refusing to export invalid scheme");
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot open output: " + path);

    f << "{" << std::endl;
    f << "    \"n\": [" << n[0] << ", " << n[1] << ", " << n[2] << "]," << std::endl;
    f << "    \"m\": " << m << "," << std::endl;
    f << "    \"z2\": false," << std::endl;
    f << "    \"complexity\": " << getComplexity() << "," << std::endl;

    saveMatrix(f, "u", m, uvw[0]);
    f << "," << std::endl;
    saveMatrix(f, "v", m, uvw[1]);
    f << "," << std::endl;
    saveMatrix(f, "w", m, uvw[2]);
    f << std::endl;
    f << "}" << std::endl;

    f.close();
    if (!f) throw std::runtime_error("cannot write output: " + path);
}
#endif

#ifndef __METAL_VERSION__
void SchemeInteger::showTensor(LOCAL const Addition &addition, int n1, int n2, std::string name, bool transpose) const {
    bool printed = false;

    std::cout << "(";

    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            auto value = addition[i * n2 + j];

            if (!value)
                continue;

            if (printed)
                std::cout << " ";

            if (value < 0)
                std::cout << "- ";
            else if (printed)
                std::cout << "+ ";

            if (value > 1)
                std::cout << value;
            else if (value < -1)
                std::cout << (-value);

            std::cout << name;

            if (transpose)
                std::cout << (j + 1) << (i + 1);
            else
                std::cout << (i + 1) << (j + 1);

            printed = true;
        }
    }

    std::cout << ")";
}
#endif

#ifndef __METAL_VERSION__
void SchemeInteger::show() const {
    for (int index = 0; index < m; index++) {
        showTensor(uvw[0][index], n[0], n[1], "a", false);
        std::cout << " x ";
        showTensor(uvw[1][index], n[1], n[2], "b", false);
        std::cout << " x ";
        showTensor(uvw[2][index], n[2], n[0], "c", true);
        std::cout << std::endl;
    }
}

#endif
