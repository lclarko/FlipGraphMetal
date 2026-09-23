#pragma once
#if defined(__METAL_VERSION__) && !defined(METAL_F2)
static_assert(MAX_RANK == 350 && MAX_PAIRS == 500, "compact buffer allocation contract changed");

struct CompactAddition {
    ushort n;
    ushort values;
    ushort signs;
    bool valid;

    CompactAddition() LOCAL_METHOD;
    CompactAddition(int n) LOCAL_METHOD;
    CompactAddition(int n, int index) LOCAL_METHOD;
    CompactAddition(int n, LOCAL int *values) LOCAL_METHOD;

    void copyTo(LOCAL CompactAddition &target) const LOCAL_METHOD;
    void set(int index, int value) LOCAL_METHOD;
    void inverse() LOCAL_METHOD;
    void random(LOCAL RandomState &state) LOCAL_METHOD;

    int compare(CompactAddition addition) const LOCAL_METHOD;

    bool operator==(CompactAddition addition) const LOCAL_METHOD;
    bool operator!=(CompactAddition addition) const LOCAL_METHOD;
    int operator[](int index) const LOCAL_METHOD;
    int operator[](int index) const device {
        int value = int((values >> index) & 1);
        return ((signs >> index) & 1) ? -value : value;
    }

    CompactAddition operator+(CompactAddition addition) const LOCAL_METHOD;
    CompactAddition operator-(CompactAddition addition) const LOCAL_METHOD;
    CompactAddition operator-() const LOCAL_METHOD;

    LOCAL CompactAddition & operator+=(CompactAddition addition) LOCAL_METHOD;
    LOCAL CompactAddition & operator-=(CompactAddition addition) LOCAL_METHOD;

    bool limit(bool firstPositiveNonZero) const LOCAL_METHOD;
    bool limitSum(CompactAddition addition, bool firstPositiveNonZero) const LOCAL_METHOD;
    bool limitSub(CompactAddition addition, bool firstPositiveNonZero = false) const LOCAL_METHOD;

    bool positiveFirstNonZero() const LOCAL_METHOD;
    bool positiveFirstNonZeroSub(CompactAddition addition) const LOCAL_METHOD;

    int nonZeroCount() const LOCAL_METHOD;

    operator bool() const LOCAL_METHOD;

    void copyTo(LOCAL CompactAddition &target) const device;
    void copyTo(device CompactAddition &target) const LOCAL_METHOD;
    void copyTo(device CompactAddition &target) const device;
    void set(int index, int value) device;
    void inverse() device;
    void random(LOCAL RandomState &state) device;
    int compare(CompactAddition addition) const device;
    bool operator==(CompactAddition addition) const device;
    bool operator!=(CompactAddition addition) const device;
    CompactAddition operator+(CompactAddition addition) const device;
    CompactAddition operator-(CompactAddition addition) const device;
    CompactAddition operator-() const device;
    device CompactAddition & operator+=(CompactAddition addition) device;
    device CompactAddition & operator-=(CompactAddition addition) device;
    operator bool() const device;
    bool limit(bool firstPositiveNonZero) const device;
    bool limitSum(CompactAddition addition, bool firstPositiveNonZero) const device;
    bool limitSub(CompactAddition addition, bool firstPositiveNonZero = false) const device;
    bool positiveFirstNonZeroSub(CompactAddition addition) const device;
    bool positiveFirstNonZero() const device;
    int nonZeroCount() const device;
};

CompactAddition::CompactAddition() LOCAL_METHOD {
    n = 0;
    values = 0;
    signs = 0;
    valid = true;
}

CompactAddition::CompactAddition(int n) LOCAL_METHOD {
    this->n = n;
    this->values = 0;
    this->signs = 0;
    this->valid = true;
}

CompactAddition::CompactAddition(int n, int index) LOCAL_METHOD {
    this->n = n;
    this->values = ushort(1) << index;
    this->signs = 0;
    this->valid = true;
}

CompactAddition::CompactAddition(int n, LOCAL int *values) LOCAL_METHOD {
    this->n = n;
    this->values = 0;
    this->valid = true;
    this->signs = 0;

    for (int i = 0; i < n; i++)
        set(i, values[i]);
}

void CompactAddition::copyTo(LOCAL CompactAddition &target) const LOCAL_METHOD {
    target.n = n;
    target.values = values;
    target.signs = signs;
    target.valid = valid;
}

void CompactAddition::set(int index, int value) LOCAL_METHOD {
    ushort mask = ushort(1) << index;

    if (value == 0) {
        values &= ~mask;
        signs &= ~mask;
    }
    else if (value == 1) {
        values |= mask;
        signs &= ~mask;
    }
    else if (value == -1) {
        values |= mask;
        signs |= mask;
    }
    else {
        valid = false;

    }
}

void CompactAddition::inverse() LOCAL_METHOD {
    signs = (~signs) & values;
}

void CompactAddition::random(LOCAL RandomState &state) LOCAL_METHOD {
    values = 0;
    signs = 0;

    for (int i = 0; i < n; i += 16) {
        uint16_t signBits = randomWord(&state) & 0xFFFF;
        uint16_t valuesBits = randomWord(&state) & 0xFFFF;

        values |= ushort(valuesBits) << i;
        signs |= ushort(signBits) << i;
    }

    if (n < 64) values &= (ushort(1) << n) - 1;
    signs &= values;
    valid = true;
}

int CompactAddition::compare(CompactAddition addition) const LOCAL_METHOD {
    if (values != addition.values)
        return 0;

    if (signs == addition.signs)
        return 1;

    if (signs == ((~addition.signs) & addition.values))
        return -1;

    return 0;
}

bool CompactAddition::operator==(CompactAddition addition) const LOCAL_METHOD {
    return values == addition.values && signs == addition.signs;
}

bool CompactAddition::operator!=(CompactAddition addition) const LOCAL_METHOD {
    return values != addition.values || signs != addition.signs;
}

int CompactAddition::operator[](int index) const LOCAL_METHOD {
    int value = int((values >> index) & 1);

    if ((signs >> index) & 1)
        value = -value;

    return value;
}

CompactAddition CompactAddition::operator+(CompactAddition addition) const LOCAL_METHOD {
    CompactAddition result(n);

    result.values = values ^ addition.values;
    result.signs = ((signs & values) | (addition.signs & addition.values)) & result.values;
    result.valid = !(values & addition.values & ~(signs ^ addition.signs));
    return result;
}

CompactAddition CompactAddition::operator-(CompactAddition addition) const LOCAL_METHOD {
    CompactAddition result(n);

    result.values = values ^ addition.values;
    result.signs = ((signs & values) | (~addition.signs & addition.values)) & result.values;
    result.valid = !(values & addition.values & (signs ^ addition.signs));
    return result;
}

CompactAddition CompactAddition::operator-() const LOCAL_METHOD {
    CompactAddition result(n);
    result.values = values;
    result.signs = (~signs) & values;
    result.valid = valid;
    return result;
}

LOCAL CompactAddition & CompactAddition::operator+=(CompactAddition addition) LOCAL_METHOD {
    ushort sv1 = signs & values;
    ushort sv2 = addition.signs & addition.values;

    valid = !((values & addition.values & ~(signs ^ addition.signs)));
    values ^= addition.values;
    signs = (sv1 | sv2) & values;
    return *this;
}

LOCAL CompactAddition & CompactAddition::operator-=(CompactAddition addition) LOCAL_METHOD {
    ushort sv1 = signs & values;
    ushort sv2 = ~addition.signs & addition.values;

    valid = !(values & addition.values & (signs ^ addition.signs));
    values ^= addition.values;
    signs = (sv1 | sv2) & values;
    return *this;
}

CompactAddition::operator bool() const LOCAL_METHOD {
    return values != 0;
}

bool CompactAddition::limit(bool firstPositiveNonZero) const LOCAL_METHOD {
    if (!valid)
        return false;

    if (firstPositiveNonZero)
        return values == 0 || (values & ~(values & (values - 1)) & ~signs);

    return true;
}

bool CompactAddition::limitSum(CompactAddition addition, bool firstPositiveNonZero) const LOCAL_METHOD {
    bool invalid = values & addition.values & ~(signs ^ addition.signs);

    if (invalid)
        return false;

    if (firstPositiveNonZero) {
        ushort sumValues = values ^ addition.values;
        ushort sumSigns = ((signs & values) | (addition.signs & addition.values)) & sumValues;

        return sumValues == 0 || (sumValues & ~(sumValues & (sumValues - 1)) & ~sumSigns);
    }

    return true;
}

bool CompactAddition::limitSub(CompactAddition addition, bool firstPositiveNonZero) const LOCAL_METHOD {
    bool invalid = (values & addition.values & (signs ^ addition.signs));

    if (invalid)
        return false;

    if (firstPositiveNonZero) {
        ushort subValues = values ^ addition.values;
        ushort subSigns = ((signs & values) | (~addition.signs & addition.values)) & subValues;

        return subValues == 0 || (subValues & ~(subValues & (subValues - 1)) & ~subSigns);
    }

    return true;
}

bool CompactAddition::positiveFirstNonZeroSub(CompactAddition addition) const LOCAL_METHOD {
    ushort subValues = values ^ addition.values;
    ushort subSigns = ((signs & values) | (~addition.signs & addition.values)) & subValues;

    return subValues == 0 || (subValues & ~(subValues & (subValues - 1)) & ~subSigns);
}

bool CompactAddition::positiveFirstNonZero() const LOCAL_METHOD {
    return values == 0 || (values & ~(values & (values - 1)) & ~signs);
}

int CompactAddition::nonZeroCount() const LOCAL_METHOD {
    return popcount(values);
}


void CompactAddition::copyTo(LOCAL CompactAddition &target) const device {
    target.n = n;
    target.values = values;
    target.signs = signs;
    target.valid = valid;
}

void CompactAddition::copyTo(device CompactAddition &target) const LOCAL_METHOD {
    target.n = n;
    target.values = values;
    target.signs = signs;
    target.valid = valid;
}

void CompactAddition::copyTo(device CompactAddition &target) const device {
    target.n = n;
    target.values = values;
    target.signs = signs;
    target.valid = valid;
}

void CompactAddition::set(int index, int value) device {
    ushort mask = ushort(1) << index;

    if (value == 0) {
        values &= ~mask;
        signs &= ~mask;
    }
    else if (value == 1) {
        values |= mask;
        signs &= ~mask;
    }
    else if (value == -1) {
        values |= mask;
        signs |= mask;
    }
    else {
        valid = false;

    }
}

void CompactAddition::inverse() device {
    signs = (~signs) & values;
}

void CompactAddition::random(LOCAL RandomState &state) device {
    values = 0;
    signs = 0;

    for (int i = 0; i < n; i += 16) {
        uint16_t signBits = randomWord(&state) & 0xFFFF;
        uint16_t valuesBits = randomWord(&state) & 0xFFFF;

        values |= ushort(valuesBits) << i;
        signs |= ushort(signBits) << i;
    }

    if (n < 64) values &= (ushort(1) << n) - 1;
    signs &= values;
    valid = true;
}

int CompactAddition::compare(CompactAddition addition) const device {
    if (values != addition.values)
        return 0;

    if (signs == addition.signs)
        return 1;

    if (signs == ((~addition.signs) & addition.values))
        return -1;

    return 0;
}

bool CompactAddition::operator==(CompactAddition addition) const device {
    return values == addition.values && signs == addition.signs;
}

bool CompactAddition::operator!=(CompactAddition addition) const device {
    return values != addition.values || signs != addition.signs;
}

CompactAddition CompactAddition::operator+(CompactAddition addition) const device {
    CompactAddition result(n);

    result.values = values ^ addition.values;
    result.signs = ((signs & values) | (addition.signs & addition.values)) & result.values;
    result.valid = !(values & addition.values & ~(signs ^ addition.signs));
    return result;
}

CompactAddition CompactAddition::operator-(CompactAddition addition) const device {
    CompactAddition result(n);

    result.values = values ^ addition.values;
    result.signs = ((signs & values) | (~addition.signs & addition.values)) & result.values;
    result.valid = !(values & addition.values & (signs ^ addition.signs));
    return result;
}

CompactAddition CompactAddition::operator-() const device {
    CompactAddition result(n);
    result.values = values;
    result.signs = (~signs) & values;
    result.valid = valid;
    return result;
}

device CompactAddition & CompactAddition::operator+=(CompactAddition addition) device {
    ushort sv1 = signs & values;
    ushort sv2 = addition.signs & addition.values;

    valid = !((values & addition.values & ~(signs ^ addition.signs)));
    values ^= addition.values;
    signs = (sv1 | sv2) & values;
    return *this;
}

device CompactAddition & CompactAddition::operator-=(CompactAddition addition) device {
    ushort sv1 = signs & values;
    ushort sv2 = ~addition.signs & addition.values;

    valid = !(values & addition.values & (signs ^ addition.signs));
    values ^= addition.values;
    signs = (sv1 | sv2) & values;
    return *this;
}

CompactAddition::operator bool() const device {
    return values != 0;
}

bool CompactAddition::limit(bool firstPositiveNonZero) const device {
    if (!valid)
        return false;

    if (firstPositiveNonZero)
        return values == 0 || (values & ~(values & (values - 1)) & ~signs);

    return true;
}

bool CompactAddition::limitSum(CompactAddition addition, bool firstPositiveNonZero) const device {
    bool invalid = values & addition.values & ~(signs ^ addition.signs);

    if (invalid)
        return false;

    if (firstPositiveNonZero) {
        ushort sumValues = values ^ addition.values;
        ushort sumSigns = ((signs & values) | (addition.signs & addition.values)) & sumValues;

        return sumValues == 0 || (sumValues & ~(sumValues & (sumValues - 1)) & ~sumSigns);
    }

    return true;
}

bool CompactAddition::limitSub(CompactAddition addition, bool firstPositiveNonZero) const device {
    bool invalid = (values & addition.values & (signs ^ addition.signs));

    if (invalid)
        return false;

    if (firstPositiveNonZero) {
        ushort subValues = values ^ addition.values;
        ushort subSigns = ((signs & values) | (~addition.signs & addition.values)) & subValues;

        return subValues == 0 || (subValues & ~(subValues & (subValues - 1)) & ~subSigns);
    }

    return true;
}

bool CompactAddition::positiveFirstNonZeroSub(CompactAddition addition) const device {
    ushort subValues = values ^ addition.values;
    ushort subSigns = ((signs & values) | (~addition.signs & addition.values)) & subValues;

    return subValues == 0 || (subValues & ~(subValues & (subValues - 1)) & ~subSigns);
}

bool CompactAddition::positiveFirstNonZero() const device {
    return values == 0 || (values & ~(values & (values - 1)) & ~signs);
}

int CompactAddition::nonZeroCount() const device {
    return popcount(values);
}

static_assert(sizeof(CompactAddition) == 8, "compact Addition stride changed");

inline void randomPermutationStrided(device int *array, int n, LOCAL RandomState &state) {
    for (int i = 0; i < n; i++) array[(i) * 32] = i;
    for (int i = n - 1; i > 0; i--) {
        int j = randomWord(&state) % (i + 1);
        int tmp = array[(i) * 32]; array[(i) * 32] = array[(j) * 32]; array[(j) * 32] = tmp;
    }
}

static_assert(sizeof(Addition) == 32, "interleaved Addition stride changed");
struct CompactFlipSet {
    uint32_t size;
    uint32_t overflow;
    device uint32_t *pairs;
    void add(uint32_t index1, uint32_t index2) LOCAL_METHOD;
    void remove(uint32_t index1, uint32_t index2) LOCAL_METHOD;
    void remove(uint32_t index) LOCAL_METHOD;
    void clear() LOCAL_METHOD;
    uint32_t index1(size_t i) const LOCAL_METHOD;
    uint32_t index2(size_t i) const LOCAL_METHOD;
};
void CompactFlipSet::add(uint32_t index1, uint32_t index2) LOCAL_METHOD {
    if (size >= MAX_PAIRS) {
        overflow = 1;
        return;
    }

    uint32_t pair = (index1 << 16) | index2;
    pairs[(size++) * 32] = pair;
}

void CompactFlipSet::remove(uint32_t index1, uint32_t index2) LOCAL_METHOD {
    uint32_t pair1 = (index1 << 16) | index2;
    uint32_t pair2 = (index2 << 16) | index1;

    for (size_t i = 0; i < size; i++) {
        if (pairs[(i) * 32] == pair1 || pairs[(i) * 32] == pair2) {
            pairs[(i) * 32] = pairs[(--size) * 32];
            return;
        }
    }
}

void CompactFlipSet::remove(uint32_t index) LOCAL_METHOD {
    for (size_t i = 0; i < size; i++)
        if (index1(i) == index || index2(i) == index)
            pairs[(i--) * 32] = pairs[(--size) * 32];
}

void CompactFlipSet::clear() LOCAL_METHOD {
    size = 0;
}

uint32_t CompactFlipSet::index1(size_t i) const LOCAL_METHOD {
    return pairs[(i) * 32] >> 16;
}

uint32_t CompactFlipSet::index2(size_t i) const LOCAL_METHOD {
    return pairs[(i) * 32] & 0xFFFF;
}
// Compact kernel admission requires n=9 and nine-bit values/signs. Every
// created term also has n=9. Preserve signs independently of values and validity.
struct CompactStoredAddition {
    uint packed;

    CompactAddition read() const device {
        CompactAddition result(9);
        result.values = ushort(packed & 511u);
        result.signs = ushort((packed >> 9) & 511u);
        result.valid = bool((packed >> 18) & 1u);
        return result;
    }
    operator CompactAddition() const device { return read(); }
    device CompactStoredAddition &operator=(CompactAddition value) device {
        packed = uint(value.values) | (uint(value.signs) << 9) | (uint(value.valid) << 18);
        return *this;
    }
    device CompactStoredAddition &operator=(const device CompactStoredAddition &value) device {
        packed = value.packed;
        return *this;
    }
    operator bool() const device { return (packed & 511u) != 0; }
    int operator[](int index) const device {
        int value = int((packed >> index) & 1u);
        return ((packed >> (index + 9)) & 1u) ? -value : value;
    }
    void inverse() device {
        CompactAddition value = read();
        value.inverse();
        *this = value;
    }
    device CompactStoredAddition &operator+=(CompactAddition other) device {
        CompactAddition value = read();
        value += other;
        *this = value;
        return *this;
    }
    device CompactStoredAddition &operator-=(CompactAddition other) device {
        CompactAddition value = read();
        value -= other;
        *this = value;
        return *this;
    }
    int compare(CompactAddition other) const device { CompactAddition value = read(); return value.compare(other); }
    bool operator==(CompactAddition other) const device { CompactAddition value = read(); return value.operator==(other); }
    bool operator!=(CompactAddition other) const device { CompactAddition value = read(); return value.operator!=(other); }
    CompactAddition operator+(CompactAddition other) const device { CompactAddition value = read(); return value.operator+(other); }
    CompactAddition operator-(CompactAddition other) const device { CompactAddition value = read(); return value.operator-(other); }
    bool limitSum(CompactAddition other, bool first = false) const device { CompactAddition value = read(); return value.limitSum(other, first); }
    bool limitSub(CompactAddition other, bool first = false) const device { CompactAddition value = read(); return value.limitSub(other, first); }
    bool positiveFirstNonZeroSub(CompactAddition other) const device { CompactAddition value = read(); return value.positiveFirstNonZeroSub(other); }
    bool positiveFirstNonZero() const device { CompactAddition value = read(); return value.positiveFirstNonZero(); }
};
static_assert(sizeof(CompactStoredAddition) == 4, "packed device term stride changed");

struct CompactScheme {
    device CompactStoredAddition &term(int p, int r) const thread { return terms[(p * MAX_RANK + r) * 32]; }
    device CompactStoredAddition *terms;
    CompactFlipSet flips[3];
    device int *scratch;
    int n[3], nn[3], m;
    bool validateEquation(int i, int j, int k) const LOCAL_METHOD;
    bool validate() const LOCAL_METHOD;
    void initFlips() LOCAL_METHOD;
    void removeZeroes() LOCAL_METHOD;
    void removeAt(int index) LOCAL_METHOD;
    void addTriplet(int i, int j, int k, CompactAddition u, CompactAddition v, CompactAddition w) LOCAL_METHOD;
    bool fixSigns() LOCAL_METHOD;
    void flip(int i, int j, int k, int index1, int index2, bool checkReduce) LOCAL_METHOD;
    void plus(int i, int j, int k, int index1, int index2, int variant) LOCAL_METHOD;
    void split(int i, int j, int k, int index, CompactAddition addition) LOCAL_METHOD;
    void reduceAdd(int i, int index1, int index2) LOCAL_METHOD;
    void reduceSub(int i, int index1, int index2) LOCAL_METHOD;
    bool tryFlip(LOCAL RandomState &state, bool checkReduce = true) LOCAL_METHOD;
    bool tryPlus(LOCAL RandomState &state) LOCAL_METHOD;
    bool trySplit(LOCAL RandomState &state) LOCAL_METHOD;
    bool trySplitExisted(LOCAL RandomState &state) LOCAL_METHOD;
    bool tryExpand(int count, LOCAL RandomState &state) LOCAL_METHOD;
    bool tryReduce() LOCAL_METHOD;
};
bool CompactScheme::validateEquation(int i, int j, int k) const LOCAL_METHOD {
    int i1 = i / n[1];
    int i2 = i % n[1];
    int j1 = j / n[2];
    int j2 = j % n[2];
    int k1 = k / n[0];
    int k2 = k % n[0];

    int target = (i2 == j1) && (i1 == k2) && (j2 == k1);
    int equation = 0;

    for (int index = 0; index < m; index++)
        equation += term(0, index)[i] * term(1, index)[j] * term(2, index)[k];

    return equation == target;
}

bool CompactScheme::validate() const LOCAL_METHOD {
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
            if (!term(i, index).read().valid || (nn[i] < 64 && (term(i, index).read().values >> nn[i]) != 0)) {

                return false;
            }
        }
    }

    return valid;
}

void CompactScheme::initFlips() LOCAL_METHOD {
    for (int i = 0; i < 3; i++) {
        flips[i].clear();

        for (int index1 = 0; index1 < m; index1++)
            for (int index2 = index1 + 1; index2 < m; index2++)
                if (term(i, index1) == term(i, index2))
                    flips[i].add(index1, index2);
    }
}

void CompactScheme::removeZeroes() LOCAL_METHOD {
    for (int index = 0; index < m; index++) {
        if (term(0, index) && term(1, index) && term(2, index))
            continue;

        m--;
        term(0, index) = term(0, m);
        term(1, index) = term(1, m);
        term(2, index) = term(2, m);
        index--;
    }
}

void CompactScheme::removeAt(int index) LOCAL_METHOD {
    m--;

    if (index == m)
        return;

    term(0, index) = term(0, m);
    term(1, index) = term(1, m);
    term(2, index) = term(2, m);
}

void CompactScheme::addTriplet(int i, int j, int k, CompactAddition u, CompactAddition v, CompactAddition w) LOCAL_METHOD {
    term(i, m) = u;
    term(j, m) = v;
    term(k, m) = w;
    m++;
}

bool CompactScheme::fixSigns() LOCAL_METHOD {
    bool changed = false;

    for (int index = 0; index < m; index++) {
        bool i = term(0, index).positiveFirstNonZero();
        bool j = term(1, index).positiveFirstNonZero();

        if (i && j)
            continue;

        if (!i && !j) {
            term(0, index).inverse();
            term(1, index).inverse();
        }
        else if (!i) {
            term(0, index).inverse();
            term(2, index).inverse();
        }
        else {
            term(1, index).inverse();
            term(2, index).inverse();
        }

        changed = true;
    }

    return changed;
}

void CompactScheme::flip(int i, int j, int k, int index1, int index2, bool checkReduce) LOCAL_METHOD {
    term(j, index1) += term(j, index2);
    term(k, index2) -= term(k, index1);

    flips[j].remove(index1);
    flips[k].remove(index2);

    if (!term(j, index1) || !term(k, index2)) {
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
            matchesJ |= uint32_t(term(j, index) == term(j, index1)) << index;
            matchesK |= uint32_t(term(k, index) == term(k, index2)) << index;
        }
        matchesJ &= ~(1u << index1);
        matchesK &= ~(1u << index2);
    }
    uint32_t matches = matchesJ | matchesK;
    for (int index = 0; index < m; index++) {
        if (useMasks) {
            if (!matches) break;
            index = ctz(matches);
            matches &= matches - 1;
        }
        if (useMasks ? (matchesJ & (1u << index)) != 0 : index != index1 && term(j, index) == term(j, index1)) {
            if (checkReduce) {
                int cmpI = term(i, index).compare(term(i, index1));
                if (cmpI == 1 && term(k, index).limitSum(term(k, index1), k != 2)) {
                    reduceAdd(k, index, index1);
                    return;
                }

                if (i == 2 && cmpI == -1 && term(k, index).limitSub(term(k, index1), true)) {
                    reduceSub(k, index, index1);
                    return;
                }

                int cmpK = term(k, index).compare(term(k, index1));
                if (cmpK == 1 && term(i, index).limitSum(term(i, index1), i != 2)) {
                    reduceAdd(i, index, index1);
                    return;
                }

                if (k == 2 && cmpK == -1 && term(i, index).limitSub(term(i, index1), true)) {
                    reduceSub(i, index, index1);
                    return;
                }
            }

            flips[j].add(index1, index);
        }

        if (useMasks ? (matchesK & (1u << index)) != 0 : index != index2 && term(k, index) == term(k, index2)) {
            if (checkReduce) {
                int cmpI = term(i, index).compare(term(i, index2));
                if (cmpI == 1 && term(j, index).limitSum(term(j, index2), j != 2)) {
                    reduceAdd(j, index, index2);
                    return;
                }

                if (i == 2 && cmpI == -1 && term(j, index).limitSub(term(j, index2), true)) {
                    reduceSub(j, index, index2);
                    return;
                }

                int cmpJ = term(j, index).compare(term(j, index2));
                if (cmpJ == 1 && term(i, index).limitSum(term(i, index2), i != 2)) {
                    reduceAdd(i, index, index2);
                    return;
                }

                if (j == 2 && cmpJ == -1 && term(i, index).limitSub(term(i, index2), true)) {
                    reduceSub(i, index, index2);
                    return;
                }
            }

            flips[k].add(index2, index);
        }
    }
}

void CompactScheme::plus(int i, int j, int k, int index1, int index2, int variant) LOCAL_METHOD {
    const CompactAddition a1 = term(i, index1);
    const CompactAddition b1 = term(j, index1);
    const CompactAddition c1 = term(k, index1);

    const CompactAddition a2 = term(i, index2);
    const CompactAddition b2 = term(j, index2);
    const CompactAddition c2 = term(k, index2);

    const CompactAddition aAdd = a1 + a2;
    const CompactAddition bAdd = b1 + b2;
    const CompactAddition cAdd = c1 + c2;

    const CompactAddition aSub = a2 - a1;
    const CompactAddition bSub = b2 - b1;
    const CompactAddition cSub = c2 - c1;

    if (variant == 0 && aSub.limit(i != 2) && bAdd.limit(j != 2) && cSub.limit(k != 2)) {
        term(j, index1) = bAdd;
        term(i, index2) = aSub;
        addTriplet(i, j, k, a1, b2, cSub);
    }
    else if (variant == 1 && aSub.limit(i != 2) && bSub.limit(j != 2) && cAdd.limit(k != 2)) {
        term(k, index1) = cAdd;
        term(j, index2) = bSub;
        addTriplet(i, j, k, aSub, b1, c2);
    }
    else if (aAdd.limit(i != 2) && bSub.limit(j != 2) && cSub.limit(k != 2)) {
        term(i, index1) = aAdd;
        term(k, index2) = cSub;
        addTriplet(i, j, k, a2, bSub, c1);
    }

    removeZeroes();
    fixSigns();
    initFlips();
}

void CompactScheme::split(int i, int j, int k, int index, CompactAddition addition) LOCAL_METHOD {
    addTriplet(i, j, k, term(i, index) - addition, term(j, index), term(k, index));
    term(i, index) = addition;

    fixSigns();
    initFlips();
}

void CompactScheme::reduceAdd(int i, int index1, int index2) LOCAL_METHOD {
    term(i, index1) += term(i, index2);
    bool isZero = !term(i, index1);

    removeAt(index2);

    if (isZero)
        removeZeroes();

    initFlips();
}

void CompactScheme::reduceSub(int i, int index1, int index2) LOCAL_METHOD {
    term(i, index1) -= term(i, index2);
    bool isZero = !term(i, index1);

    removeAt(index2);

    if (isZero)
        removeZeroes();

    initFlips();
}

bool CompactScheme::tryFlip(LOCAL RandomState &state, bool checkReduce) LOCAL_METHOD {
    int size = flips[0].size + flips[1].size + flips[2].size;
    device int *indices = scratch;

    randomPermutationStrided(indices, size, state);

    int i = 0, j = 0, k = 0, index1 = 0, index2 = 0;
    int p = 0;
    for (; p < size; p++) {
        int index = indices[(p) * 32];

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

        if (term(j, index1).limitSum(term(j, index2), j != 2) && term(k, index2).limitSub(term(k, index1))) {
            if (k != 2 && !term(k, index2).positiveFirstNonZeroSub(term(k, index1))) {
                int tmp = index1;
                index1 = index2;
                index2 = tmp;
            }
            break;
        }

        if (term(k, index1).limitSum(term(k, index2), k != 2) && term(j, index2).limitSub(term(j, index1))) {
            if (j != 2 && !term(j, index2).positiveFirstNonZeroSub(term(j, index1))) {
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

bool CompactScheme::tryPlus(LOCAL RandomState &state) LOCAL_METHOD {
    if (m >= MAX_RANK || m >= n[0] * n[1] * n[2])
        return false;

    bool candidate = false;
    for (int a = 0; a < m && !candidate; a++)
        for (int b = a + 1; b < m && !candidate; b++)
            candidate = term(0, a) != term(0, b) && term(1, a) != term(1, b) && term(2, a) != term(2, b);
    if (!candidate) return false;

    int index1 = randomWord(&state) % m;
    int index2 = randomWord(&state) % m;

    while (index1 == index2 || term(0, index1) == term(0, index2) || term(1, index1) == term(1, index2) || term(2, index1) == term(2, index2)) {
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

bool CompactScheme::trySplit(LOCAL RandomState &state) LOCAL_METHOD {
    if (m >= MAX_RANK || m >= n[0] * n[1] * n[2])
        return false;

    int index = randomWord(&state) % m;

    int permutation[3];
    randomPermutation(permutation, 3, state);
    int i = permutation[0];
    int j = permutation[1];
    int k = permutation[2];

    CompactAddition a(nn[i]);
    a.random(state);

    if (!term(i, index).limitSub(a, i != 2))
        return false;

    split(i, j, k, index, a);
    return true;
}

bool CompactScheme::trySplitExisted(LOCAL RandomState &state) LOCAL_METHOD {
    if (m >= MAX_RANK || m >= n[0] * n[1] * n[2])
        return false;

    bool candidate = false;
    for (int p = 0; p < 3 && !candidate; p++)
        for (int a = 0; a < m && !candidate; a++)
            for (int b = a + 1; b < m && !candidate; b++)
                candidate = term(p, a) != term(p, b);
    if (!candidate) return false;

    int index1, index2;
    int i;

    do {
        index1 = randomWord(&state) % m;
        index2 = randomWord(&state) % m;
        i = randomWord(&state) % 3;
    } while (index1 == index2 || term(i, index1) == term(i, index2));

    if (!term(i, index1).limitSub(term(i, index2), i != 2))
        return false;

    int j = (i + 1) % 3;
    int k = (i + 2) % 3;

    split(i, j, k, index1, term(i, index2));
    return true;
}

bool CompactScheme::tryExpand(int count, LOCAL RandomState &state) LOCAL_METHOD {
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

bool CompactScheme::tryReduce() LOCAL_METHOD {
    for (size_t i = 0; i < flips[0].size; i++) {
        int index1 = flips[0].index1(i);
        int index2 = flips[0].index2(i);

        if (term(1, index1) == term(1, index2) && term(2, index1).limitSum(term(2, index2), false)) {
            reduceAdd(2, index1, index2);
            return true;
        }

        int cmp2 = term(2, index1).compare(term(2, index2));
        if (cmp2 == 1 && term(1, index1).limitSum(term(1, index2), true)) {
            reduceAdd(1, index1, index2);
            return true;
        }

        if (cmp2 == -1 && term(1, index1).limitSub(term(1, index2), true)) {
            reduceSub(1, index1, index2);
            return true;
        }
    }

    for (size_t i = 0; i < flips[1].size; i++) {
        int index1 = flips[1].index1(i);
        int index2 = flips[1].index2(i);
        int cmp2 = term(2, index1).compare(term(2, index2));

        if (cmp2 == 1 && term(0, index1).limitSum(term(0, index2), true)) {
            reduceAdd(0, index1, index2);
            return true;
        }

        if (cmp2 == -1 && term(0, index1).limitSub(term(0, index2), true)) {
            reduceSub(0, index1, index2);
            return true;
        }
    }

    return false;
}

#endif
