#pragma once

struct Addition {
    int n;
    T values;
    T signs;
    bool valid;

    Addition() LOCAL_METHOD;
    Addition(int n) LOCAL_METHOD;
    Addition(int n, int index) LOCAL_METHOD;
    Addition(int n, LOCAL int *values) LOCAL_METHOD;

    void copyTo(LOCAL Addition &target) const LOCAL_METHOD;
    void set(int index, int value) LOCAL_METHOD;
    void inverse() LOCAL_METHOD;
    void random(LOCAL RandomState &state) LOCAL_METHOD;

    int compare(LOCAL const Addition &addition) const LOCAL_METHOD;

    bool operator==(LOCAL const Addition &addition) const LOCAL_METHOD;
    bool operator!=(LOCAL const Addition &addition) const LOCAL_METHOD;
    int operator[](int index) const LOCAL_METHOD;
#ifdef __METAL_VERSION__
    int operator[](int index) const device {
        int value = int((values >> index) & 1);
        return ((signs >> index) & 1) ? -value : value;
    }
#endif

    Addition operator+(LOCAL const Addition &addition) const LOCAL_METHOD;
    Addition operator-(LOCAL const Addition &addition) const LOCAL_METHOD;
    Addition operator-() const LOCAL_METHOD;

    LOCAL Addition & operator+=(LOCAL const Addition &addition) LOCAL_METHOD;
    LOCAL Addition & operator-=(LOCAL const Addition &addition) LOCAL_METHOD;

    bool limit(bool firstPositiveNonZero) const LOCAL_METHOD;
    bool limitSum(LOCAL const Addition &addition, bool firstPositiveNonZero) const LOCAL_METHOD;
    bool limitSub(LOCAL const Addition &addition, bool firstPositiveNonZero = false) const LOCAL_METHOD;

    bool positiveFirstNonZero() const LOCAL_METHOD;
    bool positiveFirstNonZeroSub(LOCAL const Addition &addition) const LOCAL_METHOD;

    int nonZeroCount() const LOCAL_METHOD;

    operator bool() const LOCAL_METHOD;

#ifndef __METAL_VERSION__
    friend std::ostream& operator<<(std::ostream &os, LOCAL const Addition &addition);
#endif
};

Addition::Addition() LOCAL_METHOD {
    n = 0;
    values = 0;
    signs = 0;
    valid = true;
}

Addition::Addition(int n) LOCAL_METHOD {
    this->n = n;
    this->values = 0;
    this->signs = 0;
    this->valid = true;
}

Addition::Addition(int n, int index) LOCAL_METHOD {
    this->n = n;
    this->values = T(1) << index;
    this->signs = 0;
    this->valid = true;
}

Addition::Addition(int n, LOCAL int *values) LOCAL_METHOD {
    this->n = n;
    this->values = 0;
    this->valid = true;
    this->signs = 0;

    for (int i = 0; i < n; i++)
        set(i, values[i]);
}

void Addition::copyTo(LOCAL Addition &target) const LOCAL_METHOD {
    target.n = n;
    target.values = values;
    target.signs = signs;
    target.valid = valid;
}

void Addition::set(int index, int value) LOCAL_METHOD {
    T mask = T(1) << index;

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

void Addition::inverse() LOCAL_METHOD {
    signs = (~signs) & values;
}

void Addition::random(LOCAL RandomState &state) LOCAL_METHOD {
    values = 0;
    signs = 0;

    for (int i = 0; i < n; i += 16) {
        uint16_t signBits = randomWord(&state) & 0xFFFF;
        uint16_t valuesBits = randomWord(&state) & 0xFFFF;

        values |= T(valuesBits) << i;
        signs |= T(signBits) << i;
    }

    if (n < 64) values &= (T(1) << n) - 1;
    signs &= values;
    valid = true;
}

int Addition::compare(LOCAL const Addition &addition) const LOCAL_METHOD {
    if (values != addition.values)
        return 0;

    if (signs == addition.signs)
        return 1;

    if (signs == ((~addition.signs) & addition.values))
        return -1;

    return 0;
}

bool Addition::operator==(LOCAL const Addition &addition) const LOCAL_METHOD {
    return values == addition.values && signs == addition.signs;
}

bool Addition::operator!=(LOCAL const Addition &addition) const LOCAL_METHOD {
    return values != addition.values || signs != addition.signs;
}

int Addition::operator[](int index) const LOCAL_METHOD {
    int value = int((values >> index) & 1);

    if ((signs >> index) & 1)
        value = -value;

    return value;
}

Addition Addition::operator+(LOCAL const Addition &addition) const LOCAL_METHOD {
    Addition result(n);

    result.values = values ^ addition.values;
    result.signs = ((signs & values) | (addition.signs & addition.values)) & result.values;
    result.valid = !(values & addition.values & ~(signs ^ addition.signs));
    return result;
}

Addition Addition::operator-(LOCAL const Addition &addition) const LOCAL_METHOD {
    Addition result(n);

    result.values = values ^ addition.values;
    result.signs = ((signs & values) | (~addition.signs & addition.values)) & result.values;
    result.valid = !(values & addition.values & (signs ^ addition.signs));
    return result;
}

Addition Addition::operator-() const LOCAL_METHOD {
    Addition result(n);
    result.values = values;
    result.signs = (~signs) & values;
    result.valid = valid;
    return result;
}

LOCAL Addition & Addition::operator+=(LOCAL const Addition &addition) LOCAL_METHOD {
    T sv1 = signs & values;
    T sv2 = addition.signs & addition.values;

    valid = !((values & addition.values & ~(signs ^ addition.signs)));
    values ^= addition.values;
    signs = (sv1 | sv2) & values;
    return *this;
}

LOCAL Addition & Addition::operator-=(LOCAL const Addition &addition) LOCAL_METHOD {
    T sv1 = signs & values;
    T sv2 = ~addition.signs & addition.values;

    valid = !(values & addition.values & (signs ^ addition.signs));
    values ^= addition.values;
    signs = (sv1 | sv2) & values;
    return *this;
}

Addition::operator bool() const LOCAL_METHOD {
    return values != 0;
}

bool Addition::limit(bool firstPositiveNonZero) const LOCAL_METHOD {
    if (!valid)
        return false;

    if (firstPositiveNonZero)
        return values == 0 || (values & ~(values & (values - 1)) & ~signs);

    return true;
}

bool Addition::limitSum(LOCAL const Addition &addition, bool firstPositiveNonZero) const LOCAL_METHOD {
    bool invalid = values & addition.values & ~(signs ^ addition.signs);

    if (invalid)
        return false;

    if (firstPositiveNonZero) {
        T sumValues = values ^ addition.values;
        T sumSigns = ((signs & values) | (addition.signs & addition.values)) & sumValues;

        return sumValues == 0 || (sumValues & ~(sumValues & (sumValues - 1)) & ~sumSigns);
    }

    return true;
}

bool Addition::limitSub(LOCAL const Addition &addition, bool firstPositiveNonZero) const LOCAL_METHOD {
    bool invalid = (values & addition.values & (signs ^ addition.signs));

    if (invalid)
        return false;

    if (firstPositiveNonZero) {
        T subValues = values ^ addition.values;
        T subSigns = ((signs & values) | (~addition.signs & addition.values)) & subValues;

        return subValues == 0 || (subValues & ~(subValues & (subValues - 1)) & ~subSigns);
    }

    return true;
}

bool Addition::positiveFirstNonZeroSub(LOCAL const Addition &addition) const LOCAL_METHOD {
    T subValues = values ^ addition.values;
    T subSigns = ((signs & values) | (~addition.signs & addition.values)) & subValues;

    return subValues == 0 || (subValues & ~(subValues & (subValues - 1)) & ~subSigns);
}

bool Addition::positiveFirstNonZero() const LOCAL_METHOD {
    return values == 0 || (values & ~(values & (values - 1)) & ~signs);
}

int Addition::nonZeroCount() const LOCAL_METHOD {
#ifdef __METAL_VERSION__
    return popcount(values);
#else
    return __builtin_popcountll(values);
#endif
}

#ifndef __METAL_VERSION__
std::ostream& operator<<(std::ostream &os, LOCAL const Addition &addition) {
    for (int i = 0; i < addition.n; i++) {
        if (i > 0)
            os << ", ";

        os << addition[i];
    }

    return os;
}
#endif
