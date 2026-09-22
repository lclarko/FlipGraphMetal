#pragma once

struct Pair {
    int i;
    int j;
    int count;

    bool intersects(GLOBAL const Pair &pair) const REDUCER_METHOD {
        return i == pair.i || i == abs(pair.j) || abs(j) == pair.i || abs(j) == abs(pair.j);
    }
};

template <size_t maxPairs>
class PairsCounter {
    enum { capacity = maxPairs * 2 + 1 };

    Pair pairs[maxPairs];
    int size;
    bool overflow;
    int topSize;
    int maxCount;
    int hashTable[capacity];

    void canonizePair(LOCAL int &i, LOCAL int &j) const REDUCER_METHOD;
    unsigned int getHash(int i, int j) const REDUCER_METHOD;
    int findOrCreateSlot(int i, int j) REDUCER_METHOD;
public:
    bool valid() const REDUCER_METHOD { return !overflow; }
    PairsCounter() REDUCER_METHOD;

    void insert(int i, int j) REDUCER_METHOD;
    void sort() REDUCER_METHOD;
    void clear() REDUCER_METHOD;
    void copyFrom(GLOBAL const PairsCounter<maxPairs> &counter) REDUCER_METHOD;

    Pair getGreedy() const REDUCER_METHOD;
    Pair getGreedyAlternative(LOCAL RandomState &state) const REDUCER_METHOD;
    Pair getGreedyRandom(LOCAL RandomState &state, float scale) const REDUCER_METHOD;
    Pair getGreedyIntersections(LOCAL RandomState &state, float scale) const REDUCER_METHOD;
    Pair getWeightedRandom(LOCAL RandomState &state) const REDUCER_METHOD;
    Pair getRandom(LOCAL RandomState &state) const REDUCER_METHOD;

    int count() const REDUCER_METHOD;
    operator bool() const REDUCER_METHOD;
};

template <size_t maxPairs>
PairsCounter<maxPairs>::PairsCounter() REDUCER_METHOD {
    size = 0;
    overflow = false;
    topSize = 0;
    maxCount = 0;

    for (int i = 0; i < capacity; i++)
        hashTable[i] = -1;
}

template <size_t maxPairs>
void PairsCounter<maxPairs>::insert(int i, int j) REDUCER_METHOD {
    int index = findOrCreateSlot(i, j);
    if (index == -1)
        return;

    pairs[index].count++;

    if (pairs[index].count > maxCount)
        maxCount = pairs[index].count;
}

template <size_t maxPairs>
void PairsCounter<maxPairs>::sort() REDUCER_METHOD {
    if (maxCount < 2) {
        topSize = size;
        return;
    }

    topSize = 0;

    for (int i = 0; i < size; i++) {
        if (pairs[i].count != maxCount)
            continue;

        if (i != topSize) {
            Pair tmp = pairs[i];
            pairs[i] = pairs[topSize];
            pairs[topSize] = tmp;
        }

        topSize++;
    }

    int j = topSize;

    for (int i = topSize; i < size; i++)
        if (pairs[i].count > 1)
            pairs[j++] = pairs[i];

    size = j;
}

template <size_t maxPairs>
void PairsCounter<maxPairs>::clear() REDUCER_METHOD {
    size = 0;
    overflow = false;
    topSize = 0;
    maxCount = 0;

    for (int i = 0; i < capacity; i++)
        hashTable[i] = -1;
}

template <size_t maxPairs>
void PairsCounter<maxPairs>::copyFrom(GLOBAL const PairsCounter<maxPairs> &counter) REDUCER_METHOD {
    overflow = counter.overflow;
    size = counter.size;
    topSize = counter.topSize;
    maxCount = counter.maxCount;

    for (int index = 0; index < size; index++) {
        pairs[index].i = counter.pairs[index].i;
        pairs[index].j = counter.pairs[index].j;
        pairs[index].count = counter.pairs[index].count;
    }

    for (int i = 0; i < capacity; i++)
        hashTable[i] = counter.hashTable[i];
}

template <size_t maxPairs>
Pair PairsCounter<maxPairs>::getGreedy() const REDUCER_METHOD {
    return pairs[0];
}

template <size_t maxPairs>
Pair PairsCounter<maxPairs>::getGreedyAlternative(LOCAL RandomState &state) const REDUCER_METHOD {
    return pairs[randomWord(&state) % topSize];
}

template <size_t maxPairs>
Pair PairsCounter<maxPairs>::getGreedyRandom(LOCAL RandomState &state, float scale) const REDUCER_METHOD {
    if (randomUniform(&state) < scale)
        return pairs[randomWord(&state) % size];

    return pairs[randomWord(&state) % topSize];
}

template <size_t maxPairs>
Pair PairsCounter<maxPairs>::getGreedyIntersections(LOCAL RandomState &state, float scale) const REDUCER_METHOD {
    int imax = 0;
    float maxScore = 0;

    for (int i = 0; i < size; i++) {
        float intersections = 0;

        for (int j = 0; j < size; j++) {
            if (i == j)
                continue;

            if (!pairs[i].intersects(pairs[j])) {
                intersections += pairs[j].count - 1;
            }
            else {
                intersections += randomWord(&state) % 2 ? 0 : 0.69f * (pairs[j].count);
            }
        }

        float score = pairs[i].count - 1 + scale * intersections;

        if (score > maxScore) {
            maxScore = score;
            imax = i;
        }
    }

    return pairs[imax];
}

template <size_t maxPairs>
Pair PairsCounter<maxPairs>::getWeightedRandom(LOCAL RandomState &state) const REDUCER_METHOD {
    float total = 0;

    for (int i = 0; i < size; i++)
        total += pairs[i].count;

    float p = randomUniform(&state) * total;
    float sum = 0;

    for (int i = 0; i < size; i++) {
        sum += pairs[i].count;

        if (p <= sum)
            return pairs[i];
    }

    return pairs[size - 1];
}

template <size_t maxPairs>
Pair PairsCounter<maxPairs>::getRandom(LOCAL RandomState &state) const REDUCER_METHOD {
    return pairs[randomWord(&state) % size];
}

template <size_t maxPairs>
int PairsCounter<maxPairs>::count() const REDUCER_METHOD {
    return size;
}

template <size_t maxPairs>
PairsCounter<maxPairs>::operator bool() const REDUCER_METHOD {
    return size > 0 && maxCount > 1;
}

template <size_t maxPairs>
void PairsCounter<maxPairs>::canonizePair(LOCAL int &i, LOCAL int &j) const REDUCER_METHOD {
    if (abs(i) > abs(j)) {
        int tmp = i;
        i = j;
        j = tmp;
    }

    if (i < 0) {
        i = -i;
        j = -j;
    }
}

template <size_t maxPairs>
unsigned int PairsCounter<maxPairs>::getHash(int i, int j) const REDUCER_METHOD {
    unsigned int hash = (static_cast<unsigned int>(i) * 2654435761u) ^ (static_cast<unsigned int>(j) * 2246822519u);
    return hash % capacity;
}

template <size_t maxPairs>
int PairsCounter<maxPairs>::findOrCreateSlot(int i, int j) REDUCER_METHOD {
    canonizePair(i, j);
    unsigned int hash = getHash(i, j);

    for (int attempts = 0; attempts < capacity; attempts++) {
        int index = hashTable[hash];

        if (index == -1) {
            if (size >= maxPairs) {
                overflow = true;
                return -1;
            }

            pairs[size].i = i;
            pairs[size].j = j;
            pairs[size].count = 0;
            hashTable[hash] = size;
            return size++;
        }

        if (pairs[index].i == i && pairs[index].j == j)
            return index;

        hash = (hash + 1) % capacity;
    }

    return -1;
}
