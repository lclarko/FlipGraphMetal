#pragma once

struct FlipSet {
    size_t size;
    uint32_t pairs[MAX_PAIRS];

    FlipSet() LOCAL_METHOD;

    void add(uint32_t index1, uint32_t index2) LOCAL_METHOD;
    void remove(uint32_t index1, uint32_t index2) LOCAL_METHOD;
    void remove(uint32_t index) LOCAL_METHOD;
    void clear() LOCAL_METHOD;

    uint32_t index1(size_t i) const LOCAL_METHOD;
    uint32_t index2(size_t i) const LOCAL_METHOD;
};

FlipSet::FlipSet() LOCAL_METHOD {
    size = 0;
}

void FlipSet::add(uint32_t index1, uint32_t index2) LOCAL_METHOD {
    if (size >= MAX_PAIRS)
        return;

    uint32_t pair = (index1 << 16) | index2;
    pairs[size++] = pair;
}

void FlipSet::remove(uint32_t index1, uint32_t index2) LOCAL_METHOD {
    uint32_t pair1 = (index1 << 16) | index2;
    uint32_t pair2 = (index2 << 16) | index1;

    for (size_t i = 0; i < size; i++) {
        if (pairs[i] == pair1 || pairs[i] == pair2) {
            pairs[i] = pairs[--size];
            return;
        }
    }
}

void FlipSet::remove(uint32_t index) LOCAL_METHOD {
    for (size_t i = 0; i < size; i++)
        if (index1(i) == index || index2(i) == index)
            pairs[i--] = pairs[--size];
}

void FlipSet::clear() LOCAL_METHOD {
    size = 0;
}

uint32_t FlipSet::index1(size_t i) const LOCAL_METHOD {
    return pairs[i] >> 16;
}

uint32_t FlipSet::index2(size_t i) const LOCAL_METHOD {
    return pairs[i] & 0xFFFF;
}
