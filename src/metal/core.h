#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#define LOCAL thread
#define LOCAL_METHOD thread
#define REDUCER_METHOD device
#define GLOBAL device
typedef ulong uint64_t;
typedef uint uint32_t;
typedef ushort uint16_t;
#else
#include <cstdint>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#define LOCAL
#define LOCAL_METHOD
#define REDUCER_METHOD
#define GLOBAL
#define constant
#endif

typedef uint64_t T;
constant constexpr int MAX_RANK = 350;
constant constexpr int MAX_MATRIX_ELEMENTS = 64;
constant constexpr int MAX_PAIRS = 500;
constant constexpr int MIN_PROJECT_N = 2;
constant constexpr int MAX_EXTENSION_N = 16;
constant constexpr int MAX_SANDWICHING_N = 16;
constant constexpr int MAX_SANDWICHING_ELEMENTS = 256;
struct RandomState { uint32_t value; };
inline uint32_t randomWord(LOCAL RandomState *state) {
    uint32_t x = state->value;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state->value = x;
    return x;
}
inline float randomUniform(LOCAL RandomState *state) {
    return float((randomWord(state) >> 8) + 1) * (1.0f / 16777216.0f);
}
inline int randint(int a, int b, LOCAL RandomState &state) {
    return a + randomWord(&state) % (b - a + 1);
}
inline void randomPermutation(LOCAL int *array, int n, LOCAL RandomState &state) {
    for (int i = 0; i < n; i++) array[i] = i;
    for (int i = n - 1; i > 0; i--) {
        int j = randomWord(&state) % (i + 1);
        int tmp = array[i]; array[i] = array[j]; array[j] = tmp;
    }
}

inline void randomMatrixZ2(int n, LOCAL int *matrix, LOCAL RandomState &state) {
    for (int i = 0; i < n; i++) {
        uint32_t bits = randomWord(&state);
        for (int j = 0; j < n; j++) matrix[i * n + j] = (bits >> j) & 1;
    }
}
