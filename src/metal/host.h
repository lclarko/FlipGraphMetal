#pragma once
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <stdexcept>
#include "core.h"
#include "addition.h"
#include "flip_set.h"
#include "scheme_integer.h"
#include "scheme_z2.h"
#include "pairs_counter.h"
#include "additions_reducer.h"
#include "runtime.h"
#ifdef METAL_F2
#define Scheme SchemeZ2
const std::string ring = "Z2";
#else
#define SCHEME_INTEGER
#define Scheme SchemeInteger
const std::string ring = "ZT";
#endif

const int MAX_U_EXPRESSIONS = MAX_RANK;
const int MAX_V_EXPRESSIONS = MAX_RANK;
const int MAX_W_EXPRESSIONS = MAX_MATRIX_ELEMENTS;

const int MAX_U_REAL_VARIABLES = MAX_MATRIX_ELEMENTS / 2;
const int MAX_V_REAL_VARIABLES = MAX_MATRIX_ELEMENTS / 2;
const int MAX_W_REAL_VARIABLES = MAX_RANK / 2;

const int MAX_U_SUBEXPRESSIONS = MAX_MATRIX_ELEMENTS * (MAX_MATRIX_ELEMENTS - 1) / 2;
const int MAX_V_SUBEXPRESSIONS = MAX_MATRIX_ELEMENTS * (MAX_MATRIX_ELEMENTS - 1) / 2;
const int MAX_W_SUBEXPRESSIONS = MAX_RANK * (MAX_RANK - 1) / 2;

const int MAX_U_FRESH_VARIABLES = 250;
const int MAX_V_FRESH_VARIABLES = 250;
const int MAX_W_FRESH_VARIABLES = 500;

extern int metalRounds;
std::string getKey(int n1, int n2, int n3, bool sorted = true, bool withZeros = false);
std::string getKey(const Scheme &scheme, bool sorted = true, bool withZeros = false);
std::string prettyTime(double elapsed);
std::string prettyInt(int value);

struct FlipGraphProbabilities {
    float expand;
    float reduce;
    float sandwiching;
    float basis;
    float resize;
};

class FlipGraph {
    int n1;
    int n2;
    int n3;

    int schemesCount;
    int maxIterations;
    int plusIterations;
    std::string path;

    FlipGraphProbabilities probabilities;
    int seed;

    int blockSize;
    int numBlocks;

    Scheme *schemes;
    Scheme *schemesBest;
    int *bestRanks;
    int *flips;
    RandomState *states;

    std::unordered_map<std::string, int> n2bestRank;
    std::unordered_map<std::string, int> n2knownRanks;
public:
    FlipGraph(int n1, int n2, int n3, int schemesCount, int blockSize, int maxIterations, int plusIterations, const std::string &path, const FlipGraphProbabilities &probabilities, int seed);

    bool initializeFromFile(std::istream &f);
    void initializeNaive();
    void run(int logPeriod);

    ~FlipGraph();
private:
    void initialize();
    void randomWalk();
    void resize();
    void updateRanks(int iteration, bool save);
    void report(std::chrono::high_resolution_clock::time_point startTime, int iteration, const std::vector<double> &elapsedTimes, int logPeriod, int count = 3);

    bool compareKeys(const Scheme &s1, const Scheme &s2) const;

    std::string prettyFlips(int flips) const;
    std::string getSavePath(const Scheme &scheme, int iteration, int runId) const;
    std::unordered_map<std::string, std::vector<int>> getSortedIndices(int count) const;
};

class ComplexityMinimizer {
    int initialCount;
    int schemesCount;
    std::string path;

    int maxIterations;
    int seed;
    int topCount;
    int blockSize;
    int numBlocks;

    Scheme *schemes;
    Scheme *schemesBest;
    int *bestComplexities;
    int bestComplexity;
    RandomState *states;

    std::vector<int> indices;
public:
    ComplexityMinimizer(int schemesCount, int blockSize, int maxIterations, const std::string &path, int seed, int topCount = 10);

    bool read(std::istream &f);
    void minimize(int targetComplexity, int maxNoImprovements);

    ~ComplexityMinimizer();
private:
    void initialize();
    void minimizeIteration();
    bool updateBest(int iteration);
    void report(std::chrono::high_resolution_clock::time_point startTime, int iteration, const std::vector<double> &elapsedTimes);

    std::string getSavePath(const Scheme &scheme, int iteration, int runId) const;
};

class SchemeAdditionsReducer {
    int n1, n2, n3;
    int m;
    int count;
    int schemesCount;
    int maxFlips;
    int seed;
    int blockSize;
    int numBlocks;
    std::string outputPath;

    int topCount;
    int reducedAdditions;
    int reducedFreshVars;
    int bestAdditions[3];
    int bestFreshVars[3];
    std::vector<int> indices[3];

    AdditionsReducer<MAX_U_EXPRESSIONS, MAX_U_FRESH_VARIABLES, MAX_U_REAL_VARIABLES, MAX_U_SUBEXPRESSIONS> *reducersU;
    AdditionsReducer<MAX_V_EXPRESSIONS, MAX_V_FRESH_VARIABLES, MAX_V_REAL_VARIABLES, MAX_V_SUBEXPRESSIONS> *reducersV;
    AdditionsReducer<MAX_W_EXPRESSIONS, MAX_W_FRESH_VARIABLES, MAX_W_REAL_VARIABLES, MAX_W_SUBEXPRESSIONS> *reducersW;
    SchemeInteger *schemes;
    RandomState *states;

    void initialize();
    void reduceIteration(int iteration);
    bool updateBest(int startAdditions);
    bool updateBestIndependent();
    bool updateBestTogether();
    void report(std::chrono::high_resolution_clock::time_point startTime, int iteration, const std::vector<double> &elapsedTimes);
    void save() const;

    std::string getSavePath() const;
    std::string getDimensions() const;
public:
    SchemeAdditionsReducer(int count, int schemesCount, int maxFlips, int seed, int blockSize, const std::string &outputPath, int topCount = 10);

    bool read(std::ifstream &f);
    void reduce(int maxNoImprovements, int startAdditions);

    ~SchemeAdditionsReducer();
};
