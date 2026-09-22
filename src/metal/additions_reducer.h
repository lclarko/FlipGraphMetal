#pragma once

enum SelectSubexpressionMode {
    GREEDY_MODE = 0,
    GREEDY_ALTERNATIVE_MODE = 1,
    GREEDY_RANDOM_MODE = 2,
    GREEDY_INTERSECTIONS_MODE = 3,
    WEIGHTED_RANDOM_MODE = 4,
    MIX_MODE = 5,
    RANDOM_MODE = 6,
};

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
class AdditionsReducer {
    int expressions[maxExpressionsCount][maxExpressionLength];
    int expressionSizes[maxExpressionsCount];
    int variables[maxVariablesCount][2];
    PairsCounter<maxSubexpressionsCount> subexpressions;

    bool valid;
    int expressionsCount;
    int realVariables;
    int freshVariables;
    int naiveAdditions;
    int mode;
    float scale;

    void updateSubexpressions() REDUCER_METHOD;
    void replaceSubexpression(LOCAL const Pair &subexpression) REDUCER_METHOD;
    void replaceExpression(GLOBAL int *expression, GLOBAL int &size, int index1, int index2, int varIndex) REDUCER_METHOD;

    int binarySearch(GLOBAL const int *expression, int size, int value, int start) const REDUCER_METHOD;
    Pair selectSubexpression(int mode, LOCAL RandomState &state) const REDUCER_METHOD;
public:
    bool isValid() const REDUCER_METHOD { return valid; }
    AdditionsReducer() REDUCER_METHOD;

    bool addExpression(LOCAL int *values, int count) REDUCER_METHOD;
    void reduce(LOCAL RandomState &state) REDUCER_METHOD;
    void copyFrom(GLOBAL const AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount> &reducer) REDUCER_METHOD;
    void partialInitialize(GLOBAL const AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount> &reducer, int copy) REDUCER_METHOD;
    void clear() REDUCER_METHOD;
    void setMode(int mode) REDUCER_METHOD;

    int getAdditions() const REDUCER_METHOD;
    int getMaxRealVariables() const REDUCER_METHOD;
    int getNaiveAdditions() const REDUCER_METHOD;
    int getFreshVars() const REDUCER_METHOD;
#ifndef __METAL_VERSION__
    std::string getMode() const;
#endif

#ifndef __METAL_VERSION__
    void write(std::ostream &os, const std::string &name, const std::string &indent) const;
#endif
};

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::AdditionsReducer() REDUCER_METHOD {
    valid = true;
    expressionsCount = 0;
    realVariables = 0;
    freshVariables = 0;
    naiveAdditions = 0;
    mode = GREEDY_MODE;
    scale = -1;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
bool AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::addExpression(LOCAL int *values, int count) REDUCER_METHOD {
    int size = 0;
    if (expressionsCount >= maxExpressionsCount) { valid = false; return false; }
    for (int i = 0; i < count; i++)
        if (values[i] && ++size > maxExpressionLength) { valid = false; return false; }
    size = 0;

    for (int i = 0; i < count; i++) {
        if (values[i] == 1)
            expressions[expressionsCount][size++] = i + 1;
        else if (values[i] == -1)
            expressions[expressionsCount][size++] = -(i + 1);
        else if (values[i] != 0) {
            valid = false;
            return false;
        }
    }

    if (count > realVariables)
        realVariables = count;

    expressionSizes[expressionsCount++] = size;
    naiveAdditions += size > 0 ? size - 1 : 0;
    return true;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::reduce(LOCAL RandomState &state) REDUCER_METHOD {
    if (scale < 0)
        scale = randomUniform(&state) / 2;

    while (freshVariables < maxVariablesCount) {
        updateSubexpressions();

        if (!valid || !subexpressions)
            break;

        int stepMode = mode == MIX_MODE ? randomWord(&state) % MIX_MODE : mode;
        Pair subexpression = selectSubexpression(stepMode, state);
        replaceSubexpression(subexpression);
    }
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::copyFrom(GLOBAL const AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount> &reducer) REDUCER_METHOD {
    valid = reducer.valid;
    naiveAdditions = reducer.naiveAdditions;
    expressionsCount = reducer.expressionsCount;
    realVariables = reducer.realVariables;
    freshVariables = reducer.freshVariables;
    mode = reducer.mode;
    scale = reducer.scale;

    for (int index = 0; index < expressionsCount; index++) {
        expressionSizes[index] = reducer.expressionSizes[index];

        for (int i = 0; i < expressionSizes[index]; i++)
            expressions[index][i] = reducer.expressions[index][i];
    }

    for (int i = 0; i < freshVariables; i++) {
        variables[i][0] = reducer.variables[i][0];
        variables[i][1] = reducer.variables[i][1];
    }

    subexpressions.copyFrom(reducer.subexpressions);
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::partialInitialize(GLOBAL const AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount> &reducer, int count) REDUCER_METHOD {
    for (int index = 0; index < count && index < reducer.freshVariables; index++) {
        int i = reducer.variables[index][0];
        int j = reducer.variables[index][1];

        replaceSubexpression({i, j, 0});
    }
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::clear() REDUCER_METHOD {
    valid = true;
    expressionsCount = 0;
    realVariables = 0;
    freshVariables = 0;
    naiveAdditions = 0;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::setMode(int mode) REDUCER_METHOD {
    this->mode = mode;
    scale = -1;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
int AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::getAdditions() const REDUCER_METHOD {
    int additions = freshVariables;

    for (int i = 0; i < expressionsCount; i++)
        additions += expressionSizes[i] > 0 ? expressionSizes[i] - 1 : 0;

    return additions;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
int AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::getMaxRealVariables() const REDUCER_METHOD {
    int variables = 0;

    for (int i = 0; i < expressionsCount; i++)
        if (expressionSizes[i] > variables)
            variables = expressionSizes[i];

    return variables;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
int AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::getNaiveAdditions() const REDUCER_METHOD {
    return naiveAdditions;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
int AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::getFreshVars() const REDUCER_METHOD {
    return freshVariables;
}

#ifndef __METAL_VERSION__
template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
std::string AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::getMode() const {
    if (mode == GREEDY_MODE)
        return "g";

    if (mode == GREEDY_ALTERNATIVE_MODE)
        return "ga";

    if (mode == GREEDY_RANDOM_MODE)
        return "gr" + std::to_string(int(scale * 100));

    if (mode == GREEDY_INTERSECTIONS_MODE)
        return "gi" + std::to_string(int(scale * 100));

    if (mode == WEIGHTED_RANDOM_MODE)
        return "wr";

    if (mode == RANDOM_MODE)
        return "rnd";

    return "mix";
}
#endif

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::updateSubexpressions() REDUCER_METHOD {
    subexpressions.clear();

    for (int index = 0; index < expressionsCount; index++)
        for (int i = 0; i < expressionSizes[index]; i++)
            for (int j = i + 1; j < expressionSizes[index]; j++)
                subexpressions.insert(expressions[index][i], expressions[index][j]);

    valid &= subexpressions.valid();
    subexpressions.sort();
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::replaceSubexpression(LOCAL const Pair &subexpression) REDUCER_METHOD {
    int varIndex = realVariables + freshVariables + 1;

    for (int index = 0; index < expressionsCount; index++) {
        if (expressionSizes[index] < 2)
            continue;

        int i = binarySearch(expressions[index], expressionSizes[index], subexpression.i, 0);
        if (i == -1)
            continue;

        int j = binarySearch(expressions[index], expressionSizes[index], subexpression.j, i + 1);
        if (j == -1)
            continue;

        if (expressions[index][i] == subexpression.i && expressions[index][j] == subexpression.j) {
            replaceExpression(expressions[index], expressionSizes[index], i, j, varIndex);
        }
        else if (expressions[index][i] == -subexpression.i && expressions[index][j] == -subexpression.j) {
            replaceExpression(expressions[index], expressionSizes[index], i, j, -varIndex);
        }
    }

    variables[freshVariables][0] = subexpression.i;
    variables[freshVariables][1] = subexpression.j;
    freshVariables++;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
int AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::binarySearch(GLOBAL const int *expression, int size, int value, int start) const REDUCER_METHOD {
    int low = start;
    int high = size - 1;
    int valueAbs = abs(value);

    while (low <= high) {
        int mid = low + (high - low) / 2;
        int midAbs = abs(expression[mid]);

        if (midAbs < valueAbs) {
            low = mid + 1;
        }
        else if (midAbs > valueAbs) {
            high = mid - 1;
        }
        else {
            return mid;
        }
    }

    return -1;
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
Pair AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::selectSubexpression(int mode, LOCAL RandomState &state) const REDUCER_METHOD {
    if (mode == GREEDY_MODE)
        return subexpressions.getGreedy();

    if (mode == GREEDY_ALTERNATIVE_MODE)
        return subexpressions.getGreedyAlternative(state);

    if (mode == GREEDY_RANDOM_MODE)
        return subexpressions.getGreedyRandom(state, scale);

    if (mode == GREEDY_INTERSECTIONS_MODE)
        return subexpressions.getGreedyIntersections(state, scale);

    if (mode == WEIGHTED_RANDOM_MODE)
        return subexpressions.getWeightedRandom(state);

    return subexpressions.getRandom(state);
}

template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::replaceExpression(GLOBAL int *expression, GLOBAL int &size, int index1, int index2, int varIndex) REDUCER_METHOD {
    int j = index1;

    for (int i = index1 + 1; i < size; i++)
        if (i != index2)
            expression[j++] = expression[i];

    size--;
    expression[j] = varIndex;
}

#ifndef __METAL_VERSION__
template <size_t maxExpressionsCount, size_t maxVariablesCount, size_t maxExpressionLength, size_t maxSubexpressionsCount>
void AdditionsReducer<maxExpressionsCount, maxVariablesCount, maxExpressionLength, maxSubexpressionsCount>::write(std::ostream &os, const std::string &name, const std::string &indent) const {
    os << indent << "\"" << name << "_fresh\": [" << std::endl;

    for (int i = 0; i < freshVariables; i++) {
        int index1 = abs(variables[i][0]) - 1;
        int value1 = variables[i][0] > 0 ? 1 : -1;

        int index2 = abs(variables[i][1]) - 1;
        int value2 = variables[i][1] > 0 ? 1 : -1;

        os << indent << indent << "[{\"index\": " << index1 << ", \"value\": " << value1 << "}, {\"index\": " << index2 << ", \"value\": " << value2 << "}]";

        if (i < freshVariables - 1)
            os << ",";

        os << std::endl;
    }

    os << indent << "]," << std::endl;
    os << indent << "\"" << name << "\": [" << std::endl;

    for (int i = 0; i < expressionsCount; i++) {
        os << indent << indent << "[";

        for (int j = 0; j < expressionSizes[i]; j++) {
            int index = abs(expressions[i][j]) - 1;
            int value = expressions[i][j] > 0 ? 1 : -1;

            os << "{\"index\": " << index << ", \"value\": " << value << "}";

            if (j < expressionSizes[i] - 1)
                os << ", ";
        }

        os << "]";

        if (i < expressionsCount - 1)
            os << ",";

        os << std::endl;
    }

    os << indent << "]";
}
#endif
