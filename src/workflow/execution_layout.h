#pragma once
#include "execution.h"
#include "../metal/controlled.h"
#include "../metal/controlled_capture.h"

// Include after the native Metal arithmetic declarations. Keep actual compiled
// type sizes at the caller boundary; execution.cpp does not duplicate arithmetic.
inline fgm::ExecutionLayout nativeExecutionLayout() {
    using U = AdditionsReducer<350,250,32,2016>;
    using W = AdditionsReducer<64,500,175,61075>;
    return {sizeof(SchemeInteger), sizeof(SchemeZ2),
            2*sizeof(U) + sizeof(W) + sizeof(SchemeInteger), sizeof(RandomState), sizeof(ControlledState)+sizeof(int)};
}
