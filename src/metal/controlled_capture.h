#pragma once
struct ControlledCaptureMeta {
    uint64_t control;
    uint32_t operation;
    uint32_t reserved;
};
static_assert(sizeof(ControlledCaptureMeta) == 16, "controlled capture layout");
