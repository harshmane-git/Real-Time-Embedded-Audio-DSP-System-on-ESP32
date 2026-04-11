#include "pipeline_control.h"

pipeline_state_t pipeline = {false, false};   // default: no mute, no bypass

void pipeline_init(void) {
    pipeline.mute = false;
    pipeline.bypass = false;
}

void pipeline_set_mute(bool enable) {
    pipeline.mute = enable;
}

void pipeline_set_bypass(bool enable) {
    pipeline.bypass = enable;
}