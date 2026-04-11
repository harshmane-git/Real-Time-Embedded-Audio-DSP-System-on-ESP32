#ifndef PIPELINE_CONTROL_H
#define PIPELINE_CONTROL_H

#include <stdbool.h>

typedef struct {
    bool mute;      // true = entire output muted (silence)
    bool bypass;    // true = bypass all filters (raw input passes through)
} pipeline_state_t;

extern pipeline_state_t pipeline;

void pipeline_init(void);
void pipeline_set_mute(bool enable);
void pipeline_set_bypass(bool enable);

#endif