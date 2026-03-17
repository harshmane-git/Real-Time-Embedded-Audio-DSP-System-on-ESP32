#ifndef GAIN_H
#define GAIN_H

#include <stdint.h>
#include <stdbool.h>

#define GAIN_PRESET_COUNT 3

typedef struct
{
    float current_gain;
    float preset_table[GAIN_PRESET_COUNT];
    uint8_t preset_index;
    volatile bool update_requested;
} gain_t;

void gain_init(gain_t *g);
void gain_process_block(gain_t *g, float *buffer, uint16_t block_size);
void gain_request_next_preset(gain_t *g);
void gain_apply_update(gain_t *g);

#endif