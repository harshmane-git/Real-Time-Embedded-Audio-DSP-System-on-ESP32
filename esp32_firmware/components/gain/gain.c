#include "gain.h"

void gain_init(gain_t *g)
{
    g->preset_table[0] = 1.0f;
    g->preset_table[1] = 1.5f;
    g->preset_table[2] = 2.0f;

    g->preset_index = 0;
    g->current_gain = g->preset_table[0];
    g->update_requested = false;
}

void gain_request_next_preset(gain_t *g)
{
    g->preset_index++;
    if (g->preset_index >= GAIN_PRESET_COUNT)
        g->preset_index = 0;

    g->update_requested = true;
}

void gain_apply_update(gain_t *g)
{
    if (g->update_requested)
    {
        g->current_gain = g->preset_table[g->preset_index];
        g->update_requested = false;
    }
}

void gain_process_block(gain_t *g, float *buffer, uint16_t block_size)
{
    for (uint16_t i = 0; i < block_size; i++)
    {
        buffer[i] *= g->current_gain;
    }
}