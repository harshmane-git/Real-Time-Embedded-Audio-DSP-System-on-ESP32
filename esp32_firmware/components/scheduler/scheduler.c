#include "scheduler.h"
#include <stdio.h>

// Replace later with your DSP pipeline
static void dsp_process(int32_t *in, int32_t *out)
{
    for (int i = 0; i < RB_SAMPLES_PER_SLOT; i++)
        out[i] = in[i];
}

void scheduler_task(void *arg)
{
    ring_buffer_t *rb = (ring_buffer_t *)arg;

    int32_t input_block[RB_SAMPLES_PER_SLOT];
    int32_t output_block[RB_SAMPLES_PER_SLOT];

    while (1)
    {
        // calculate distance (LOCK-FREE)
        int distance = (rb->write_slot - rb->read_slot + RB_NUM_SLOTS) % RB_NUM_SLOTS;

        if (distance >= READ_TRIGGER)
        {
            if (DMA_read(rb, input_block))
            {
                dsp_process(input_block, output_block);

                // TODO: send output to I2S DAC
            }
            else
            {
                printf("[SCHEDULER] Underrun!\n");
            }
        }

        // small yield (important for watchdog)
        vTaskDelay(1);
    }
}
