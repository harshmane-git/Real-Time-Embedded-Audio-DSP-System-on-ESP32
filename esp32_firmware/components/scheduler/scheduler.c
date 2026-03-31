#include "scheduler.h"
#include "amp.h"
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

extern ring_buffer_t rb;

// DSP (dummy)
static void dsp_process(int32_t *in, int32_t *out)
{
    for (int i = 0; i < RB_SAMPLES_PER_SLOT; i++)
        out[i] = in[i];
}

void scheduler_task(void *arg)
{
    int32_t input_block[RB_SAMPLES_PER_SLOT];
    int32_t output_block[RB_SAMPLES_PER_SLOT];

    while (1)
    {
        int distance = (rb.write_slot - rb.read_slot + RB_NUM_SLOTS) % RB_NUM_SLOTS;

        if (distance >= READ_TRIGGER)
        {
            if (DMA_read(&rb, input_block))
            {
                dsp_process(input_block, output_block);

                // OUTPUT
                amp_write_block(output_block, RB_SAMPLES_PER_SLOT);
            }
            else
            {
                printf("[SCHEDULER] Underrun!\n");
            }
        }

        vTaskDelay(1);
    }
}