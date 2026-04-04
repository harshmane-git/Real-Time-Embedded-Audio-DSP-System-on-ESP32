#include "scheduler.h"
#include "amp.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// DSP hook (replace later with gain/EQ/delay)
static void dsp_process(float *input, float *output)
{
    for (int i = 0; i < RB_SAMPLES_PER_SLOT; i++)
        output[i] = input[i];
}

void scheduler_task(void *arg)
{
    ring_buffer_t *rb = (ring_buffer_t *)arg;

    float input_block[RB_SAMPLES_PER_SLOT];
    float output_block[RB_SAMPLES_PER_SLOT];

    while (1)
    {
        // Calculate distance between write and read
        int distance = (rb->write_slot - rb->read_slot + RB_NUM_SLOTS) % RB_NUM_SLOTS;

        if (distance >= READ_TRIGGER)
        {
            if (rb_read_block(rb, input_block))
            {
                // DSP stage
                dsp_process(input_block, output_block);

                // Output
                amp_write_block(output_block, RB_SAMPLES_PER_SLOT);
            }
        }
        else
        {
            // Yield CPU until enough data arrives
            taskYIELD();
        }
    }
}