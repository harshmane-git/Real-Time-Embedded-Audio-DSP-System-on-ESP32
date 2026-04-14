#include "scheduler.h"
#include "amp.h"
#include "gain.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

void scheduler_task(void *arg)
{   
    ring_buffer_t *rb = (ring_buffer_t *)arg;

    float input_block[RB_SAMPLES_PER_SLOT];

    while (1)
    {
        int distance = (rb->write_slot - rb->read_slot + RB_NUM_SLOTS) % RB_NUM_SLOTS;

        if (distance >= READ_TRIGGER)
        {
            if (rb_read_block(rb, input_block))
            {
                apply_gain(input_block, RB_SAMPLES_PER_SLOT, 5.0f);

                amp_write_block(input_block, RB_SAMPLES_PER_SLOT);
            }
        }
        else
        {
            taskYIELD();
        }
    }
}