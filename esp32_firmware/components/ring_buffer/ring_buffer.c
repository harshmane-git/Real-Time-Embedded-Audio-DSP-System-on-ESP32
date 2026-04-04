#include "ring_buffer.h"
#include <string.h>

void rb_init(ring_buffer_t *rb)
{
    rb->write_slot = 0;
    rb->read_slot  = 0;

    for (int i = 0; i < RB_NUM_SLOTS; i++) {
        rb->slots[i].ready = false;
        memset(rb->slots[i].samples, 0,
               sizeof(rb->slots[i].samples));
    }
}

// PRODUCER (mic_task)
bool rb_write_block(ring_buffer_t *rb, float *data)
{
    int ws = rb->write_slot;

    // Overflow → drop block (real-time safe)
    if (rb->slots[ws].ready == true) {
        return false;
    }

    memcpy(rb->slots[ws].samples, data,
           RB_SAMPLES_PER_SLOT * sizeof(float));

    rb->slots[ws].ready = true;  // mark full AFTER copy

    rb->write_slot = (rb->write_slot + 1) % RB_NUM_SLOTS;

    return true;
}

// CONSUMER (scheduler_task)
bool rb_read_block(ring_buffer_t *rb, float *dest)
{
    int rs = rb->read_slot;

    // Underrun
    if (rb->slots[rs].ready == false)
        return false;

    memcpy(dest, rb->slots[rs].samples,
           RB_SAMPLES_PER_SLOT * sizeof(float));

    rb->slots[rs].ready = false; // mark empty AFTER copy

    rb->read_slot = (rb->read_slot + 1) % RB_NUM_SLOTS;

    return true;
}

// Utility for scheduler
int rb_slots_filled(ring_buffer_t *rb)
{
    return (rb->write_slot - rb->read_slot + RB_NUM_SLOTS) % RB_NUM_SLOTS;
}