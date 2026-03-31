#include "ring_buffer.h"
#include <string.h>
#include <stdio.h>

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

// PRODUCER (I2S DMA / ISR)
bool I2S_write(ring_buffer_t *rb, int32_t *data)
{
    int ws = rb->write_slot;

    // overflow
    if (rb->slots[ws].ready == true) {
        // drop data (real-time safe choice)
        printf("[RB] Overflow!\n");
        return false;
    }

    memcpy(rb->slots[ws].samples, data,
           RB_SAMPLES_PER_SLOT * sizeof(int32_t));

    rb->slots[ws].ready = true;  // LAST

    rb->write_slot = (rb->write_slot + 1) % RB_NUM_SLOTS;

    return true;
}

// CONSUMER (DSP)
bool DMA_read(ring_buffer_t *rb, int32_t *dest)
{
    int rs = rb->read_slot;

    if (rb->slots[rs].ready == false)
        return false; // underrun

    memcpy(dest, rb->slots[rs].samples,
           RB_SAMPLES_PER_SLOT * sizeof(int32_t));

    rb->slots[rs].ready = false; // LAST

    rb->read_slot = (rb->read_slot + 1) % RB_NUM_SLOTS;

    return true;
}
