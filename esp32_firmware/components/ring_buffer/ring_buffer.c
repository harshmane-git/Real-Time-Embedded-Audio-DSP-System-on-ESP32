// ring_buffer.c
#include "ring_buffer.h"
#include <string.h>
#include <stdio.h>

#define READ_DELAY_SLOTS  2   // wait until write is 2 slots ahead before reading (slot 1 write → read starts at slot 3)

void rb_init(ring_buffer_t *rb)
{
    rb->write_slot   = 0;
    rb->read_slot    = 0;
    rb->slots_filled = 0;

    for (int i = 0; i < RB_NUM_SLOTS; i++) {
        memset(rb->slots[i].samples, 0, sizeof(rb->slots[i].samples));
        rb->slots[i].state = SLOT_EMPTY;
    }

    rb->mutex = xSemaphoreCreateMutex();
}

// Called by I2S ISR/task — writes 256 samples into the current write slot
bool I2S_write(ring_buffer_t *rb, int32_t *data, int len)
{
    if (len != RB_SAMPLES_PER_SLOT) return false;

    xSemaphoreTake(rb->mutex, portMAX_DELAY);

    // Overwrite protection — if next slot is still being read, we have overflow
    if (rb->slots_filled >= RB_NUM_SLOTS) {
        printf("[RB] WARNING: Overflow! Write dropped.\n");
        xSemaphoreGive(rb->mutex);
        return false;
    }

    int ws = rb->write_slot;
    rb->slots[ws].state = SLOT_WRITING;
    memcpy(rb->slots[ws].samples, data, len * sizeof(int32_t));
    rb->slots[ws].state = SLOT_READY;

    rb->write_slot  = (rb->write_slot + 1) % RB_NUM_SLOTS;
    rb->slots_filled++;

    xSemaphoreGive(rb->mutex);
    return true;
}

// Called by DMA read task — reads 256 samples, but only if write is ≥2 slots ahead
bool DMA_read(ring_buffer_t *rb, int32_t *dest)
{
    xSemaphoreTake(rb->mutex, portMAX_DELAY);

    // Core rule: read only starts when write is at least 2 slots ahead
    // i.e., write@slot3 before read@slot1 begins
    if (rb->slots_filled < READ_DELAY_SLOTS + 1) {
        // Not enough slots written yet — keep waiting
        xSemaphoreGive(rb->mutex);
        return false;
    }

    // Read never overtakes write — guaranteed by slots_filled check above
    int rs = rb->read_slot;

    if (rb->slots[rs].state != SLOT_READY) {
        xSemaphoreGive(rb->mutex);
        return false;
    }

    rb->slots[rs].state = SLOT_READING;
    memcpy(dest, rb->slots[rs].samples, RB_SAMPLES_PER_SLOT * sizeof(int32_t));
    rb->slots[rs].state = SLOT_EMPTY;

    rb->read_slot = (rb->read_slot + 1) % RB_NUM_SLOTS;
    rb->slots_filled--;

    xSemaphoreGive(rb->mutex);
    return true;
}

int rb_slots_available(ring_buffer_t *rb)
{
    return rb->slots_filled;
}

