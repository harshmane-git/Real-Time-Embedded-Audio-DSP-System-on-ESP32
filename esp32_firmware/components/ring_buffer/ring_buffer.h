#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include <stdint.h>
#include <stdbool.h>

#define RB_NUM_SLOTS        8
#define RB_SAMPLES_PER_SLOT 256

typedef struct {
    int32_t samples[RB_SAMPLES_PER_SLOT];
    volatile bool ready;
} rb_slot_t;

typedef struct {
    rb_slot_t slots[RB_NUM_SLOTS];
    volatile int write_slot;
    volatile int read_slot;
} ring_buffer_t;

// API
void rb_init(ring_buffer_t *rb);
bool I2S_write(ring_buffer_t *rb, int32_t *data);
bool DMA_read(ring_buffer_t *rb, int32_t *dest);

#endif
