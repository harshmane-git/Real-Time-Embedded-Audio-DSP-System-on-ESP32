#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include <stdbool.h>
#include <stdint.h>

#define RB_NUM_SLOTS        8
#define RB_SAMPLES_PER_SLOT 256

typedef struct {
    float samples[RB_SAMPLES_PER_SLOT];   // float for DSP pipeline
    volatile bool ready;                 // true = full, false = empty
} rb_slot_t;

typedef struct {
    rb_slot_t slots[RB_NUM_SLOTS];
    volatile int write_slot;
    volatile int read_slot;
} ring_buffer_t;

// API
void rb_init(ring_buffer_t *rb);
bool rb_write_block(ring_buffer_t *rb, float *data);
bool rb_read_block(ring_buffer_t *rb, float *dest);
int  rb_slots_filled(ring_buffer_t *rb);

#endif