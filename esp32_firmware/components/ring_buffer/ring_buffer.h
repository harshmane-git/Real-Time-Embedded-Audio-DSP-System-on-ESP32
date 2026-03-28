// ring_buffer.h
#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include <stdint.h>
#include <stdbool.h>
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

#define RB_NUM_SLOTS     8
#define RB_SAMPLES_PER_SLOT  256
#define RB_TOTAL_SAMPLES (RB_NUM_SLOTS * RB_SAMPLES_PER_SLOT)  // 2048

// Slot states
typedef enum {
    SLOT_EMPTY,
    SLOT_WRITING,
    SLOT_READY,
    SLOT_READING
} slot_state_t;

typedef struct {
    int32_t     samples[RB_SAMPLES_PER_SLOT];  // 256 × 4 bytes = 1KB per slot
    slot_state_t state;
} rb_slot_t;

typedef struct {
    rb_slot_t   slots[RB_NUM_SLOTS];           // 8 slots = 8KB
    volatile int write_slot;                    // I2S write pointer (slot index)
    volatile int read_slot;                     // DMA read pointer (slot index)
    volatile int slots_filled;                  // how many slots are ready to read
    SemaphoreHandle_t mutex;
} ring_buffer_t;

void     rb_init(ring_buffer_t *rb);
bool     I2S_write(ring_buffer_t *rb, int32_t *data, int len); // write 256 samples into next slot
bool     DMA_read(ring_buffer_t *rb, int32_t *dest);           // read 256 samples from ready slot
int      rb_slots_available(ring_buffer_t *rb);

#endif