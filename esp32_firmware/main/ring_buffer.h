#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include <stdint.h>
#include <stdbool.h>

// ─────────────────────────────────────────────
//  Ring Buffer Specifications
//  Sample rate : 16000 Hz
//  Slot size   : 256 float samples = 1 KB
//  Total slots : 8  →  8 KB total
//  Safety gap  : 2 slots between write and read
//  Timing      : each slot = 256/16000 = 16 ms
// ─────────────────────────────────────────────

#define RB_NUM_SLOTS     8
#define RB_SLOT_SAMPLES  256
#define RB_SAMPLE_RATE   16000

// Slot state flags
typedef enum {
    SLOT_FREE      = 0,   // available for I2S_write
    SLOT_WRITTEN   = 1,   // written by I2S, ready for DMA_read
    SLOT_READING   = 2,   // currently being copied by DMA_read
    SLOT_CONSUMED  = 3    // all 3 filter copies taken, safe to free
} SlotState;

// Individual slot
typedef struct {
    float      samples[RB_SLOT_SAMPLES];  // 256 × 4 = 1024 bytes = 1 KB
    SlotState  state;
    uint32_t   slot_index;
} RingBufferSlot;

// Full ring buffer
typedef struct {
    RingBufferSlot slots[RB_NUM_SLOTS];
    volatile uint8_t write_ptr;   // I2S_write advances this
    volatile uint8_t read_ptr;    // DMA_read advances this
    volatile uint32_t overflow_count;   // write caught up to read
    volatile uint32_t underflow_count;  // read caught up to write
} RingBuffer;

// Working buffers — one per filter (Option A: copy on read)
typedef struct {
    float lpf_buf[RB_SLOT_SAMPLES];   // copy for Low Pass Filter
    float bpf_buf[RB_SLOT_SAMPLES];   // copy for Band Pass Filter
    float hpf_buf[RB_SLOT_SAMPLES];   // copy for High Pass Filter
    bool  ready;                       // true = all 3 copies are fresh
} FilterWorkBuffers;

// ─── API ───────────────────────────────────────
#ifdef __cplusplus
extern "C" {
#endif

// Initialise — call once in setup()
void rb_init(RingBuffer *rb);

// I2S_write: fill next free slot with 'samples'
// Returns true on success, false if buffer is full (overflow)
bool rb_write(RingBuffer *rb, const float *samples, uint16_t len);

// DMA_read: copy next ready slot into all 3 filter work buffers
// Enforces 2-slot safety gap — will not read if gap < 2
// Returns true on success, false if not enough data yet (underflow)
bool rb_read(RingBuffer *rb, FilterWorkBuffers *work);

// Utility
uint8_t rb_slots_available_to_write(RingBuffer *rb);
uint8_t rb_slots_available_to_read(RingBuffer *rb);
bool    rb_is_safe_to_read(RingBuffer *rb);  // gap check

#ifdef __cplusplus
}
#endif

#endif // RING_BUFFER_H
