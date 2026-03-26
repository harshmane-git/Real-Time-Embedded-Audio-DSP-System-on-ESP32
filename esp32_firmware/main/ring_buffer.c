#include "ring_buffer.h"
#include <string.h>   // memcpy, memset

// ─────────────────────────────────────────────
//  Helper: circular distance from read to write
//  i.e. how many slots ahead is write_ptr from read_ptr
// ─────────────────────────────────────────────
static uint8_t gap_write_ahead(RingBuffer *rb) {
    return (rb->write_ptr - rb->read_ptr + RB_NUM_SLOTS) % RB_NUM_SLOTS;
}

// ─────────────────────────────────────────────
//  rb_init
// ─────────────────────────────────────────────
void rb_init(RingBuffer *rb) {
    memset(rb, 0, sizeof(RingBuffer));
    for (int i = 0; i < RB_NUM_SLOTS; i++) {
        rb->slots[i].state       = SLOT_FREE;
        rb->slots[i].slot_index  = i;
    }
    rb->write_ptr        = 0;
    rb->read_ptr         = 0;
    rb->overflow_count   = 0;
    rb->underflow_count  = 0;
}

// ─────────────────────────────────────────────
//  rb_is_safe_to_read
//  Read pointer is safe only when write is
//  at least 2 slots ahead of read.
// ─────────────────────────────────────────────
bool rb_is_safe_to_read(RingBuffer *rb) {
    return gap_write_ahead(rb) >= 2;
}

// ─────────────────────────────────────────────
//  rb_slots_available_to_write
//  Write must stay at least 2 behind read
//  (from the other direction) to avoid
//  overwriting slots not yet consumed.
//  Max usable write slots = NUM_SLOTS - 2
// ─────────────────────────────────────────────
uint8_t rb_slots_available_to_write(RingBuffer *rb) {
    uint8_t used = gap_write_ahead(rb);
    if (used >= RB_NUM_SLOTS - 2) return 0;
    return (RB_NUM_SLOTS - 2) - used;
}

uint8_t rb_slots_available_to_read(RingBuffer *rb) {
    return gap_write_ahead(rb);
}

// ─────────────────────────────────────────────
//  rb_write  (called by I2S_write task)
//  Writes one slot worth of samples.
//  Will NOT overwrite a slot that hasn't been
//  read yet (overflow protection).
// ─────────────────────────────────────────────
bool rb_write(RingBuffer *rb, const float *samples, uint16_t len) {
    // Clamp to slot size
    if (len > RB_SLOT_SAMPLES) len = RB_SLOT_SAMPLES;

    // Check if next write slot is free
    RingBufferSlot *slot = &rb->slots[rb->write_ptr];

    if (slot->state != SLOT_FREE) {
        // Write caught up to unread data — overflow
        rb->overflow_count++;
        return false;
    }

    // Copy samples into slot
    memcpy(slot->samples, samples, len * sizeof(float));

    // Zero-pad if fewer samples provided
    if (len < RB_SLOT_SAMPLES) {
        memset(&slot->samples[len], 0,
               (RB_SLOT_SAMPLES - len) * sizeof(float));
    }

    slot->state = SLOT_WRITTEN;

    // Advance write pointer (circular)
    rb->write_ptr = (rb->write_ptr + 1) % RB_NUM_SLOTS;

    return true;
}

// ─────────────────────────────────────────────
//  rb_read  (called by DMA_read task)
//  Copies next ready slot into all 3 filter
//  working buffers (Option A — copy on read).
//  Enforces 2-slot safety gap.
// ─────────────────────────────────────────────
bool rb_read(RingBuffer *rb, FilterWorkBuffers *work) {
    // Safety gap check — read must be at least 2 behind write
    if (!rb_is_safe_to_read(rb)) {
        rb->underflow_count++;
        work->ready = false;
        return false;
    }

    RingBufferSlot *slot = &rb->slots[rb->read_ptr];

    // Slot must be in WRITTEN state
    if (slot->state != SLOT_WRITTEN) {
        rb->underflow_count++;
        work->ready = false;
        return false;
    }

    slot->state = SLOT_READING;

    // ── Option A: copy into all 3 filter buffers ──
    memcpy(work->lpf_buf, slot->samples, RB_SLOT_SAMPLES * sizeof(float));
    memcpy(work->bpf_buf, slot->samples, RB_SLOT_SAMPLES * sizeof(float));
    memcpy(work->hpf_buf, slot->samples, RB_SLOT_SAMPLES * sizeof(float));
    work->ready = true;

    // Mark slot free — all 3 copies taken
    slot->state = SLOT_CONSUMED;
    slot->state = SLOT_FREE;

    // Advance read pointer (circular)
    rb->read_ptr = (rb->read_ptr + 1) % RB_NUM_SLOTS;

    return true;
}
