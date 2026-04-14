#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include "common_types.h"
#include <stdint.h>

// Slot-based ring buffer configuration
#define RB_SLOTS 8
#define RB_SAMPLES_PER_SLOT 256
#define RB_TOTAL_SAMPLES (RB_SLOTS * RB_SAMPLES_PER_SLOT)  // 2048

typedef struct
{
    uint32_t size;  // Not used with slot-based design, kept for API compatibility
} rb_config;

typedef struct
{
    float *buffer;              // Total: 8 slots × 256 samples × 4 bytes = 8KB
    uint32_t write_slot;        // Current write slot (0-7)
    uint32_t read_slot;         // Current read slot (0-7)
    uint32_t write_pos;         // Position within write slot (0-255)
    uint32_t read_pos;          // Position within read slot (0-255)
    uint32_t slots_available;   // Number of complete slots available for reading
} rb_hdl;

STATUS rb_Open(uint32_t *size);
STATUS rb_Initialize(rb_hdl *hdl, const rb_config *cfg);
STATUS rb_Process(rb_hdl *hdl, const float *input, float *output, uint32_t samples);
STATUS rb_Close(rb_hdl *hdl);

#endif