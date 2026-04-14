#include "ring_buffer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static uint32_t rb_overflow_warn_count = 0;
STATUS rb_Open(uint32_t *size)
{
    *size = sizeof(rb_hdl);
    return STATUS_OK;
}

STATUS rb_Initialize(rb_hdl *hdl, const rb_config *cfg)
{
    uint32_t size = cfg->size;

    hdl->buffer = malloc(size * sizeof(float));
    if (!hdl->buffer)
        return STATUS_NOT_OK;

    memset(hdl->buffer, 0, size * sizeof(float));

    hdl->write_slot = 0;
    hdl->read_slot = 0;
    hdl->slots_available = 0;

    return STATUS_OK;
}

STATUS rb_Process(rb_hdl *hdl, const float *input, float *output, uint32_t samples)
{
    // Ensure we're processing BLOCK_SIZE (256) samples per slot
    if (samples != AUDIO_RB_SAMPLES_PER_SLOT)
        return STATUS_NOT_OK;

    // =========================================================================
    // WRITE PHASE: Write 256 samples into current write slot
    // =========================================================================
    uint32_t write_base_index = hdl->write_slot * AUDIO_RB_SAMPLES_PER_SLOT;
    
    for (uint32_t i = 0; i < samples; i++)
    {
        hdl->buffer[write_base_index + i] = input[i];
    }

    // Move to next slot after writing complete slot
    hdl->write_slot = (hdl->write_slot + 1) % AUDIO_RB_SLOTS;

    // Increment available slots with overflow protection
    if (hdl->slots_available < AUDIO_RB_SLOTS)
    {
        hdl->slots_available++;
    }
    else
    {
        // Buffer is full - circular overwrite mode
        // Auto-advance read pointer to discard oldest slot
        hdl->read_slot = (hdl->read_slot + 1) % AUDIO_RB_SLOTS;
        if ((rb_overflow_warn_count++ % 100) == 0)
        {
            printf("[WARN] Ring buffer full - oldest slot overwritten\n");
        }    
    }

    // =========================================================================
    // READ PHASE: Check if we have complete slot available
    // =========================================================================
    if (hdl->slots_available == 0)
    {
        // No complete slot available for reading - buffer still priming (startup condition)
        memset(output, 0, samples * sizeof(float));  // Output silence
        return STATUS_NOT_OK;
    }

    // Read 256 samples from current read slot
    uint32_t read_base_index = hdl->read_slot * AUDIO_RB_SAMPLES_PER_SLOT;
    
    for (uint32_t i = 0; i < samples; i++)
    {
        output[i] = hdl->buffer[read_base_index + i];
    }

    // Move to next slot after reading complete slot
    hdl->read_slot = (hdl->read_slot + 1) % AUDIO_RB_SLOTS;

    // Decrement available slots
    hdl->slots_available--;

    // Sanity check: slots_available should never be negative (unsigned)
    // This is defensive programming - should never happen with correct logic
    if (hdl->slots_available > AUDIO_RB_SLOTS)
    {
        printf("[ERROR] Ring buffer state corrupted - slots_available overflow\n");
        return STATUS_NOT_OK;
    }

    return STATUS_OK;
}

STATUS rb_Reset(rb_hdl *hdl)
{
    // Safe reset through API instead of direct field manipulation
    if (!hdl || !hdl->buffer)
        return STATUS_NOT_OK;

    // Clear all data
    memset(hdl->buffer, 0, AUDIO_RB_SIZE * sizeof(float));

    // Reset state
    hdl->write_slot = 0;
    hdl->read_slot = 0;
    hdl->slots_available = 0;

    return STATUS_OK;
}

STATUS rb_Close(rb_hdl *hdl)
{
    if (hdl->buffer)
    {
        free(hdl->buffer);
        hdl->buffer = NULL;
    }
    hdl->write_slot = 0;
    hdl->read_slot = 0;
    hdl->slots_available = 0;

    return STATUS_OK;
}