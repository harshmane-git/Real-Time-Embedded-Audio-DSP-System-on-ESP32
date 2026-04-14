#include "ring_buffer.h"
#include <stdlib.h>
#include <string.h>

STATUS rb_Open(uint32_t *size)
{
    *size = sizeof(rb_hdl);
    return STATUS_OK;
}

STATUS rb_Initialize(rb_hdl *hdl, const rb_config *cfg)
{
    // Allocate: 8 slots × 256 samples × sizeof(float) = 8KB
    hdl->buffer = malloc(RB_TOTAL_SAMPLES * sizeof(float));

    if (!hdl->buffer)
        return STATUS_NOT_OK;

    // Initialize all slots to zero
    memset(hdl->buffer, 0, RB_TOTAL_SAMPLES * sizeof(float));

    hdl->write_slot = 0;
    hdl->read_slot = 0;
    hdl->write_pos = 0;
    hdl->read_pos = 0;
    hdl->slots_available = 0;

    return STATUS_OK;
}

STATUS rb_Process(rb_hdl *hdl, const float *input, float *output, uint32_t samples)
{
    // Ensure we're processing BLOCK_SIZE (256) samples
    if (samples != RB_SAMPLES_PER_SLOT)
        return STATUS_NOT_OK;

    // 🔹 WRITE: Write 256 samples into current write slot
    uint32_t write_base_index = hdl->write_slot * RB_SAMPLES_PER_SLOT;
    
    for (uint32_t i = 0; i < samples; i++)
    {
        hdl->buffer[write_base_index + i] = input[i];
    }

    // Move to next slot after writing complete slot
    hdl->write_slot = (hdl->write_slot + 1) % RB_SLOTS;
    hdl->slots_available++;

    // 🔹 READ: Check if we have complete slot available
    if (hdl->slots_available <= 0)
    {
        // No complete slot available for reading
        return STATUS_NOT_OK;
    }

    // Read 256 samples from current read slot
    uint32_t read_base_index = hdl->read_slot * RB_SAMPLES_PER_SLOT;
    
    for (uint32_t i = 0; i < samples; i++)
    {
        output[i] = hdl->buffer[read_base_index + i];
    }

    // Move to next slot after reading complete slot
    hdl->read_slot = (hdl->read_slot + 1) % RB_SLOTS;
    hdl->slots_available--;

    // Ensure read doesn't exceed write
    if (hdl->read_slot == hdl->write_slot && hdl->slots_available < 0)
    {
        return STATUS_NOT_OK;  // Read caught up to write
    }

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
    hdl->write_pos = 0;
    hdl->read_pos = 0;
    hdl->slots_available = 0;
    
    return STATUS_OK;
}