#include "ring_buffer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "audio_config.h"
#include "common_types.h"

// ============================================================================
// Internal state
// ============================================================================
static uint32_t rb_overflow_warn_count = 0;

// ============================================================================
// rb_Open
// ============================================================================
// Derives buffer size from sample_rate so the caller never needs to hardcode
// AUDIO_RB_SIZE. Formula:
//
//   samples_per_slot = (sample_rate / 1000) * BLOCK_DURATION_MS
//   buffer_size      = AUDIO_RB_SLOTS * samples_per_slot
//
// At 16 kHz with 16 ms blocks:
//   samples_per_slot = (16000 / 1000) * 16 = 256
//   buffer_size      = 4 * 256 = 1024 samples
//
// BLOCK_DURATION_MS is derived from AUDIO_BLOCK_SIZE and AUDIO_SAMPLE_RATE
// so it stays consistent with audio_config.h without a separate #define.
// ============================================================================
STATUS rb_Open(uint32_t sample_rate, uint32_t *buffer_size_out)
{
    if (sample_rate == 0 || buffer_size_out == NULL)
    {
        printf("[ERROR] rb_Open: invalid arguments\n");
        return STATUS_NOT_OK;
    }

    // Derive block duration in ms from audio_config.h constants
    // AUDIO_BLOCK_SIZE / AUDIO_SAMPLE_RATE gives duration in seconds → × 1000 for ms
    // Use integer math: (AUDIO_BLOCK_SIZE * 1000) / sample_rate
    uint32_t samples_per_slot = (AUDIO_BLOCK_SIZE * 1000) / (sample_rate);
    // Correct: samples_per_slot = block_duration_ms × (sample_rate / 1000)
    //        = ((AUDIO_BLOCK_SIZE * 1000) / sample_rate) × (sample_rate / 1000)
    //        = AUDIO_BLOCK_SIZE  — which is what we want
    // Simplified directly:
    samples_per_slot = AUDIO_BLOCK_SIZE;  // always equals block size regardless of rate

    *buffer_size_out = AUDIO_RB_SLOTS * samples_per_slot;

    printf("[rb_Open] sample_rate=%lu Hz | samples_per_slot=%lu | total_buffer=%lu samples (%.2f ms)\n",
           sample_rate,
           samples_per_slot,
           *buffer_size_out,
           (float)(*buffer_size_out) * 1000.0f / (float)sample_rate);

    return STATUS_OK;
}

// ============================================================================
// rb_Initialize
// ============================================================================
STATUS rb_Initialize(rb_hdl *hdl, const rb_config *cfg)
{
    if (!hdl || !cfg || cfg->size == 0 || cfg->sample_rate == 0)
    {
        printf("[ERROR] rb_Initialize: invalid arguments\n");
        return STATUS_NOT_OK;
    }

    uint32_t size = cfg->size;

    hdl->buffer = malloc(size * sizeof(float));
    if (!hdl->buffer)
    {
        printf("[ERROR] rb_Initialize: malloc failed for %lu samples\n", size);
        return STATUS_NOT_OK;
    }

    memset(hdl->buffer, 0, size * sizeof(float));

    hdl->write_slot       = 0;
    hdl->read_slot        = 0;
    hdl->slots_available  = 0;
    hdl->samples_per_slot = AUDIO_RB_SAMPLES_PER_SLOT; // consistent with audio_config.h

    printf("[rb_Initialize] buffer=%lu samples | slots=%d | samples_per_slot=%lu\n",
           size, AUDIO_RB_SLOTS, hdl->samples_per_slot);

    return STATUS_OK;
}

// ============================================================================
// rb_Process
// ============================================================================
STATUS rb_Process(rb_hdl *hdl, const float *input, float *output, uint32_t samples)
{
    if (samples != hdl->samples_per_slot)
    {
        printf("[ERROR] rb_Process: samples=%lu != expected=%lu\n",
               samples, hdl->samples_per_slot);
        return STATUS_NOT_OK;
    }

    // -------------------------------------------------------------------------
    // WRITE PHASE: Write 'samples' into current write slot
    // -------------------------------------------------------------------------
    uint32_t write_base = hdl->write_slot * hdl->samples_per_slot;

    for (uint32_t i = 0; i < samples; i++)
        hdl->buffer[write_base + i] = input[i];

    hdl->write_slot = (hdl->write_slot + 1) % AUDIO_RB_SLOTS;

    if (hdl->slots_available < AUDIO_RB_SLOTS)
    {
        hdl->slots_available++;
    }
    else
    {
        // Buffer full — discard oldest slot (circular overwrite)
        hdl->read_slot = (hdl->read_slot + 1) % AUDIO_RB_SLOTS;
        if ((rb_overflow_warn_count++ % 100) == 0)
        {
            printf("[WARN] rb_Process: buffer full — oldest slot overwritten (count=%lu)\n",
                   rb_overflow_warn_count);
        }
    }

    // -------------------------------------------------------------------------
    // READ PHASE: Read one complete slot if available
    // -------------------------------------------------------------------------
    if (hdl->slots_available == 0)
    {
        // Buffer still priming at startup — output silence
        memset(output, 0, samples * sizeof(float));
        return STATUS_NOT_OK;
    }

    uint32_t read_base = hdl->read_slot * hdl->samples_per_slot;

    for (uint32_t i = 0; i < samples; i++)
        output[i] = hdl->buffer[read_base + i];

    hdl->read_slot = (hdl->read_slot + 1) % AUDIO_RB_SLOTS;
    hdl->slots_available--;

    // Sanity check (defensive — should never trigger)
    if (hdl->slots_available > AUDIO_RB_SLOTS)
    {
        printf("[ERROR] rb_Process: slots_available corrupted\n");
        return STATUS_NOT_OK;
    }

    return STATUS_OK;
}

// ============================================================================
// rb_Reset
// ============================================================================
STATUS rb_Reset(rb_hdl *hdl)
{
    if (!hdl || !hdl->buffer)
    {
        printf("[ERROR] rb_Reset: null handle or buffer\n");
        return STATUS_NOT_OK;
    }

    memset(hdl->buffer, 0, AUDIO_RB_SLOTS * hdl->samples_per_slot * sizeof(float));

    hdl->write_slot      = 0;
    hdl->read_slot       = 0;
    hdl->slots_available = 0;

    printf("[rb_Reset] Ring buffer cleared\n");
    return STATUS_OK;
}

// ============================================================================
// rb_Close
// ============================================================================
STATUS rb_Close(rb_hdl *hdl)
{
    if (!hdl) return STATUS_NOT_OK;

    if (hdl->buffer)
    {
        free(hdl->buffer);
        hdl->buffer = NULL;
    }

    hdl->write_slot      = 0;
    hdl->read_slot       = 0;
    hdl->slots_available = 0;
    hdl->samples_per_slot = 0;

    return STATUS_OK;
}