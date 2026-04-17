#include "audio_pipeline.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// ============================================================================
// Audio Pipeline Buffers
// ============================================================================
static float mic_block[AUDIO_BLOCK_SIZE];        // Raw mic input (I2S)
static float rb_output_block[AUDIO_BLOCK_SIZE];  // Ring buffer output
static float dsp_output_block[AUDIO_BLOCK_SIZE]; // After DSP processing

volatile int preset_request = 0;
static int current_preset = 0;

// Statistics for monitoring
static uint32_t frame_count = 0;
static uint32_t rb_underflow_count = 0;
static uint32_t preset_changes = 0;

// ============================================================================
// DSP Configuration - Presets
// ============================================================================
typedef struct
{
    float gain;
} preset_config;

static preset_config presets[NUM_PRESETS] =
{
    {PRESET_GAIN_0},    // Preset 0: Default (1.0x)
    {PRESET_GAIN_1},    // Preset 1: -6dB (0.5x)
    {PRESET_GAIN_2}     // Preset 2: +6dB (2.0x)
};

// ============================================================================
// Internal Helper Functions
// ============================================================================

/**
 * Validate and apply preset change
 * @param new_preset: Requested preset index
 * @return STATUS_OK if valid, STATUS_NOT_OK if invalid
 */
static STATUS audio_apply_preset(int new_preset)
{
    // Bounds check - prevent out-of-range access
    if (new_preset < PRESET_MIN || new_preset > PRESET_MAX)
    {
        printf("[ERROR] Invalid preset: %d (valid range: %d-%d)\n", 
               new_preset, PRESET_MIN, PRESET_MAX);
        return STATUS_NOT_OK;
    }

    current_preset = new_preset;
    preset_changes++;

    return STATUS_OK;
}

/**
 * Safe ring buffer reset through API
 * @param rb: Ring buffer handle
 * @return STATUS_OK if successful
 */
/*static STATUS audio_reset_ring_buffer(rb_hdl *rb)
{
    // Use API instead of directly manipulating internals
    STATUS status = rb_Reset(rb);
    
    if (status != STATUS_OK)
    {
        printf("[ERROR] Ring buffer reset failed\n");
        return STATUS_NOT_OK;
    }

    return STATUS_OK;
}*/

// ============================================================================
// DSP Processing Layer
// ============================================================================
// Applies DSP processing (gain, EQ, limiter, delay) to audio block
// Separated from transport layer for clean ownership by DSP team
//
// TODO (DSP Implementation Team):
//   - Add EQ filter implementation
//   - Add Limiter with lookahead
//   - Add Delay effect
// ============================================================================
static void pipeline_process_dsp(float *input, float *output, uint32_t samples)
{
    // Apply current preset gain
    for (uint32_t i = 0; i < samples; i++)
    {
        output[i] = input[i] * presets[current_preset].gain;
    }

    // Additional DSP processing will be added here by implementation team:
    // - EQ filtering (~2ms)
    // - Limiter with lookahead (~5ms)
    // - Delay effect (~50ms for large delays)
    // - Dynamic processing
    // - etc.
}

// ============================================================================
// Audio Pipeline Public API
// ============================================================================

STATUS audio_Open(audio_hdl *hdl)
{
    static mic_hdl mic_instance;
    static amp_hdl amp_instance;
    static rb_hdl rb_instance;

    hdl->mic = &mic_instance;
    hdl->amp = &amp_instance;
    hdl->rb  = &rb_instance;

    return STATUS_OK;
}

STATUS audio_Initialize(audio_hdl *hdl)
{
    mic_config mic_cfg = {AUDIO_SAMPLE_RATE};
    amp_config amp_cfg = {AUDIO_SAMPLE_RATE};
    rb_config rb_cfg = {AUDIO_RB_SIZE};

    if (mic_Initialize(hdl->mic, &mic_cfg) != STATUS_OK)
    {
        printf("[ERROR] Mic initialization failed\n");
        return STATUS_NOT_OK;
    }

    if (amp_Initialize(hdl->amp, &amp_cfg) != STATUS_OK)
    {
        printf("[ERROR] Amp initialization failed\n");
        return STATUS_NOT_OK;
    }

    if (rb_Initialize(hdl->rb, &rb_cfg) != STATUS_OK)
    {
        printf("[ERROR] Ring buffer initialization failed\n");
        return STATUS_NOT_OK;
    }

    // Reset statistics
    frame_count = 0;
    rb_underflow_count = 0;
    preset_changes = 0;

    printf("Audio pipeline initialized:\n");
    printf("  Sample rate: %u Hz\n", AUDIO_SAMPLE_RATE);
    printf("  Block size: %u samples (%.2f ms)\n", AUDIO_BLOCK_SIZE, 
           (float)AUDIO_BLOCK_SIZE * 1000 / AUDIO_SAMPLE_RATE);
    printf("  Ring buffer: %u samples / %u slots (capacity: %.2f ms, pipeline latency: %.2f ms)\n",
        AUDIO_RB_SIZE, AUDIO_RB_SLOTS,
        (float)AUDIO_RB_SIZE * 1000 / AUDIO_SAMPLE_RATE,
        (float)AUDIO_BLOCK_SIZE * 1000 / AUDIO_SAMPLE_RATE);
    printf("  Presets: %u available (0-%u)\n", NUM_PRESETS, PRESET_MAX);

    return STATUS_OK;
}

STATUS audio_Process(audio_hdl *hdl)
{
    frame_count++;

    // 🔹 Step 1: Mic input
    if (mic_Process(hdl->mic, mic_block, AUDIO_BLOCK_SIZE) != STATUS_OK)
    {
        printf("[ERROR] Mic read failed at frame %lu\n", frame_count);
        return STATUS_NOT_OK;
    }

    // 🔹 Step 2: Preset update (block boundary safe)
    if (preset_request != 0)
    {
        if (audio_apply_preset(preset_request) == STATUS_OK)
        {
            if (rb_Reset(hdl->rb) == STATUS_OK)
            {
                printf("Preset changed to %d (gain=%.2f)\n",
                       current_preset, presets[current_preset].gain);
            }
        }
        preset_request = 0;
    }

    // 🔹 Step 3: Ring buffer transport
    STATUS rb_status = rb_Process(hdl->rb, mic_block, rb_output_block, AUDIO_BLOCK_SIZE);

    if (rb_status != STATUS_OK)
    {
        rb_underflow_count++;

        // Recovery: output silence
        memset(rb_output_block, 0, AUDIO_BLOCK_SIZE * sizeof(float));

        if ((frame_count % 100) == 0)
        {
            printf("[WARN] RB underflow (total=%lu)\n", rb_underflow_count);
        }
    }

    // 🔹 Step 4: DSP processing
    pipeline_process_dsp(rb_output_block, dsp_output_block, AUDIO_BLOCK_SIZE);

    // 🔹 Step 5: Output to amp
    if (amp_Process(hdl->amp, dsp_output_block, AUDIO_BLOCK_SIZE) != STATUS_OK)
    {
        printf("[ERROR] Amp write failed at frame %lu\n", frame_count);
        return STATUS_NOT_OK;
    }

    return STATUS_OK;
}

STATUS audio_Close(audio_hdl *hdl)
{
    if (!hdl) return STATUS_NOT_OK;

    if (hdl->rb)
    {
        rb_Close(hdl->rb);
    }

    if (hdl->mic)
    {
        mic_Close(hdl->mic);
    }

    if (hdl->amp)
    {
        amp_Close(hdl->amp);
    }

    printf("Audio pipeline closed\n");
    printf("Frames: %lu | RB underflows: %lu | Preset changes: %lu\n",
           frame_count, rb_underflow_count, preset_changes);

    return STATUS_OK;
}