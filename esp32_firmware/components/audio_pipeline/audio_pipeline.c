#include "audio_pipeline.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "audio_config.h"
#include "common_types.h"

// ============================================================================
// Audio Pipeline Buffers
// ============================================================================
static float mic_block[AUDIO_BLOCK_SIZE];        // Raw mic input from I2S
static float rb_output_block[AUDIO_BLOCK_SIZE];  // Ring buffer output
static float dsp_output_block[AUDIO_BLOCK_SIZE]; // After full DSP chain

volatile int preset_request = 0;
static int  current_preset  = 0;

// Statistics
static uint32_t frame_count        = 0;
static uint32_t rb_underflow_count = 0;
static uint32_t preset_changes     = 0;

// ============================================================================
// Filter handles — one set per filter type
// ============================================================================
static low_pass_hdl_t  lpf_hdl;
static band_pass_hdl_t bpf_hdl;
static high_pass_hdl_t hpf_hdl;

// Scratch buffers — each filter writes into its own independent buffer
static float lpf_out[AUDIO_BLOCK_SIZE];
static float bpf_out[AUDIO_BLOCK_SIZE];
static float hpf_out[AUDIO_BLOCK_SIZE];

// ============================================================================
// DSP Configuration - Presets
// ============================================================================
// Signal flow:
//
//   Mic → RingBuffer → [LP | BP | HP] (parallel) → weighted sum → master gain
//                                                                      ↓
//                                                                  [Delay]   ← TODO
//                                                                      ↓
//                                                                 [Limiter]  ← TODO
//                                                                      ↓
//                                                                    Amp
//
// Master gain is applied AFTER the EQ sum so the filters always receive
// a clean normalized signal. Applying gain before filters would distort
// the input and invalidate the filter frequency response.
//
// Each preset carries:
//   - Its own filter coefficients (lp/bp/hp) — reloaded on preset change
//   - Per-band gains  (bass / mid / treble)  — applied inside EQ sum
//   - Master gain                             — applied after EQ sum
// ============================================================================

typedef struct
{
    // Filter coefficients for this preset
    low_pass_config_t  lp_cfg;
    band_pass_config_t bp_cfg;
    high_pass_config_t hp_cfg;

    // Mixing gains
    float master_gain;  // applied AFTER EQ sum
    float bass_gain;    // applied to LP output inside sum
    float mid_gain;     // applied to BP output inside sum
    float treble_gain;  // applied to HP output inside sum
} preset_config;

static const preset_config presets[NUM_PRESETS] =
{
    // -------------------------------------------------------------------------
    // Preset 0: Flat — no coloration, unity gain on all bands
    // -------------------------------------------------------------------------
    {
        .lp_cfg = {
            .s1 = {0.00312629f,  0.00625258f,  0.00312629f, -1.79158896f,  0.80409412f},
            .s2 = {0.00331660f,  0.00663319f,  0.00331660f, -1.90064888f,  0.91391527f}
        },
        .bp_cfg = {
            .s1 = {0.12491506f,  0.00000000f, -0.12491506f, -1.66451047f,  0.75016988f},
            .s2 = {0.05582542f,  0.00000000f, -0.05582542f, -1.79592677f,  0.88834915f}
        },
        .hp_cfg = {
            .s1 = {0.51627979f, -1.03255959f,  0.51627979f, -0.85540037f,  0.20971880f},
            .s2 = {0.67177700f, -1.34355400f,  0.67177700f, -1.11303657f,  0.57407142f}
        },
        .master_gain = 1.0f,
        .bass_gain   = 1.0f,
        .mid_gain    = 1.0f,
        .treble_gain = 1.0f,
    },

    // -------------------------------------------------------------------------
    // Preset 1: Bass boost
    // -------------------------------------------------------------------------
    {
        .lp_cfg = {
            .s1 = {0.00312629f,  0.00625258f,  0.00312629f, -1.79158896f,  0.80409412f},
            .s2 = {0.00331660f,  0.00663319f,  0.00331660f, -1.90064888f,  0.91391527f}
        },
        .bp_cfg = {
            .s1 = {0.12491506f,  0.00000000f, -0.12491506f, -1.66451047f,  0.75016988f},
            .s2 = {0.05582542f,  0.00000000f, -0.05582542f, -1.79592677f,  0.88834915f}
        },
        .hp_cfg = {
            .s1 = {0.51627979f, -1.03255959f,  0.51627979f, -0.85540037f,  0.20971880f},
            .s2 = {0.67177700f, -1.34355400f,  0.67177700f, -1.11303657f,  0.57407142f}
        },
        .master_gain = 0.5f,
        .bass_gain   = 1.5f,
        .mid_gain    = 1.0f,
        .treble_gain = 0.5f,
    },

    // -------------------------------------------------------------------------
    // Preset 2: Treble boost
    // -------------------------------------------------------------------------
    {
        .lp_cfg = {
            .s1 = {0.00312629f,  0.00625258f,  0.00312629f, -1.79158896f,  0.80409412f},
            .s2 = {0.00331660f,  0.00663319f,  0.00331660f, -1.90064888f,  0.91391527f}
        },
        .bp_cfg = {
            .s1 = {0.12491506f,  0.00000000f, -0.12491506f, -1.66451047f,  0.75016988f},
            .s2 = {0.05582542f,  0.00000000f, -0.05582542f, -1.79592677f,  0.88834915f}
        },
        .hp_cfg = {
            .s1 = {0.51627979f, -1.03255959f,  0.51627979f, -0.85540037f,  0.20971880f},
            .s2 = {0.67177700f, -1.34355400f,  0.67177700f, -1.11303657f,  0.57407142f}
        },
        .master_gain = 2.0f,
        .bass_gain   = 0.7f,
        .mid_gain    = 1.0f,
        .treble_gain = 1.5f,
    },
};

// ============================================================================
// Internal: audio_restart_system
// ============================================================================
// Full system restart sequence for preset change:
//   1. Close mic + amp I2S peripherals
//   2. Reset ring buffer (clears all buffered audio — avoids stale data
//      from old preset playing through new filter coefficients)
//   3. Re-initialize mic + amp with same sample rate
//   4. Re-initialize all three filters with NEW preset coefficients
//      (biquad delay lines x1,x2,y1,y2 are cleared inside each init)
//
// Why full restart instead of just resetting filter state?
//   - I2S DMA may have stale data in its internal buffers from the old preset
//   - Ring buffer slots may contain audio processed with old coefficients
//   - A clean restart ensures no old-preset audio leaks into new-preset output
// ============================================================================
static STATUS audio_restart_system(audio_hdl *hdl, int new_preset)
{
    printf("[PRESET] Restarting system for preset %d...\n", new_preset);

    // Step 1: Close hardware peripherals
    mic_Close(hdl->mic);
    amp_Close(hdl->amp);

    // Step 2: Reset ring buffer — discard all stale audio
    if (rb_Reset(hdl->rb) != STATUS_OK)
    {
        printf("[ERROR] rb_Reset failed during preset restart\n");
        return STATUS_NOT_OK;
    }

    // Step 3: Re-initialize hardware with same sample rate
    mic_config mic_cfg = {AUDIO_SAMPLE_RATE};
    amp_config amp_cfg = {AUDIO_SAMPLE_RATE};

    if (mic_Initialize(hdl->mic, &mic_cfg) != STATUS_OK)
    {
        printf("[ERROR] mic_Initialize failed during preset restart\n");
        return STATUS_NOT_OK;
    }

    if (amp_Initialize(hdl->amp, &amp_cfg) != STATUS_OK)
    {
        printf("[ERROR] amp_Initialize failed during preset restart\n");
        return STATUS_NOT_OK;
    }

    // Step 4: Load new filter coefficients and clear biquad delay lines
    low_pass_init (&lpf_hdl, &presets[new_preset].lp_cfg);
    band_pass_init(&bpf_hdl, &presets[new_preset].bp_cfg);
    high_pass_init(&hpf_hdl, &presets[new_preset].hp_cfg);

    current_preset = new_preset;
    preset_changes++;

    printf("[PRESET] System restarted — preset=%d master_gain=%.2f "
           "bass=%.2f mid=%.2f treble=%.2f\n",
           current_preset,
           presets[current_preset].master_gain,
           presets[current_preset].bass_gain,
           presets[current_preset].mid_gain,
           presets[current_preset].treble_gain);

    return STATUS_OK;
}

// ============================================================================
// Internal: pipeline_process_dsp
// ============================================================================
// DSP signal flow:
//
//   input (from ring buffer)
//       │
//       ├──→ LP filter → lpf_out × bass_gain ──┐
//       ├──→ BP filter → bpf_out × mid_gain  ──┤→ SUM → × master_gain → hard clip → output
//       └──→ HP filter → hpf_out × treble_gain ┘
//
// NOTE: master_gain is applied AFTER the EQ sum.
//       Filters receive the raw normalized mic signal.
//       This preserves filter frequency response accuracy.
//
// TODO: Insert Delay block after master_gain application
// TODO: Insert Limiter block after Delay
// ============================================================================
static void pipeline_process_dsp(float *input, float *output, uint32_t samples)
{
    const preset_config *p = &presets[current_preset];

    // Step A: Run three filters IN PARALLEL on the raw input
    // Each filter receives the same unmodified signal from the ring buffer
    low_pass_process (&lpf_hdl, input, lpf_out, samples);  // bass:   0 Hz – ~300 Hz
    band_pass_process(&bpf_hdl, input, bpf_out, samples);  // mid:  ~300 Hz – ~3 kHz
    high_pass_process(&hpf_hdl, input, hpf_out, samples);  // treble: ~3 kHz+

    // Step B: Weighted sum of band outputs + master gain applied AFTER sum
    for (uint32_t i = 0; i < samples; i++)
    {
        // EQ sum with per-band gains
        float out = (lpf_out[i] * p->bass_gain)
                  + (bpf_out[i] * p->mid_gain)
                  + (hpf_out[i] * p->treble_gain);

        // Master gain applied after EQ
        // Gain > 1.0 scales up the already-filtered signal — does not affect
        // filter frequency response since filters already ran on clean input
        out *= p->master_gain;

        // TODO Step C: Apply Delay effect here
        // out = delay_process(&delay_hdl, out);

        // TODO Step D: Apply Limiter here
        // out = limiter_process(&limiter_hdl, out);

        // Hard clip — amp expects samples strictly within [-1.0, 1.0]
        if (out >  1.0f) out =  1.0f;
        if (out < -1.0f) out = -1.0f;

        output[i] = out;
    }
}

// ============================================================================
// Audio Pipeline Public API
// ============================================================================

STATUS audio_Open(audio_hdl *hdl)
{
    if (!hdl) return STATUS_NOT_OK;

    static mic_hdl mic_instance;
    static amp_hdl amp_instance;
    static rb_hdl  rb_instance;

    hdl->mic = &mic_instance;
    hdl->amp = &amp_instance;
    hdl->rb  = &rb_instance;

    return STATUS_OK;
}

STATUS audio_Initialize(audio_hdl *hdl)
{
    if (!hdl) return STATUS_NOT_OK;

    // --- Query ring buffer size from sample rate ---
    uint32_t rb_buffer_size = 0;
    if (rb_Open(AUDIO_SAMPLE_RATE, &rb_buffer_size) != STATUS_OK)
    {
        printf("[ERROR] rb_Open failed\n");
        return STATUS_NOT_OK;
    }

    // --- Initialize hardware ---
    mic_config mic_cfg = {AUDIO_SAMPLE_RATE};
    amp_config amp_cfg = {AUDIO_SAMPLE_RATE};
    rb_config  rb_cfg  = {
        .size        = rb_buffer_size,
        .sample_rate = AUDIO_SAMPLE_RATE
    };

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

    // --- Initialize filters with Preset 0 coefficients ---
    low_pass_init (&lpf_hdl, &presets[0].lp_cfg);
    band_pass_init(&bpf_hdl, &presets[0].bp_cfg);
    high_pass_init(&hpf_hdl, &presets[0].hp_cfg);

    current_preset = 0;

    // Reset statistics
    frame_count        = 0;
    rb_underflow_count = 0;
    preset_changes     = 0;

    printf("Audio pipeline initialized:\n");
    printf("  Sample rate  : %u Hz\n", AUDIO_SAMPLE_RATE);
    printf("  Block size   : %u samples (%.2f ms)\n",
           AUDIO_BLOCK_SIZE,
           (float)AUDIO_BLOCK_SIZE * 1000.0f / AUDIO_SAMPLE_RATE);
    printf("  RB size      : %u samples / %u slots (%.2f ms capacity)\n",
           rb_buffer_size, AUDIO_RB_SLOTS,
           (float)rb_buffer_size * 1000.0f / AUDIO_SAMPLE_RATE);
    printf("  Active preset: %d (master_gain=%.2f)\n",
           current_preset, presets[current_preset].master_gain);
    printf("  Signal flow  : Mic → RB → [LP|BP|HP] → Sum → Gain → [Delay] → [Limiter] → Amp\n");

    return STATUS_OK;
}

STATUS audio_Process(audio_hdl *hdl)
{
    if (!hdl) return STATUS_NOT_OK;

    frame_count++;

    // Step 1: Read block from mic (I2S)
    if (mic_Process(hdl->mic, mic_block, AUDIO_BLOCK_SIZE) != STATUS_OK)
    {
        printf("[ERROR] Mic read failed at frame %lu\n", frame_count);
        return STATUS_NOT_OK;
    }

    // Step 2: Preset change — full system restart at block boundary
    // Checked here (after mic read, before RB write) so we never write
    // stale audio into the ring buffer with new-preset filters active
    if (preset_request != 0)
    {
        int requested = preset_request;
        preset_request = 0; // clear flag immediately to avoid re-entry

        if (requested < PRESET_MIN || requested > PRESET_MAX)
        {
            printf("[ERROR] Invalid preset request: %d (valid: %d-%d)\n",
                   requested, PRESET_MIN, PRESET_MAX);
        }
        else
        {
            if (audio_restart_system(hdl, requested) != STATUS_OK)
            {
                printf("[ERROR] Preset restart failed — continuing with preset %d\n",
                       current_preset);
            }
        }
    }

    // Step 3: Ring buffer transport (mic → RB → DSP decoupling)
    STATUS rb_status = rb_Process(hdl->rb, mic_block, rb_output_block, AUDIO_BLOCK_SIZE);

    if (rb_status != STATUS_OK)
    {
        rb_underflow_count++;
        memset(rb_output_block, 0, AUDIO_BLOCK_SIZE * sizeof(float));

        if ((frame_count % 100) == 0)
        {
            printf("[WARN] RB underflow at frame %lu (total=%lu)\n",
                   frame_count, rb_underflow_count);
        }
    }

    // Step 4: DSP chain — EQ → master gain → [Delay] → [Limiter]
    pipeline_process_dsp(rb_output_block, dsp_output_block, AUDIO_BLOCK_SIZE);

    // Step 5: Write processed block to amp (I2S)
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

    if (hdl->rb)  rb_Close(hdl->rb);
    if (hdl->mic) mic_Close(hdl->mic);
    if (hdl->amp) amp_Close(hdl->amp);

    printf("Audio pipeline closed.\n");
    printf("  Frames processed : %lu\n", frame_count);
    printf("  RB underflows    : %lu\n", rb_underflow_count);
    printf("  Preset changes   : %lu\n", preset_changes);

    return STATUS_OK;
}