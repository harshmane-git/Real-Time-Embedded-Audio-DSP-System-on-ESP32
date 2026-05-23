#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "audio_pipeline.h"
#include "audio_config.h"
#include "low_pass.h"
#include "band_pass.h"
#include "high_pass.h"
#include "equalizer.h"
#include "gain.h"
#include "delay.h"
#include "esp_timer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// ============================================================================
// Static working buffers — BSS, never on stack
// ============================================================================
static float input_block [AUDIO_BLOCK_SIZE];
static float rb_out      [AUDIO_BLOCK_SIZE];
static float lp_out      [AUDIO_BLOCK_SIZE];
static float bp_out      [AUDIO_BLOCK_SIZE];
static float hp_out      [AUDIO_BLOCK_SIZE];
static float biquad_sum  [AUDIO_BLOCK_SIZE];
static float eq_low      [AUDIO_BLOCK_SIZE];
static float eq_mid      [AUDIO_BLOCK_SIZE];
static float eq_high     [AUDIO_BLOCK_SIZE];
static float final_out   [AUDIO_BLOCK_SIZE];

// ============================================================================
// DSP handles
// ============================================================================
static low_pass_hdl_t   bq_lp;
static band_pass_hdl_t  bq_bp;
static high_pass_hdl_t  bq_hp;
static equalizer_hdl_t  eq_hdl;
static eq_gain_hdl_t    eq_gain_hdl;
static gain_hdl_t       master_gain_hdl;
static delay_hdl_t      delay_hdl;

// ============================================================================
// All config structs at file scope — never on stack
// ============================================================================
static low_pass_config_t   eq_lp_cfg;
static band_pass_config_t  eq_bp_cfg;
static high_pass_config_t  eq_hp_cfg;
static low_pass_config_t   bq_lp_cfg;
static band_pass_config_t  bq_bp_cfg;
static high_pass_config_t  bq_hp_cfg;
static gain_config_t       master_g_cfg;
static delay_config_t      delay_d_cfg;

// ============================================================================
// State — shared with main.c via extern
// ============================================================================
int eq_preset         = 0;
int eq_preset_pending = -1;

static int eq_enabled    = 0;
static int gain_enabled  = 0;
static int delay_enabled = 0;
static int gain_index    = 1;   // index 1 = 0dB unity

// ============================================================================
// EQ preset coefficient tables — reference from C_implementation main_test.c
// Format: {b0, b1, b2, a1, a2}
// ============================================================================
static const float eq_lp_s1[EQ_NUM_PRESETS][5] = {
    { 0.00312629f,  0.00625258f,  0.00312629f, -1.79158896f,  0.80409412f },
    { 0.00554263f,  0.01108527f,  0.00554263f, -1.77860502f,  0.80077556f },
    { 0.18668319f,  0.37336638f,  0.18668319f, -0.46291040f,  0.20964317f },
    { 0.75705202f,  1.51410405f,  0.75705202f,  1.45419681f,  0.57401129f },
};
static const float eq_lp_s2[EQ_NUM_PRESETS][5] = {
    { 0.00331660f,  0.00663319f,  0.00331660f, -1.90064888f,  0.91391527f },
    { 0.00554263f,  0.01108527f,  0.00554263f, -1.77860502f,  0.80077556f },
    { 0.18668319f,  0.37336638f,  0.18668319f, -0.46291040f,  0.20964317f },
    { 0.75705202f,  1.51410405f,  0.75705202f,  1.45419681f,  0.57401129f },
};
static const float eq_bp_s1[EQ_NUM_PRESETS][5] = {
    { 0.12491506f,  0.00000000f, -0.12491506f, -1.66451047f,  0.75016988f },
    { 0.03822972f,  0.00000000f, -0.03822972f, -1.94301883f,  0.94902703f },
    { 0.23438022f,  0.00000000f, -0.23438022f, -1.40309918f,  0.68749303f },
    { 0.35317580f,  0.00000000f, -0.35317580f,  0.58516083f,  0.52909893f },
};
static const float eq_bp_s2[EQ_NUM_PRESETS][5] = {
    { 0.05582542f,  0.00000000f, -0.05582542f, -1.79592677f,  0.88834915f },
    { 0.03822972f,  0.00000000f, -0.03822972f, -1.94301883f,  0.94902703f },
    { 0.23438022f,  0.00000000f, -0.23438022f, -1.40309918f,  0.68749303f },
    { 0.35317580f,  0.00000000f, -0.35317580f,  0.58516083f,  0.52909893f },
};
static const float eq_hp_s1[EQ_NUM_PRESETS][5] = {
    { 0.51627979f, -1.03255959f,  0.51627979f, -0.85540037f,  0.20971880f },
    { 0.98347477f, -1.96694954f,  0.98347477f, -1.96667652f,  0.96722256f },
    { 0.92005550f, -1.84011099f,  0.92005550f, -1.83371141f,  0.84651057f },
    { 0.56900695f, -1.13801389f,  0.56900695f, -0.94276158f,  0.33326621f },
};
static const float eq_hp_s2[EQ_NUM_PRESETS][5] = {
    { 0.67177700f, -1.34355400f,  0.67177700f, -1.11303657f,  0.57407142f },
    { 0.98347477f, -1.96694954f,  0.98347477f, -1.96667652f,  0.96722256f },
    { 0.92005550f, -1.84011099f,  0.92005550f, -1.83371141f,  0.84651057f },
    { 0.56900695f, -1.13801389f,  0.56900695f, -0.94276158f,  0.33326621f },
};

static const eq_gain_config_t eq_gain_presets[EQ_NUM_PRESETS] = {
    { .low_gain_db = -2.0f, .mid_gain_db =  3.0f, .high_gain_db =  1.0f },
    { .low_gain_db =  6.0f, .mid_gain_db = -3.0f, .high_gain_db = -6.0f },
    { .low_gain_db = -3.0f, .mid_gain_db =  6.0f, .high_gain_db =  1.0f },
    { .low_gain_db = -6.0f, .mid_gain_db = -3.0f, .high_gain_db =  6.0f },
};

// Gain steps in dB matching GAIN_ARRAY {0.5, 1.0, 1.5, 2.0}
static const float gain_db_steps[GAIN_STEPS] = {
    -6.0f, 0.0f, 3.5f, 6.0f
};

// ============================================================================
// eq_load_preset — all 6 coefficient arrays loaded, no printf
// ============================================================================
static void eq_load_preset(int idx)
{
    if (idx < 0 || idx >= EQ_NUM_PRESETS) return;

    memcpy(eq_lp_cfg.s1, eq_lp_s1[idx], 5 * sizeof(float));
    memcpy(eq_lp_cfg.s2, eq_lp_s2[idx], 5 * sizeof(float));
    memcpy(eq_bp_cfg.s1, eq_bp_s1[idx], 5 * sizeof(float));
    memcpy(eq_bp_cfg.s2, eq_bp_s2[idx], 5 * sizeof(float));
    memcpy(eq_hp_cfg.s1, eq_hp_s1[idx], 5 * sizeof(float));
    memcpy(eq_hp_cfg.s2, eq_hp_s2[idx], 5 * sizeof(float));

    low_pass_init (&eq_hdl.low_hdl, &eq_lp_cfg);
    band_pass_init(&eq_hdl.mid_hdl, &eq_bp_cfg);
    high_pass_init(&eq_hdl.high_hdl, &eq_hp_cfg);

    eq_gain_init(&eq_gain_hdl, &eq_gain_presets[idx]);
}

// ============================================================================
// audio_apply_pending — called from main.c BEFORE audio_Process
// Safe to do heavy work here — outside I2S timing window
// ============================================================================
void audio_apply_pending(void)
{
    // TEST: do nothing — check if hang is in this function
    //eq_preset_pending = -1;

    if (eq_preset_pending >= 0)
    {
        int idx           = eq_preset_pending;
        eq_preset_pending = -1;
        eq_preset         = idx;
        eq_load_preset(idx);
        eq_enabled        = 1;
        printf("[EQ] Preset changed to %d\n", idx);
        taskYIELD();
    }
}

// ============================================================================
// audio_Open
// ============================================================================
STATUS audio_Open(audio_hdl *hdl)
{
    if (!hdl) return STATUS_NOT_OK;

    hdl->mic = malloc(sizeof(mic_hdl));
    hdl->amp = malloc(sizeof(amp_hdl));
    hdl->rb  = malloc(sizeof(rb_hdl));

    if (!hdl->mic || !hdl->amp || !hdl->rb)
    {
        printf("[ERROR] audio_Open: malloc failed\n");
        return STATUS_NOT_OK;
    }
    return STATUS_OK;
}

// ============================================================================
// audio_Initialize
// ============================================================================
STATUS audio_Initialize(audio_hdl *hdl)
{
    if (!hdl) return STATUS_NOT_OK;

    mic_config mic_cfg = { .sample_rate = AUDIO_SAMPLE_RATE };
    if (mic_Initialize(hdl->mic, &mic_cfg) != STATUS_OK)
    {
        printf("[ERROR] mic_Initialize failed\n");
        return STATUS_NOT_OK;
    }

    amp_config amp_cfg = { .sample_rate = AUDIO_SAMPLE_RATE };
    if (amp_Initialize(hdl->amp, &amp_cfg) != STATUS_OK)
    {
        printf("[ERROR] amp_Initialize failed\n");
        return STATUS_NOT_OK;
    }

    uint32_t rb_size = 0;
    rb_Open(AUDIO_SAMPLE_RATE, &rb_size);
    rb_config rb_cfg = { .size = rb_size, .sample_rate = AUDIO_SAMPLE_RATE };
    if (rb_Initialize(hdl->rb, &rb_cfg) != STATUS_OK)
    {
        printf("[ERROR] rb_Initialize failed\n");
        return STATUS_NOT_OK;
    }

    // Hardcoded biquad filters — both stages initialised
    float lp_s1[] = BIQUAD_LP_S1;  float lp_s2[] = BIQUAD_LP_S2;
    float bp_s1[] = BIQUAD_BP_S1;  float bp_s2[] = BIQUAD_BP_S2;
    float hp_s1[] = BIQUAD_HP_S1;  float hp_s2[] = BIQUAD_HP_S2;

    memcpy(bq_lp_cfg.s1, lp_s1, sizeof(bq_lp_cfg.s1));
    memcpy(bq_lp_cfg.s2, lp_s2, sizeof(bq_lp_cfg.s2));
    memcpy(bq_bp_cfg.s1, bp_s1, sizeof(bq_bp_cfg.s1));
    memcpy(bq_bp_cfg.s2, bp_s2, sizeof(bq_bp_cfg.s2));
    memcpy(bq_hp_cfg.s1, hp_s1, sizeof(bq_hp_cfg.s1));
    memcpy(bq_hp_cfg.s2, hp_s2, sizeof(bq_hp_cfg.s2));

    low_pass_init (&bq_lp, &bq_lp_cfg);
    band_pass_init(&bq_bp, &bq_bp_cfg);
    high_pass_init(&bq_hp, &bq_hp_cfg);

    // EQ — preset 0 at startup
    eq_load_preset(0);

    // Master gain — unity (0dB)
    master_g_cfg.gain_db = gain_db_steps[gain_index];
    gain_init(&master_gain_hdl, &master_g_cfg);

    // Delay — 250ms
    delay_d_cfg.delay_seconds = 0.25f;
    if (delay_init(&delay_hdl, &delay_d_cfg) != STATUS_OK)
    {
        printf("[ERROR] delay_init failed\n");
        return STATUS_NOT_OK;
    }

    printf("[INIT] Pipeline ready\n");
    printf("  SW1=EQ  SW2=Gain  SW3=Delay — all OFF at start\n");

    return STATUS_OK;
}

// ============================================================================
// audio_Process — serial chain, no printf
// ============================================================================

STATUS audio_Process(audio_hdl *hdl)
{
    if (!hdl) return STATUS_NOT_OK;

    // 1. Input Stage: Read from microphone
    mic_Process(hdl->mic, input_block, AUDIO_BLOCK_SIZE);
    
    // 2. Input Buffering: Stream through the Ring Buffer
    if (rb_Process(hdl->rb, input_block, rb_out, AUDIO_BLOCK_SIZE) != STATUS_OK)
    {
        memset(rb_out, 0, sizeof(rb_out));
    }

    // 3. Always-On Parallel Biquad Filter Bank
    low_pass_process (&bq_lp, rb_out, lp_out, AUDIO_BLOCK_SIZE);
    band_pass_process(&bq_bp, rb_out, bp_out, AUDIO_BLOCK_SIZE);
    high_pass_process(&bq_hp, rb_out, hp_out, AUDIO_BLOCK_SIZE);

    // 4. Parallel Summing Stage with 1/3 Scaling for Headroom
    for (uint32_t i = 0; i < AUDIO_BLOCK_SIZE; i++)
    {
        biquad_sum[i] = (lp_out[i] + bp_out[i] + hp_out[i]) * 0.3333f;
    }

    // Initialize the serial processing pointer to tap the biquad_sum
    float *dsp_out = biquad_sum;

    //  Note: Equalizer (SW1) is skipped entirely here to bypass the hardware issue

    // 5. Master Gain Stage (SW2) — Serial routing via pointer tracking
    if (gain_enabled)
    {
        gain_process(&master_gain_hdl, dsp_out, final_out, AUDIO_BLOCK_SIZE);
        dsp_out = final_out;
    }

    // 6. Delay Stage (SW3) — Serial routing via pointer tracking
    if (delay_enabled)
    {
        delay_process(&delay_hdl, dsp_out, final_out, AUDIO_BLOCK_SIZE);
        dsp_out = final_out;
    }

    // 7. Output Stage: Send final accumulated pointer to the amplifier
    amp_Process(hdl->amp, dsp_out, AUDIO_BLOCK_SIZE);

    return STATUS_OK;
}

// //Temperory check
// STATUS audio_Process(audio_hdl *hdl)
// {
//     mic_Process(hdl->mic, input_block, AUDIO_BLOCK_SIZE);
//     amp_Process(hdl->amp, input_block, AUDIO_BLOCK_SIZE);
//     return STATUS_OK;
// }
// ============================================================================
// audio_Close
// ============================================================================
STATUS audio_Close(audio_hdl *hdl)
{
    if (!hdl) return STATUS_NOT_OK;

    mic_Close(hdl->mic);
    amp_Close(hdl->amp);
    rb_Close(hdl->rb);
    delay_close(&delay_hdl);
    gain_close(&master_gain_hdl);
    eq_gain_close(&eq_gain_hdl);

    free(hdl->mic);
    free(hdl->amp);
    free(hdl->rb);

    return STATUS_OK;
}

// ============================================================================
// Gain and Delay enable — called from main.c on SW2/SW3 press
// ============================================================================
void audio_set_gain(int index)
{
    gain_index           = index % GAIN_STEPS;
    gain_enabled         = 1;
    master_g_cfg.gain_db = gain_db_steps[gain_index];
    gain_init(&master_gain_hdl, &master_g_cfg);
}

void audio_toggle_delay(void)
{
    delay_enabled = !delay_enabled;
}