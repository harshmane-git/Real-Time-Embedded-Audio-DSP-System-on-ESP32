#include "audio_pipeline.h"
#include "audio_config.h"
#include "low_pass.h"
#include "band_pass.h"
#include "high_pass.h"
#include "equalizer.h"
#include "esp_timer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// Static working buffers — in BSS, never on stack
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
// Hardcoded biquad filter handles (always ON)
// ============================================================================
static low_pass_hdl_t   bq_lp;
static band_pass_hdl_t  bq_bp;
static high_pass_hdl_t  bq_hp;

// ============================================================================
// EQ
// ============================================================================
static equalizer_hdl_t  eq_hdl;
static int              eq_preset  = 0;
static int              eq_enabled = 0;
static int eq_preset_pending = -1;  // -1 = no change pending

// Explicit coefficient arrays — avoids macro initializer alignment issues
// Format: {b0, b1, b2, a1, a2}
static const float eq_lp_s1[EQ_NUM_PRESETS][5] = {
    { +0.75705202f, +1.51410405f, +0.75705202f, +1.45419681f, +0.57401129f }, // P0 flat
    { +0.00554263f, +0.01108527f, +0.00554263f, -1.77860502f, +0.80077556f }, // P1 bass
    { +0.18668319f, +0.37336638f, +0.18668319f, -0.46291040f, +0.20964317f }, // P2 mid
    { +0.75705202f, +1.51410405f, +0.75705202f, +1.45419681f, +0.57401129f }, // P3 treble
};
static const float eq_lp_s2[EQ_NUM_PRESETS][5] = {
    { +0.75705202f, +1.51410405f, +0.75705202f, +1.45419681f, +0.57401129f },
    { +0.00554263f, +0.01108527f, +0.00554263f, -1.77860502f, +0.80077556f },
    { +0.18668319f, +0.37336638f, +0.18668319f, -0.46291040f, +0.20964317f },
    { +0.75705202f, +1.51410405f, +0.75705202f, +1.45419681f, +0.57401129f },
};
static const float eq_bp_s1[EQ_NUM_PRESETS][5] = {
    { +0.15026695f, +0.00000000f, -0.15026695f, -1.45110604f, +0.57066586f }, // P0 flat
    { +0.03822972f, +0.00000000f, -0.03822972f, -1.94301883f, +0.94902703f }, // P1 bass
    { +0.23438022f, +0.00000000f, -0.23438022f, -1.40309918f, +0.68749303f }, // P2 mid
    { +0.35317580f, +0.00000000f, -0.35317580f, +0.58516083f, +0.52909893f }, // P3 treble
};
static const float eq_bp_s2[EQ_NUM_PRESETS][5] = {
    { +0.15026695f, +0.00000000f, -0.15026695f, -1.45110604f, +0.57066586f },
    { +0.03822972f, +0.00000000f, -0.03822972f, -1.94301883f, +0.94902703f },
    { +0.23438022f, +0.00000000f, -0.23438022f, -1.40309918f, +0.68749303f },
    { +0.35317580f, +0.00000000f, -0.35317580f, +0.58516083f, +0.52909893f },
};
static const float eq_hp_s1[EQ_NUM_PRESETS][5] = {
    { +0.97802727f, -1.95605454f, +0.97802727f, -1.95557182f, +0.95653726f }, // P0 flat
    { +0.98347477f, -1.96694954f, +0.98347477f, -1.96667652f, +0.96722256f }, // P1 bass
    { +0.92005550f, -1.84011099f, +0.92005550f, -1.83371141f, +0.84651057f }, // P2 mid
    { +0.56900695f, -1.13801389f, +0.56900695f, -0.94276158f, +0.33326621f }, // P3 treble
};
static const float eq_hp_s2[EQ_NUM_PRESETS][5] = {
    { +0.97802727f, -1.95605454f, +0.97802727f, -1.95557182f, +0.95653726f },
    { +0.98347477f, -1.96694954f, +0.98347477f, -1.96667652f, +0.96722256f },
    { +0.92005550f, -1.84011099f, +0.92005550f, -1.83371141f, +0.84651057f },
    { +0.56900695f, -1.13801389f, +0.56900695f, -0.94276158f, +0.33326621f },
};

// ============================================================================
// Gain
// ============================================================================
static const float gain_array[GAIN_STEPS] = GAIN_ARRAY;
static int         gain_index   = 1;    // index 1 = 1.0f unity
static int         gain_enabled = 0;

// ============================================================================
// Switch state
// ============================================================================
static int sw1_prev = 1;
static int sw2_prev = 1;
static int sw3_prev = 1;


// ============================================================================
// eq_load_preset — all locals static to avoid stack pressure
// ============================================================================
static void eq_load_preset(int idx)
{
    if (idx < 0 || idx >= EQ_NUM_PRESETS) return;

    printf("[EQ] loading preset %d...\n", idx);

    printf("[EQ] step 1: copy LP coeffs\n");
    static low_pass_config_t lp_cfg;
    memcpy(lp_cfg.s1, eq_lp_s1[idx], 5 * sizeof(float));
    memcpy(lp_cfg.s2, eq_lp_s2[idx], 5 * sizeof(float));

    printf("[EQ] step 2: init LP\n");
    low_pass_init(&eq_hdl.low_hdl, &lp_cfg);

    printf("[EQ] step 3: copy BP coeffs\n");
    static band_pass_config_t bp_cfg;
    memcpy(bp_cfg.s1, eq_bp_s1[idx], 5 * sizeof(float));
    memcpy(bp_cfg.s2, eq_bp_s2[idx], 5 * sizeof(float));

    printf("[EQ] step 4: init BP\n");
    band_pass_init(&eq_hdl.mid_hdl, &bp_cfg);

    printf("[EQ] step 5: copy HP coeffs\n");
    static high_pass_config_t hp_cfg;
    memcpy(hp_cfg.s1, eq_hp_s1[idx], 5 * sizeof(float));
    memcpy(hp_cfg.s2, eq_hp_s2[idx], 5 * sizeof(float));

    printf("[EQ] step 6: init HP\n");
    high_pass_init(&eq_hdl.high_hdl, &hp_cfg);

    printf("[EQ] Preset %d loaded OK\n", idx);
}

// ============================================================================
// handle_switches — debounced edge detection
// ============================================================================
static void handle_switches(void)
{
    static int64_t last_sw1 = 0;
    static int64_t last_sw2 = 0;
    static int64_t last_sw3 = 0;
    const  int64_t DEBOUNCE_US = 200000;

    int s1 = sw1_level;
    int s2 = sw2_level;
    int s3 = sw3_level;

    int64_t now = esp_timer_get_time();

    // SW1 — just set pending flag, do NOT call eq_load_preset here
    if (sw1_prev == 1 && s1 == 0 && (now - last_sw1) > DEBOUNCE_US)
    {
        last_sw1          = now;
        eq_preset         = (eq_preset + 1) % EQ_NUM_PRESETS;
        eq_preset_pending = eq_preset;   // signal main loop
    }

    // SW2 — gain cycle (safe, no heavy ops)
    if (sw2_prev == 1 && s2 == 0 && (now - last_sw2) > DEBOUNCE_US)
    {
        last_sw2     = now;
        gain_index   = (gain_index + 1) % GAIN_STEPS;
        gain_enabled = 1;
        printf("[SW2] Gain → %.2f\n", gain_array[gain_index]);
    }

    // SW3 — delay passthrough
    if (sw3_prev == 1 && s3 == 0 && (now - last_sw3) > DEBOUNCE_US)
    {
        last_sw3 = now;
        printf("[SW3] Delay (passthrough)\n");
    }

    sw1_prev = s1;
    sw2_prev = s2;
    sw3_prev = s3;
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

    // Mic
    mic_config mic_cfg = { .sample_rate = AUDIO_SAMPLE_RATE };
    if (mic_Initialize(hdl->mic, &mic_cfg) != STATUS_OK)
    {
        printf("[ERROR] mic_Initialize failed\n");
        return STATUS_NOT_OK;
    }

    // Amp
    amp_config amp_cfg = { .sample_rate = AUDIO_SAMPLE_RATE };
    if (amp_Initialize(hdl->amp, &amp_cfg) != STATUS_OK)
    {
        printf("[ERROR] amp_Initialize failed\n");
        return STATUS_NOT_OK;
    }

    // Ring buffer
    uint32_t rb_size = 0;
    rb_Open(AUDIO_SAMPLE_RATE, &rb_size);
    rb_config rb_cfg = { .size = rb_size, .sample_rate = AUDIO_SAMPLE_RATE };
    if (rb_Initialize(hdl->rb, &rb_cfg) != STATUS_OK)
    {
        printf("[ERROR] rb_Initialize failed\n");
        return STATUS_NOT_OK;
    }

    // Hardcoded biquad filters
    static low_pass_config_t  lp_cfg;
    static band_pass_config_t bp_cfg;
    static high_pass_config_t hp_cfg;

    float lp_s1[] = BIQUAD_LP_S1;
    float lp_s2[] = BIQUAD_LP_S2;
    float bp_s1[] = BIQUAD_BP_S1;
    float bp_s2[] = BIQUAD_BP_S2;
    float hp_s1[] = BIQUAD_HP_S1;
    float hp_s2[] = BIQUAD_HP_S2;

    memcpy(lp_cfg.s1, lp_s1, sizeof(lp_cfg.s1));
    memcpy(lp_cfg.s2, lp_s2, sizeof(lp_cfg.s2));
    memcpy(bp_cfg.s1, bp_s1, sizeof(bp_cfg.s1));
    memcpy(bp_cfg.s2, bp_s2, sizeof(bp_cfg.s2));
    memcpy(hp_cfg.s1, hp_s1, sizeof(hp_cfg.s1));
    memcpy(hp_cfg.s2, hp_s2, sizeof(hp_cfg.s2));

    low_pass_init (&bq_lp, &lp_cfg);
    band_pass_init(&bq_bp, &bp_cfg);
    high_pass_init(&bq_hp, &hp_cfg);

    // EQ — load preset 0 at startup
    eq_load_preset(0);

    printf("[INIT] Pipeline ready\n");
    printf("  SW1(GPIO32)=EQ  SW2(GPIO33)=Gain  SW3(GPIO35)=Delay\n");
    printf("  All effects OFF at start — unity gain, flat EQ\n");

    return STATUS_OK;
}

// ============================================================================
// audio_Process
// ============================================================================
STATUS audio_Process(audio_hdl *hdl)
{
    if (!hdl) return STATUS_NOT_OK;

    handle_switches();

    // 1. Mic
    mic_Process(hdl->mic, input_block, AUDIO_BLOCK_SIZE);

    // 2. Ring buffer
    if (rb_Process(hdl->rb, input_block, rb_out, AUDIO_BLOCK_SIZE) != STATUS_OK)
        memset(rb_out, 0, sizeof(rb_out));

    // 3. Hardcoded LP+BP+HP (always ON)
    low_pass_process (&bq_lp, rb_out, lp_out, AUDIO_BLOCK_SIZE);
    band_pass_process(&bq_bp, rb_out, bp_out, AUDIO_BLOCK_SIZE);
    high_pass_process(&bq_hp, rb_out, hp_out, AUDIO_BLOCK_SIZE);

    for (uint32_t i = 0; i < AUDIO_BLOCK_SIZE; i++)
        biquad_sum[i] = (lp_out[i] + bp_out[i] + hp_out[i]) * (1.0f / 3.0f);

    // 4. EQ (SW1)
    const float *dsp_out = biquad_sum;

    if (eq_enabled)
    {
        equalizer_process(&eq_hdl,
                          biquad_sum,
                          eq_low, eq_mid, eq_high,
                          AUDIO_BLOCK_SIZE);

        for (uint32_t i = 0; i < AUDIO_BLOCK_SIZE; i++)
            final_out[i] = (eq_low[i] + eq_mid[i] + eq_high[i]) * (1.0f / 3.0f);

        dsp_out = final_out;
    }

    // 5. Gain (SW2)
    float gain = gain_enabled ? gain_array[gain_index] : 1.0f;

    for (uint32_t i = 0; i < AUDIO_BLOCK_SIZE; i++)
        final_out[i] = dsp_out[i] * gain;

    // 6. Delay (SW3) — passthrough TODO

    // 7. Amp
    amp_Process(hdl->amp, final_out, AUDIO_BLOCK_SIZE);

    return STATUS_OK;
}

// ============================================================================
// audio_Close
// ============================================================================
STATUS audio_Close(audio_hdl *hdl)
{
    if (!hdl) return STATUS_NOT_OK;

    mic_Close(hdl->mic);
    amp_Close(hdl->amp);
    rb_Close(hdl->rb);

    free(hdl->mic);
    free(hdl->amp);
    free(hdl->rb);

    return STATUS_OK;
}



void audio_apply_pending(void)
{
    if (eq_preset_pending >= 0)
    {
        int idx           = eq_preset_pending;
        eq_preset_pending = -1;
        eq_load_preset(idx);
        eq_enabled = 1;
        printf("[SW1] EQ preset → %d\n", idx);
    }
}