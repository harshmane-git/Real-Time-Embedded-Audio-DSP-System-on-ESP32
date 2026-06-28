#include "audio_pipeline.h"
#include "audio_config.h"
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_err.h"
#include <stdio.h>

// Shared with audio_pipeline.c
extern int eq_preset;
extern int eq_preset_pending;

// Declared here, used by audio_pipeline.c via extern in audio_pipeline.h
volatile int sw1_level = 1;
volatile int sw2_level = 1;
volatile int sw3_level = 1;

// Functions defined in audio_pipeline.c
extern void audio_set_gain(int index);
extern void audio_toggle_delay(void);

static void gpio_init_switches(void)
{
    gpio_config_t io_cfg = {
        .pin_bit_mask = (1ULL << GPIO_SWITCH1) |
                        (1ULL << GPIO_SWITCH2) |
                        (1ULL << GPIO_SWITCH3),
        .mode         = GPIO_MODE_INPUT,
        .pull_up_en   = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_cfg);
}

void app_main(void)
{
    // Initialize the input switch configuration
    gpio_init_switches();

    audio_hdl hdl;

    // Open structural resource descriptors
    if (audio_Open(&hdl) != STATUS_OK)
    {
        return;
    }

    // Initialize all active hardware drivers and filter blocks
    if (audio_Initialize(&hdl) != STATUS_OK)
    {
        return;
    }

    int prev_sw1 = 1, prev_sw2 = 1, prev_sw3 = 1;
    int gain_step = 1;

    // Edge tracking debounce timers
    TickType_t last_sw1 = 0, last_sw2 = 0, last_sw3 = 0;
    const TickType_t DEBOUNCE = pdMS_TO_TICKS(200);

    while (1)
    {
        TickType_t now = xTaskGetTickCount();

        // Sample current hardware signal levels
        int s1 = gpio_get_level(GPIO_SWITCH1);
        int s2 = gpio_get_level(GPIO_SWITCH2);
        int s3 = gpio_get_level(GPIO_SWITCH3);

        // ====================================================================
        // INTERACTIVE ASYNCHRONOUS EVENT PROCESSING
        // ====================================================================

        // SW1 — Equalizer Preset Cycling (Falling edge + Debounce)
        if (prev_sw1 == 1 && s1 == 0 && (now - last_sw1) > DEBOUNCE)
        {
            last_sw1          = now;
            eq_preset_pending = (eq_preset + 1) % EQ_NUM_PRESETS;
            printf("[SW1] EQ preset -> %d\n", eq_preset_pending);
        }

        // SW2 — Master Gain Regulation Cycling (Falling edge + Debounce)
        if (prev_sw2 == 1 && s2 == 0 && (now - last_sw2) > DEBOUNCE)
        {
            last_sw2  = now;
            gain_step = (gain_step + 1) % GAIN_STEPS;
            audio_set_gain(gain_step);
            printf("[SW2] Gain step -> %d\n", gain_step);
        }

        // SW3 — Feedback Delay Line Activation (Falling edge + Debounce)
        if (prev_sw3 == 1 && s3 == 0 && (now - last_sw3) > DEBOUNCE)
        {
            last_sw3 = now;
            audio_toggle_delay();
            printf("[SW3] Delay toggled\n");
        }

        // Archive state history for next sample comparison
        prev_sw1 = s1;
        prev_sw2 = s2;
        prev_sw3 = s3;

        // Apply pending filter changes safely at block boundaries
        audio_apply_pending();

        // Process a block of data — naturally blocks task execution for 16ms via I2S DMA
        audio_Process(&hdl);
    }
}