#include "audio_pipeline.h"
#include "audio_config.h"
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <stdio.h>

// Defined in audio_pipeline.c — main writes, pipeline reads
volatile int sw1_level = 1;
volatile int sw2_level = 1;
volatile int sw3_level = 1;

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
    gpio_init_switches();

    audio_hdl hdl;

    if (audio_Open(&hdl) != STATUS_OK)
    {
        printf("[ERROR] audio_Open failed\n");
        return;
    }

    if (audio_Initialize(&hdl) != STATUS_OK)
    {
        printf("[ERROR] audio_Initialize failed\n");
        return;
    }

    printf("[MAIN] Running. SW1=EQ  SW2=Gain  SW3=Delay\n");

    while (1)
    {
        // Read GPIO levels — active-low (0 = pressed)
        sw1_level = gpio_get_level(GPIO_SWITCH1);
        sw2_level = gpio_get_level(GPIO_SWITCH2);
        sw3_level = gpio_get_level(GPIO_SWITCH3);

        // Apply any pending preset change BEFORE next audio block
        audio_apply_pending();
        audio_Process(&hdl);
    }
}