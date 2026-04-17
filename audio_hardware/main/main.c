#include "audio_pipeline.h"
#include "audio_config.h"
#include "driver/gpio.h"
#include <stdio.h>

extern volatile int preset_request;

void gpio_init_custom()
{
    gpio_set_direction(GPIO_SWITCH1, GPIO_MODE_INPUT);
    gpio_set_direction(GPIO_SWITCH2, GPIO_MODE_INPUT);
}

void app_main(void)
{
    mic_hdl mic;
    mic_config cfg = {AUDIO_SAMPLE_RATE};
    float block[AUDIO_BLOCK_SIZE];

    mic_Initialize(&mic, &cfg);

    printf("Mic test started\n");

    for (int frame = 0; frame < 100; frame++)
    {
        mic_Process(&mic, block, AUDIO_BLOCK_SIZE);

        // Find peak in this block
        float peak = 0.0f;
        for (int i = 0; i < AUDIO_BLOCK_SIZE; i++)
        {
            float abs_val = block[i] < 0 ? -block[i] : block[i];
            if (abs_val > peak) peak = abs_val;
        }

        printf("Frame %3d | Peak: %.6f\n", frame, peak);
    }

    mic_Close(&mic);
}