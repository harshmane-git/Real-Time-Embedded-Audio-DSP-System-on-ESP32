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
    audio_hdl hdl;

    if (audio_Open(&hdl) != STATUS_OK)
    {
        printf("audio_Open failed\n");
        return;
    }

    if (audio_Initialize(&hdl) != STATUS_OK)
    {
        printf("audio_Initialize failed\n");
        audio_Close(&hdl);
        return;
    }

    gpio_init_custom();

    int last1 = 0, last2 = 0;

    while (1)
    {
        int s1 = gpio_get_level(GPIO_SWITCH1);
        int s2 = gpio_get_level(GPIO_SWITCH2);

        // Rising edge detection
        if (s1 && !last1) preset_request = 1;
        if (s2 && !last2) preset_request = 2;

        last1 = s1;
        last2 = s2;

        // 🔹 Process audio
        if (audio_Process(&hdl) != STATUS_OK)
        {
            // Optional: error handling (minimal)
        }
    }
}