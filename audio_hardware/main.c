#include "audio_pipeline.h"
#include "driver/gpio.h"
#include <stdio.h>

#define SWITCH1 32
#define SWITCH2 34

extern volatile int preset_request;

void gpio_init_custom()
{
    gpio_set_direction(SWITCH1, GPIO_MODE_INPUT);
    gpio_set_direction(SWITCH2, GPIO_MODE_INPUT);
}

void app_main(void)
{
    audio_hdl hdl;

    // 🔹 Open
    if (audio_Open(&hdl) != STATUS_OK)
    {
        printf("audio_Open failed\n");
        return;
    }

    // 🔹 Initialize
    if (audio_Initialize(&hdl) != STATUS_OK)
    {
        printf("audio_Initialize failed\n");
        audio_Close(&hdl);
        return;
    }

    gpio_init_custom();

    // 🔹 Edge detection for switches (prevents continuous triggering)
    int last1 = 0, last2 = 0;

    while (1)
    {
        int s1 = gpio_get_level(SWITCH1);
        int s2 = gpio_get_level(SWITCH2);

        if (s1 && !last1)
        {
            preset_request = 1;
        }

        if (s2 && !last2)
        {
            preset_request = 2;
        }

        last1 = s1;
        last2 = s2;

        audio_Process(&hdl);
    }
}