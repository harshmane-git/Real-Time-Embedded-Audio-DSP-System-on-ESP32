#include "audio_pipeline.h"
#include "driver/gpio.h"

#define SWITCH1 32
#define SWITCH2 33

extern volatile int preset_request;

void gpio_init_custom()
{
    gpio_set_direction(SWITCH1, GPIO_MODE_INPUT);
    gpio_set_direction(SWITCH2, GPIO_MODE_INPUT);
}

void app_main(void)
{
    audio_hdl hdl;

    audio_Open(&hdl);
    audio_Initialize(&hdl);
    gpio_init_custom();

    while (1)
    {
        if (gpio_get_level(SWITCH1)) preset_request = 1;
        if (gpio_get_level(SWITCH2)) preset_request = 2;

        audio_Process(&hdl);
    }
}