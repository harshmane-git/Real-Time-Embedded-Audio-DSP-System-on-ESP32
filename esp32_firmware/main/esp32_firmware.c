#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "gain.h"

gain_t gain;

void app_main(void)
{
    gain_init(&gain);

    while (1)
    {
        float buffer[256] = {0};

        gain_apply_update(&gain);
        gain_process_block(&gain, buffer, 256);

        printf("Current Gain: %.2f\n", gain.current_gain);

        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}