#include "audio_pipeline.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

void app_main(void)
{
    audio_init();

    while(1)
    {
        audio_process();

        vTaskDelay(pdMS_TO_TICKS(16));
    }
}