#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "ring_buffer.h"
#include "mic.h"
#include "amp.h"
#include "scheduler.h"

ring_buffer_t rb;

void app_main(void)
{
    printf("=== NEW DSP PIPELINE STARTED ===\n");

    rb_init(&rb);

    mic_init();
    amp_init();

    xTaskCreate(mic_task, "mic_task", 4096, NULL, 5, NULL);
    xTaskCreate(scheduler_task, "scheduler_task", 4096, NULL, 5, NULL);
}