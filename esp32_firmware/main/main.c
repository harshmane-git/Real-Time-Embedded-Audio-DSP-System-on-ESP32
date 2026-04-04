#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "ring_buffer.h"
#include "audio_pipeline.h"
#include "scheduler.h"
#include "mic.h"

// Global ring buffer
ring_buffer_t rb;

void app_main(void)
{
    printf("=== NEW DSP PIPELINE STARTED ===\n");

    rb_init(&rb);

    // Init hardware (clean abstraction)
    audio_init();

    // Mic task (producer)
    if (xTaskCreatePinnedToCore(mic_task, "mic_task", 4096, &rb, 6, NULL, 0) != pdPASS)
    {
        printf("Failed to create mic_task\n");
    }

    // Scheduler task (consumer + DSP)
    if (xTaskCreatePinnedToCore(scheduler_task, "scheduler_task", 4096, &rb, 5, NULL, 1) != pdPASS)
    {
        printf("Failed to create scheduler_task\n");
    }
}