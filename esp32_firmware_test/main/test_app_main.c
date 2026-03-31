#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "test_ring_buffer.h"

void app_main(void)
{
    printf("Starting Ring Buffer Test Suite...\n");

    // Level 1 - static unit tests
    test_rb_run();

    // Small gap between levels
    vTaskDelay(pdMS_TO_TICKS(500));

    // Level 2 - FreeRTOS two-task test
    test_rb_rtos();

    while(1)
        vTaskDelay(pdMS_TO_TICKS(1000));
}