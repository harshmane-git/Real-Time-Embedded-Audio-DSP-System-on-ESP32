#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "test_ring_buffer.h"

void app_main(void)
{
    printf("Starting Ring Buffer Test Suite...\n");
    test_rb_run();

    while(1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}