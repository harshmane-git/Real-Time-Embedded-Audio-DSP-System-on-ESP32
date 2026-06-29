#include "pipeline.h"
#include "switch_control.h"
#include <stdio.h>

void app_main(void)
{
    pipeline_hdl_t pipeline = NULL;

    if (pipeline_open(&pipeline) != STATUS_OK) {
        printf("[FATAL ERROR] Failed to bootstrap opaque tracking context layout!\n");
        return;
    }

    if (pipeline_init(pipeline) != STATUS_OK) {
        printf("[FATAL ERROR] Audio Hardware Layer Bring-up or Driver Latch Failure!\n");
        pipeline_close(pipeline);
        return;
    }

    switch_control_init(pipeline);
    printf("[SYSTEM] Real-Time Audio Encapsulation Engine Active and Running.\n");

    while (1) {
        switch_control_poll();
        
        if (pipeline_process(pipeline) != STATUS_OK) {
            printf("[RUNTIME STREAM FAULT] Real-time audio stream deadline missed or I2S Sync Broken!\n");
            break; 
        }
    }

    pipeline_close(pipeline);
}