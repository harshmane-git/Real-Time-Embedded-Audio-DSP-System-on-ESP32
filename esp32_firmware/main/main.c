#include "audio_pipeline.h"
#include <stdio.h>

void app_main(void)
{
    audio_hdl hdl;

    if (audio_Open(&hdl) != STATUS_OK)
    {
        printf("[ERROR] audio_Open failed\n");
        return;
    }

    if (audio_Initialize(&hdl) != STATUS_OK)
    {
        printf("[ERROR] audio_Initialize failed\n");
        return;
    }

    printf("[MAIN] Stage 2 running: mic → ring buffer → amp\n");

    while (1)
    {
        audio_Process(&hdl);
    }
}