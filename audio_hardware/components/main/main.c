#include "audio_pipeline.h"

void app_main(void)
{
    audio_hdl hdl;

    audio_Open(&hdl);
    audio_Initialize(&hdl);

    while (1)
    {
        audio_Process(&hdl);
    }
}