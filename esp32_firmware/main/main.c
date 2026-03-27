#include "audio_pipeline.h"

void app_main(void)
{
    audio_init();

    while(1)
    {
        audio_process();
    }
}
