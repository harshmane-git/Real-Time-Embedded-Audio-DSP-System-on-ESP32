#include "audio_pipeline.h"
#include "mic.h"
#include "amp.h"

void audio_init(void)
{
    mic_init();
    amp_init();
    // rb_init() intentionally not here — main.c owns rb
}