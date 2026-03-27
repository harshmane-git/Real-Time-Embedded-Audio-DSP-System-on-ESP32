#include "mic.h"
#include "wav_data.h"

static int index = 0;

void mic_init(void)
{
    index = 0;
}

void mic_read_block(float *buffer, int size)
{
    for(int i = 0; i < size; i++)
    {
        buffer[i] = audio_data[index];

        // Safe circular indexing
        index = (index + 1) % audio_data_size;
    }
}
