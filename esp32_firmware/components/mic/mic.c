#include "mic.h"
#include "wav_data.h"
#include "ring_buffer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

extern ring_buffer_t rb;

static int index = 0;

void mic_init(void)
{
    index = 0;
}

void mic_task(void *arg)
{
    int32_t block[RB_SAMPLES_PER_SLOT];

    while (1)
    {
        for (int i = 0; i < RB_SAMPLES_PER_SLOT; i++)
        {
            block[i] = (int32_t)(audio_data[index] * 2147483647); // float → int32

            index = (index + 1) % audio_data_size;
        }

        I2S_write(&rb, block);

        vTaskDelay(1);
    }
}