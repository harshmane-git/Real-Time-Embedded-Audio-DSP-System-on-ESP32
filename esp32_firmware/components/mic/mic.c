#include "mic.h"
#include "ring_buffer.h"
#include "driver/i2s.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <stdint.h>

#define I2S_PORT    I2S_NUM_0
#define SAMPLE_RATE 16000

#define I2S_BCLK    26
#define I2S_LRC     25
#define I2S_DIN     33

void mic_init(void)
{
    i2s_config_t i2s_config = {
        .mode                 = I2S_MODE_MASTER | I2S_MODE_RX,
        .sample_rate          = SAMPLE_RATE,
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_I2S,
        .intr_alloc_flags     = 0,
        .dma_buf_count        = 8,
        .dma_buf_len          = 256,
        .use_apll             = false,
        .tx_desc_auto_clear   = false,
        .fixed_mclk           = 0
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num   = I2S_BCLK,
        .ws_io_num    = I2S_LRC,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num  = I2S_DIN
    };

    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);
    i2s_zero_dma_buffer(I2S_PORT);
}

void mic_task(void *arg)
{
    ring_buffer_t *rb = (ring_buffer_t *)arg;

    int32_t raw_block[RB_SAMPLES_PER_SLOT];
    float   float_block[RB_SAMPLES_PER_SLOT];
    size_t  bytes_read;

    while (1)
    {
        i2s_read(I2S_PORT,
                 raw_block,
                 RB_SAMPLES_PER_SLOT * sizeof(int32_t),
                 &bytes_read,
                 portMAX_DELAY);

        int samples_read = bytes_read / sizeof(int32_t);

        for (int i = 0; i < samples_read; i++)
        {
            float_block[i] = ((float)(raw_block[i] >> 14) / 32768.0f) * 0.3f;
        }

        for (int i = samples_read; i < RB_SAMPLES_PER_SLOT; i++)
        {
            float_block[i] = 0.0f;
        }

        if (!rb_write_block(rb, float_block))
        {
            // overflow → drop block (real-time safe)
        }
    }
}