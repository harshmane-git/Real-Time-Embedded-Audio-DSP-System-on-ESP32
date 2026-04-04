#include "mic.h"
#include "driver/i2s.h"
#include <stdint.h>

#define I2S_PORT I2S_NUM_0

#define SAMPLE_RATE 16000
#define BLOCK_SIZE 256

#define I2S_BCLK 26
#define I2S_LRC  25
#define I2S_DIN  33

static int32_t i2s_read_buffer[BLOCK_SIZE];

void mic_init(void)
{
    i2s_config_t i2s_config = {
        .mode = I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_TX,
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_I2S,
        .intr_alloc_flags = 0,
        .dma_buf_count = 8,
        .dma_buf_len = 256,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_BCLK,
        .ws_io_num = I2S_LRC,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_DIN
    };

    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);
    i2s_zero_dma_buffer(I2S_PORT);
}

void mic_process(float *buffer, int size)
{
    size_t bytes_read;

    i2s_read(I2S_PORT, i2s_read_buffer, size * sizeof(int32_t), &bytes_read, portMAX_DELAY);

    int samples = bytes_read / sizeof(int32_t);

    for(int i = 0; i < samples; i++)
    {
        // Convert 32-bit mic data → float [-1,1]
        buffer[i] = ((float)(i2s_read_buffer[i] >> 14) / 32768.0f)*0.3f ;
    }
}

void mic_close(void)
{
    i2s_driver_uninstall(I2S_PORT);
}