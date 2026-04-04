#include "amp.h"
#include "driver/i2s.h"
#include <stdint.h>

#define I2S_PORT I2S_NUM_0
#define SAMPLE_RATE 16000
#define BLOCK_SIZE 256

#define I2S_BCLK 26
#define I2S_LRC  25
#define I2S_DOUT 22

void amp_init(void)
{
    i2s_config_t i2s_config = {
        .mode = I2S_MODE_MASTER | I2S_MODE_TX,
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .dma_buf_count = 8,
        .dma_buf_len = 256,
        .use_apll = false,
        .tx_desc_auto_clear = true
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_BCLK,
        .ws_io_num = I2S_LRC,
        .data_out_num = I2S_DOUT,
        .data_in_num = I2S_PIN_NO_CHANGE
    };

    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);
    i2s_zero_dma_buffer(I2S_PORT);
}

void amp_write_block(float *buffer, int size)
{
    int16_t i2s_buffer[size];

    for(int i = 0; i < size; i++)
    {
        float sample = buffer[i];

        //CLAMP (prevents distortion)
        if(sample > 1.0f) sample = 1.0f;
        if(sample < -1.0f) sample = -1.0f;

        // SCALE (reduced to avoid flicker)
        i2s_buffer[i] = (int16_t)(sample * 15000);
    }

    size_t bytes_written;
    i2s_write(I2S_NUM_0, i2s_buffer, size * sizeof(int16_t), &bytes_written, 0);
}