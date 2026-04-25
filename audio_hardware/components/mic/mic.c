#include "mic.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <stdint.h>

#define I2S_PORT I2S_NUM_0
#define SAMPLE_RATE 16000
#define BLOCK_SIZE 256

static int32_t i2s_buffer[BLOCK_SIZE];

STATUS mic_Open(uint32_t *size)
{
    *size = sizeof(mic_hdl);
    return STATUS_OK;
}

STATUS mic_Initialize(mic_hdl *hdl, const mic_config *cfg)
{
    i2s_chan_handle_t rx_handle;

    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_PORT, I2S_ROLE_MASTER);
    i2s_new_channel(&chan_cfg, NULL, &rx_handle);

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),

        // keep 32-bit (mic uses it)
        .slot_cfg = I2S_STD_MSB_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_32BIT,
            I2S_SLOT_MODE_MONO
        ),

        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = 26,
            .ws = 25,
            .dout = I2S_GPIO_UNUSED,
            .din = 34
        }
    };

    i2s_channel_init_std_mode(rx_handle, &std_cfg);
    i2s_channel_enable(rx_handle);

    hdl->rx_handle = rx_handle;

    return STATUS_OK;
}

STATUS mic_Process(mic_hdl *hdl, float *output, uint32_t samples)
{
    size_t bytes_read;

    i2s_channel_read(
        hdl->rx_handle,
        i2s_buffer,
        samples * sizeof(int32_t),
        &bytes_read,
        portMAX_DELAY
    );

    for (uint32_t i = 0; i < samples; i++)
    {
        // ✅ FIX: remove garbage lower bits
        int32_t raw = i2s_buffer[i] >> 8;

        // ✅ normalize properly (24-bit mic data)
        output[i] = (float)raw / 8388608.0f;
    }

    return STATUS_OK;
}

STATUS mic_Close(mic_hdl *hdl)
{
    return STATUS_OK;
}