#include "amp.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define I2S_PORT I2S_NUM_1
#define BLOCK_SIZE 256

// ✅ FIX: use 16-bit buffer
static int16_t i2s_tx_buffer[BLOCK_SIZE];

STATUS amp_Open(uint32_t *size)
{
    *size = sizeof(amp_hdl);
    return STATUS_OK;
}

STATUS amp_Initialize(amp_hdl *hdl, const amp_config *cfg)
{
    i2s_chan_handle_t tx_handle;

    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_PORT, I2S_ROLE_MASTER);
    i2s_new_channel(&chan_cfg, &tx_handle, NULL);

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(16000),

        // ✅ FIX: 16-bit + stereo
        .slot_cfg = I2S_STD_MSB_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_16BIT,
            I2S_SLOT_MODE_STEREO
        ),

        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = 27,
            .ws = 14,
            .dout = 22,
            .din = I2S_GPIO_UNUSED
        }
    };

    i2s_channel_init_std_mode(tx_handle, &std_cfg);
    i2s_channel_enable(tx_handle);

    hdl->tx_handle = tx_handle;

    return STATUS_OK;
}

STATUS amp_Process(amp_hdl *hdl, const float *input, uint32_t samples)
{
    if (samples > BLOCK_SIZE) samples = BLOCK_SIZE;

    for (uint32_t i = 0; i < samples; i++)
    {
        float sample = input[i];

        // ✅ soft clipping (reduces crackle)
        if (sample > 0.8f) sample = 0.8f;
        if (sample < -0.8f) sample = -0.8f;

        // ✅ convert float → int16
        i2s_tx_buffer[i] = (int16_t)(sample * 32767);
    }

    size_t bytes_written;
    i2s_channel_write(
        hdl->tx_handle,
        i2s_tx_buffer,
        samples * sizeof(int16_t),
        &bytes_written,
        portMAX_DELAY
    );

    return STATUS_OK;
}

STATUS amp_Close(amp_hdl *hdl)
{
    return STATUS_OK;
}