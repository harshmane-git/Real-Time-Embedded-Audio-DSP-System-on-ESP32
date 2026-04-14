#include "mic.h"
#include "driver/i2s_std.h"

STATUS mic_Open(uint32_t *size)
{
    *size = sizeof(mic_hdl);
    return STATUS_OK;
}

STATUS mic_Initialize(mic_hdl *hdl, const mic_config *cfg)
{
    // 🔥 I2S CONFIG (RX)
    i2s_chan_handle_t rx_handle;

    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(MIC_I2S_PORT, I2S_ROLE_MASTER);
    i2s_new_channel(&chan_cfg, NULL, &rx_handle);

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(cfg->sample_rate),
        .slot_cfg = I2S_STD_MSB_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = MIC_GPIO_BCLK,
            .ws = MIC_GPIO_WS,
            .dout = I2S_GPIO_UNUSED,
            .din = MIC_GPIO_DIN
        }
    };

    i2s_channel_init_std_mode(rx_handle, &std_cfg);
    i2s_channel_enable(rx_handle);

    hdl->handle = rx_handle;

    return STATUS_OK;
}

STATUS mic_Process(mic_hdl *hdl, float *output, uint32_t samples)
{
    static int32_t i2s_buffer[AUDIO_BLOCK_SIZE];
    size_t bytes_read;
    i2s_chan_handle_t rx_handle = hdl->handle;

    i2s_channel_read(rx_handle, i2s_buffer, samples * sizeof(int32_t), &bytes_read, portMAX_DELAY);

    for (uint32_t i = 0; i < samples; i++)
    {
        // int32 → float32 normalization (no scaling)
        output[i] = (float)i2s_buffer[i] / 2147483648.0f;
    }

    return STATUS_OK;
}

STATUS mic_Close(mic_hdl *hdl)
{
    i2s_chan_handle_t rx_handle = hdl->handle;
    i2s_channel_disable(rx_handle);
    i2s_del_channel(rx_handle);
    return STATUS_OK;
}