#include "amp.h"
#include "driver/i2s_std.h"

#define I2S_PORT I2S_NUM_1

static int32_t i2s_tx_buffer[256];

STATUS amp_Open(uint32_t *size)
{
    *size = sizeof(amp_hdl);
    return STATUS_OK;
}

STATUS amp_Initialize(amp_hdl *hdl, const amp_config *cfg)
{
    hdl->handle = tx_handle;
    i2s_chan_handle_t tx_handle = (i2s_chan_handle_t)hdl->handle;

    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_PORT, I2S_ROLE_MASTER);
    i2s_new_channel(&chan_cfg, &tx_handle, NULL);

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(16000),
        .slot_cfg = I2S_STD_MSB_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
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

    hdl->dummy = (int)tx_handle;

    return STATUS_OK;
}

STATUS amp_Process(amp_hdl *hdl, const float *input, uint32_t samples)
{
    i2s_chan_handle_t tx_handle = (i2s_chan_handle_t)hdl->dummy;

    for (uint32_t i = 0; i < samples; i++)
    {
        float sample = input[i];

        // 🔥 saturation
        if (sample > 1.0f) sample = 1.0f;                   //corrections
        if (sample < -1.0f) sample = -1.0f;

        i2s_tx_buffer[i] = (int32_t)(sample * 2147483647);
    }

    size_t bytes_written;
    i2s_channel_write(tx_handle,
                  i2s_tx_buffer,
                  samples * sizeof(int32_t),
                  &bytes_written,
                  portMAX_DELAY);

        return STATUS_OK;
}