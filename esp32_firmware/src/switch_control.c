#include "switch_control.h"
#include "audio_config.h"
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define DEBOUNCE_INTERVAL pdMS_TO_TICKS(200)

static pipeline_hdl_t s_pipeline_hdl = NULL;
static int prev_sw1 = 1, prev_sw2 = 1, prev_sw3 = 1;
static TickType_t last_sw1 = 0, last_sw2 = 0, last_sw3 = 0;

void switch_control_init(pipeline_hdl_t hdl)
{
    s_pipeline_hdl = hdl;

    gpio_config_t io_cfg = {
        .pin_bit_mask = (1ULL << GPIO_SWITCH1) |
                        (1ULL << GPIO_SWITCH2) |
                        (1ULL << GPIO_SWITCH3),
        .mode         = GPIO_MODE_INPUT,
        .pull_up_en   = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_cfg);
}

void switch_control_poll(void)
{
    if (!s_pipeline_hdl) return;

    TickType_t now = xTaskGetTickCount();
    int s1 = gpio_get_level(GPIO_SWITCH1);
    int s2 = gpio_get_level(GPIO_SWITCH2);
    int s3 = gpio_get_level(GPIO_SWITCH3);

    if (prev_sw1 == 1 && s1 == 0 && (now - last_sw1) > DEBOUNCE_INTERVAL) {
        last_sw1 = now;
        int next_preset = (pipeline_get_eq_preset(s_pipeline_hdl) + 1) % EQ_NUM_PRESETS;
        printf("[SW1] EQ switch pressed. Cycling preset to -> %d\n", next_preset);
        pipeline_set_eq(s_pipeline_hdl, next_preset);
    }

    if (prev_sw2 == 1 && s2 == 0 && (now - last_sw2) > DEBOUNCE_INTERVAL) {
        last_sw2 = now;
        int next_gain = (pipeline_get_gain_step(s_pipeline_hdl) + 1) % GAIN_STEPS;
        pipeline_set_gain(s_pipeline_hdl, next_gain);
    }

    if (prev_sw3 == 1 && s3 == 0 && (now - last_sw3) > DEBOUNCE_INTERVAL) {
        last_sw3 = now;
        pipeline_toggle_delay(s_pipeline_hdl);
    }

    prev_sw1 = s1;
    prev_sw2 = s2;
    prev_sw3 = s3;
}