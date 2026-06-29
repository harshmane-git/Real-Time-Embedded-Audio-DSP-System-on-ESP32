#include "pipeline.h"
#include "audio_pipeline.h"
#include "audio_config.h"
#include <string.h>

// Explicit global linking declarations managed safely at top of file scope
extern int eq_preset;
extern int eq_preset_pending;
extern int eq_preset; 
extern void audio_set_gain(int index);
extern void audio_toggle_delay(void);
extern void audio_apply_pending(void);

// Concrete layout hidden entirely within the .c scope
struct pipeline_ctx_t {
    audio_hdl audio;
    int       eq_preset;
    int       eq_preset_pending;
    int       gain_step;
};

// Pure static allocation in BSS — Zero malloc, zero runtime fragmentation risks
static struct pipeline_ctx_t s_pipeline;

STATUS pipeline_open(pipeline_hdl_t *out_hdl)
{
    if (!out_hdl) return STATUS_NOT_OK;
    
    // Completely clear tracking layout memory space on bootstrap
    memset(&s_pipeline, 0, sizeof(struct pipeline_ctx_t));
    
    // Set baseline structural configurations to match audio_pipeline baseline defaults
    s_pipeline.eq_preset         = 0;
    s_pipeline.eq_preset_pending = -1; 
    s_pipeline.gain_step         = 1; // Synchronizes explicitly with firmware baseline defaults
    
    *out_hdl = &s_pipeline;
    return STATUS_OK;
}

STATUS pipeline_init(pipeline_hdl_t hdl)
{
    if (!hdl) return STATUS_NOT_OK;
    
    // Unified hardware bring-up: safely bundle low-level allocation and I2S configuration
    if (audio_Open(&hdl->audio) != STATUS_OK) {
        return STATUS_NOT_OK;
    }
    
    if (audio_Initialize(&hdl->audio) != STATUS_OK) {
        return STATUS_NOT_OK;
    }
    
    return STATUS_OK;
}

STATUS pipeline_process(pipeline_hdl_t hdl)
{
    if (!hdl) return STATUS_NOT_OK;

    // Latch switch mutations into firmware core tracking structures at block boundaries
    if (hdl->eq_preset_pending >= 0)
    {
        eq_preset_pending = hdl->eq_preset_pending;
        hdl->eq_preset = hdl->eq_preset_pending; // Synchronize local tracking container
        hdl->eq_preset_pending = -1;            // Clear latch state
    }

    // Safely apply/evaluate coefficients right before execution loop transfers
    audio_apply_pending();

    return audio_Process(&hdl->audio);
}

STATUS pipeline_close(pipeline_hdl_t hdl)
{
    if (!hdl) return STATUS_NOT_OK;
    STATUS ret = audio_Close(&hdl->audio);
    
    // Wipe out the structure parameters completely to ensure no stale values persist
    memset(&s_pipeline, 0, sizeof(struct pipeline_ctx_t));
    return ret;
}

void pipeline_set_eq(pipeline_hdl_t hdl, int preset)
{
    // Strict upper and lower signed boundary checks applied definitively
    if (hdl && preset >= 0 && preset < EQ_NUM_PRESETS) {
        hdl->eq_preset_pending = preset;
    }
}

void pipeline_set_gain(pipeline_hdl_t hdl, int step)
{
    if (hdl && step >= 0 && step < GAIN_STEPS) {
        hdl->gain_step = step;
        audio_set_gain(step);
    }
}

void pipeline_toggle_delay(pipeline_hdl_t hdl)
{
    if (hdl) {
        audio_toggle_delay();
    }
}

int pipeline_get_eq_preset(pipeline_hdl_t hdl)
{// Direct link to the audio core's active preset
    return eq_preset;
}

int pipeline_get_gain_step(pipeline_hdl_t hdl)
{
    return hdl ? hdl->gain_step : 1;
}