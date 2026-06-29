#ifndef PIPELINE_H
#define PIPELINE_H

#include "common_types.h" // For STATUS definitions

// Opaque context handle definition 
typedef struct pipeline_ctx_t *pipeline_hdl_t;

/**
 * @brief Bootstraps the opaque pipeline structure state tracker.
 * @note  Does not initialize hardware drivers. Pre-allocates tracking context in BSS.
 * @param[out] out_hdl Pointer to the initialized opaque pipeline context handle.
 * @return STATUS_OK if structural context assignment succeeds.
 */
STATUS pipeline_open(pipeline_hdl_t *out_hdl);

/**
 * @brief Executes both driver opening and peripheral initialization for the audio layer.
 * @note  This function acts as the unified hardware bring-up layer (calls both audio_Open and audio_Initialize).
 * @param[in] hdl Opaque pipeline handle instance to initialize.
 * @return STATUS_OK if both audio driver open and peripheral initialization succeed.
 */
STATUS pipeline_init(pipeline_hdl_t hdl);

STATUS pipeline_process(pipeline_hdl_t hdl);
STATUS pipeline_close(pipeline_hdl_t hdl);

// State Mutators called by switch_control layer
void pipeline_set_eq(pipeline_hdl_t hdl, int preset);
void pipeline_set_gain(pipeline_hdl_t hdl, int step);
void pipeline_toggle_delay(pipeline_hdl_t hdl);

// Clean Getters to prevent switch_control from reaching into internals
int pipeline_get_eq_preset(pipeline_hdl_t hdl);
int pipeline_get_gain_step(pipeline_hdl_t hdl);

#endif // PIPELINE_H