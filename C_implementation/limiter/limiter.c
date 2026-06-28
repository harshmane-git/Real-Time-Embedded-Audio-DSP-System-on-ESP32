#include "limiter.h"

STATUS limiter_open(uint32_t *pui32Size) {
    if (pui32Size == NULL) return STATUS_NOT_OK;
    *pui32Size = sizeof(limiter_hdl_t);
    return STATUS_OK;
}

//  This maps the user defined volume ceiling threshold into our tracking handle memory structure
STATUS limiter_init(limiter_hdl_t *phdl, const limiter_config_t *psConfig) {
    if (phdl == NULL || psConfig == NULL) return STATUS_NOT_OK;
    phdl->fThreshold = psConfig->fThreshold;
    return STATUS_OK;
}

//  This loops sample by sample checking if any audio wave peaks exceed our threshold and chops them off to prevent speaker distortion
STATUS limiter_process(limiter_hdl_t *phdl, const float *pfInput, float *pfOutput, uint32_t ui32NumSamples) {
    if (phdl == NULL || pfInput == NULL || pfOutput == NULL) return STATUS_NOT_OK;
    
    // Process our block array sample by sample to protect the output against clipping spikes
    for (uint32_t i = 0; i < ui32NumSamples; i++) {
        // Grab the current raw audio sample value
        float val = pfInput[i];
        
        // Checking for empty and full states is not actually implementing boundary checks here because a hard limiter acts on individual variables instantly without using memory history queues or circular ring buffer arrays
        
        // Check if the signal wave peak shoots above our positive ceiling limit and clamp it down
        if (val > phdl->fThreshold)      val = phdl->fThreshold;
        
        // Check if the signal wave peak drops below our mirror negative floor limit and pull it up
        else if (val < -phdl->fThreshold) val = -phdl->fThreshold;
        
        // Export our safely constrained audio value out to the destination output buffer
        pfOutput[i] = val;
    }
    return STATUS_OK;
}

STATUS limiter_close(limiter_hdl_t *phdl) {
    if (phdl == NULL) return STATUS_NOT_OK;
    return STATUS_OK;
}
