#include "biquad.h"

STATUS biquad_open(uint32_t *pui32Size) {
    if (pui32Size == NULL) return STATUS_NOT_OK;
    
    *pui32Size = sizeof(biquad_t);
    return STATUS_OK;
}

//  This sets up our filter by copying the math coefficients into our handle structure and clearing out any old history so we start with pure silence
STATUS biquad_init(biquad_t *phdl, const float coeffs[5]) {
    if (phdl == NULL || coeffs == NULL) return STATUS_NOT_OK;
    
    // Copy feedforward b coefficients which scale our incoming signal to alter frequency volume amplitudes
    phdl->b[0] = coeffs[0]; phdl->b[1] = coeffs[1]; phdl->b[2] = coeffs[2];
    
    // Copy feedback a coefficients which scale our past states to create filter resonance and sharpness
    phdl->a[0] = coeffs[3]; phdl->a[1] = coeffs[4];
    
    // Clear out the state history buffers to ensure no residual system noise sounds at startup
    phdl->w[0] = 0.0f;
    phdl->w[1] = 0.0f;
    return STATUS_OK;
}

//  This is the main math loop that processes a whole block of audio samples using an optimized Direct Form Two difference equation layout
STATUS biquad_process_block(biquad_t *phdl, const float *pfInput, float *pfOutput, uint32_t ui32NumSamples) {
    if (phdl == NULL || pfInput == NULL || pfOutput == NULL) return STATUS_NOT_OK;
    
    // Loop step by step through every individual audio sample in the block array passed to us
    for (uint32_t i = 0; i < ui32NumSamples; i++) {
        // Grab the current raw audio sample from the input buffer
        float x = pfInput[i];
        
        // Logic note: Checking for empty and full states is not actually implementing boundary checks here because a biquad filter does not use a growing ring buffer queue it just updates a tiny fixed array of past sample calculations
        
        // Calculate the intermediate state value by subtracting past feedback loops scaled by our a coefficients from the input
        float w0 = x - phdl->a[0]*phdl->w[0] - phdl->a[1]*phdl->w[1];
        
        // Combine the current intermediate state with past states using the feedforward b coefficients to create our filtered output
        pfOutput[i] = phdl->b[0]*w0 + phdl->b[1]*phdl->w[0] + phdl->b[2]*phdl->w[1];
        
        // Shift our history states back by one step in time so they are ready to be used by the very next audio sample in line
        phdl->w[1] = phdl->w[0];
        phdl->w[0] = w0;
    }
    return STATUS_OK;
}

STATUS biquad_close(biquad_t *phdl) {
    if (phdl == NULL) return STATUS_NOT_OK;
    return STATUS_OK;
}
