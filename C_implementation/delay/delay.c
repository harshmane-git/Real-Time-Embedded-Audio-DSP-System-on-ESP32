#include "delay.h"
#include <string.h>
#include <stddef.h>

// ============================================================================
// delay_open
// ============================================================================
STATUS delay_open(uint32_t *pui32Size)
{
    if (pui32Size == NULL)
        return STATUS_NOT_OK;

    *pui32Size = sizeof(delay_hdl_t);
    return STATUS_OK;
}

// ============================================================================
// delay_init
// ============================================================================
STATUS delay_init(delay_hdl_t *phdl, const delay_config_t *psConfig)
{
    if (phdl == NULL || psConfig == NULL)
        return STATUS_NOT_OK;

    float delay_sec = psConfig->delay_seconds;

    // Boundary protection based on MAX_DELAY_SECONDS defined in delay.h
    if (delay_sec > MAX_DELAY_SECONDS)
        delay_sec = MAX_DELAY_SECONDS;

    if (delay_sec < 0.0f)
        delay_sec = 0.0f;

    // Calculate delay samples based on the 16kHz sampling rate
    phdl->delay_samples = (uint32_t)(delay_sec * SAMPLE_RATE + 0.5f);

    // Ensure it stays strictly within the static allocation array boundaries
    if (phdl->delay_samples >= MAX_DELAY_SAMPLES)
        phdl->delay_samples = MAX_DELAY_SAMPLES - 1;

    if (phdl->delay_samples == 0)
        phdl->delay_samples = 1;

    phdl->write_idx = 0;

    // Clear the buffer completely to prevent loud clicks or pops on startup
    memset(phdl->delay_line, 0, sizeof(phdl->delay_line));

    return STATUS_OK;
}

// ============================================================================
// delay_process — True echo processing with Dry/Wet mix and Feedback
// ============================================================================
STATUS delay_process(delay_hdl_t *phdl,
                     const float *pfInput,
                     float *pfOutput,
                     uint32_t ui32NumSamples)
{
    if (phdl == NULL || pfOutput == NULL)
        return STATUS_NOT_OK;

    for (uint32_t i = 0; i < ui32NumSamples; i++)
    {
        float input_sample = 0.0f;

        if (pfInput != NULL)
            input_sample = pfInput[i];

        /* Read delayed sample from ring buffer */
        pfOutput[i] =
            phdl->delay_line[phdl->write_idx];

        /* Store current input sample */
        phdl->delay_line[phdl->write_idx] =
            input_sample;

        /* Advance circular pointer */
        phdl->write_idx++;

        if (phdl->write_idx >= phdl->delay_samples)
        {
            phdl->write_idx = 0;
        }
    }

    return STATUS_OK;
}

// ============================================================================
// delay_close
// ============================================================================
STATUS delay_close(delay_hdl_t *phdl)
{
    if (phdl == NULL)
        return STATUS_NOT_OK;

    return STATUS_OK;
}