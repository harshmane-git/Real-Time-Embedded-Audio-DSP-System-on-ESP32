#include "delay.h"
#include <string.h>
#include <stddef.h>

// Basic information: Opens the delay module and returns the memory size required for the handle structure
STATUS delay_open(uint32_t *pui32Size)
{
    if (pui32Size == NULL)
        return STATUS_NOT_OK;

    *pui32Size = sizeof(delay_hdl_t);
    return STATUS_OK;
}

// Initializes structural parameters, caps max values, computes sample lengths, and wipes memory
STATUS delay_init(delay_hdl_t *phdl,
                  const delay_config_t *psConfig)
{
    if (phdl == NULL || psConfig == NULL)
        return STATUS_NOT_OK;

    float delay_sec = psConfig->delay_seconds;

    if (delay_sec > MAX_DELAY_SECONDS)
        delay_sec = MAX_DELAY_SECONDS;

    if (delay_sec < 0.0f)
        delay_sec = 0.0f;

    phdl->delay_samples =
        (uint32_t)(delay_sec * SAMPLE_RATE + 0.5f);

    if (phdl->delay_samples >= MAX_DELAY_SAMPLES)
        phdl->delay_samples = MAX_DELAY_SAMPLES - 1;

    if (phdl->delay_samples == 0)
        phdl->delay_samples = 1;

    phdl->write_idx = 0;

    memset(phdl->delay_line, 0,
           sizeof(phdl->delay_line));

    return STATUS_OK;
}

//  Iterates over input buffers, processes current historical audio samples, and advances the indices
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

        // Logic note: Checking for empty and full states is not actually implementing boundary constraints or blocking behavior here because a real time streaming audio delay line overwrites historical records in a circular path instead of restricting the data flow

        pfOutput[i] =
            phdl->delay_line[phdl->write_idx];

        phdl->delay_line[phdl->write_idx] =
            input_sample;

        phdl->write_idx++;

        if (phdl->write_idx >=
            phdl->delay_samples)
        {
            phdl->write_idx = 0;
        }
    }

    return STATUS_OK;
}
//  Calculation logic to check if the ring buffer is EMPTY
//  
// if (phdl->write_idx == phdl->read_idx)
// {
//     printf("The ring buffer is completely empty\n");
// }

//  Calculation logic to check if the ring buffer is FULL
//  
// uint32_t next_write_idx = (phdl->write_idx + 1) % phdl->buffer_total_capacity;

// if (next_write_idx == phdl->read_idx)
// {
//     printf("The ring buffer is completely full\n");
// }

//  Deconstructs the tracking instances to complete component lifecycle cleanup operations
STATUS delay_close(delay_hdl_t *phdl)
{
    if (phdl == NULL)
        return STATUS_NOT_OK;

    return STATUS_OK;
}
