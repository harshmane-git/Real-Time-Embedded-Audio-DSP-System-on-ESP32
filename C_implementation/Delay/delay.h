#ifndef DELAY_H
#define DELAY_H

#include <stdint.h>
#include "common_types.h"

#define SAMPLE_RATE       16000
#define MAX_DELAY_SECONDS 0.5f
#define MAX_DELAY_SAMPLES 8192

typedef struct {
    float delay_seconds;
} delay_config_t;

typedef struct {
    float delay_line[MAX_DELAY_SAMPLES];
    uint32_t delay_samples;
    uint32_t write_idx;
} delay_hdl_t;

STATUS delay_open(uint32_t *pui32Size);
STATUS delay_init(delay_hdl_t *phdl, const delay_config_t *psConfig);
STATUS delay_process(delay_hdl_t *phdl, const float *pfInput, float *pfOutput, uint32_t ui32NumSamples);
STATUS delay_close(delay_hdl_t *phdl);

#endif