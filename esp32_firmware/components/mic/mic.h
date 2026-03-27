#ifndef MIC_H
#define MIC_H

void mic_init(void);

void mic_read_block(float *buffer, int size);

#endif