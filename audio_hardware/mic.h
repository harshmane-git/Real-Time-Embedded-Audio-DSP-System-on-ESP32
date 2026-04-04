#ifndef MIC_H
#define MIC_H

void mic_init(void);
void mic_process(float *buffer, int size);
void mic_close(void);

#endif