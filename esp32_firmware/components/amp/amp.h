#ifndef AMP_H
#define AMP_H
#include <inttypes.h>
#include <stdint.h>

void amp_init(void);
void amp_write_block(int32_t *buffer, int size);

#endif