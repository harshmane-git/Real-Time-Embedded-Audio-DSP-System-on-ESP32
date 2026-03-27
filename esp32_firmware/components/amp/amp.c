#include "amp.h"
#include <stdio.h>

void amp_init(void)
{
}

void amp_write_block(float *buffer, int size)
{
    printf("OUT: %f\n", buffer[0]);
}
