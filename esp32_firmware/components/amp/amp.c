#include "amp.h"
#include <stdio.h>
#include <inttypes.h>

void amp_init(void)
{
}

void amp_write_block(int32_t *buffer, int size)
{
    printf("OUT: %" PRId32 "\n", buffer[0]);
}