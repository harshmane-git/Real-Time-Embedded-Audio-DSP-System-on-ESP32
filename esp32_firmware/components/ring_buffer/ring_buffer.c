#include "ring_buffer.h"

static float buffer[RB_SIZE];
static int write_idx = 0;
static int read_idx = 0;

void rb_init(void)
{
    write_idx = 0;
    read_idx = 0;
}

void rb_write(float *data, int len)
{
    for(int i=0;i<len;i++)
    {
        buffer[write_idx] = data[i];
        write_idx = (write_idx + 1) % RB_SIZE;
    }
}

void rb_read(float *data, int len)
{
    for(int i=0;i<len;i++)
    {
        data[i] = buffer[read_idx];
        read_idx = (read_idx + 1) % RB_SIZE;
    }
}

int rb_available(void)
{
    return (write_idx - read_idx + RB_SIZE) % RB_SIZE;
}

