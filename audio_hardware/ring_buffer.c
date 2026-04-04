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
    for(int i = 0; i < len; i++)
    {
        int next = (write_idx + 1) % RB_SIZE;

        if(next == read_idx)
            break;  // prevent overwrite

        buffer[write_idx] = data[i];
        write_idx = next;
    }
}

int rb_read(float *data, int len)
{
    int available = (write_idx - read_idx + RB_SIZE) % RB_SIZE;

    if(available < len)
        return -1;  // not enough data

    for(int i = 0; i < len; i++)
    {
        data[i] = buffer[read_idx];
        read_idx = (read_idx + 1) % RB_SIZE;
    }

    return 0;
}