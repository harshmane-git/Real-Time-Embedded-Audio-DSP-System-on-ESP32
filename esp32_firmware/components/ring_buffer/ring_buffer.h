#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#define RB_SIZE 4096

void rb_init(void);

void rb_write(float *data, int len);

void rb_read(float *data, int len);

int rb_available(void);

#endif
