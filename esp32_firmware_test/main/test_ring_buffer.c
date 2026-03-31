#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "ring_buffer.h"
#include "test_ring_buffer.h"

// ===============================================
// LEVEL 1 — Static Unit Tests
// ===============================================

static ring_buffer_t rb;

static void fill_dummy(int32_t *buf, int32_t seed)
{
    for(int i = 0; i < RB_SAMPLES_PER_SLOT; i++)
        buf[i] = seed + i;
}

static bool verify_dummy(int32_t *buf, int32_t seed)
{
    for(int i = 0; i < RB_SAMPLES_PER_SLOT; i++) {
        if(buf[i] != seed + i) {
            printf("  MISMATCH at [%d]: expected %ld got %ld\n",
                   i, (long)(seed + i), (long)buf[i]);
            return false;
        }
    }
    return true;
}

void test_rb_run(void)
{
    int32_t tx[RB_SAMPLES_PER_SLOT];
    int32_t rx[RB_SAMPLES_PER_SLOT];

    printf("\n========== RING BUFFER TESTS ==========\n");

    // TEST 1: Init state
    printf("\n[TEST 1] Init state...\n");
    rb_init(&rb);
    bool t1 = (rb_slots_available(&rb) == 0);
    printf(t1 ? "  PASS\n" : "  FAIL\n");

    // TEST 2: Read before 2-slot delay (should block)
    printf("\n[TEST 2] Read before 2-slot delay (must return false)...\n");
    fill_dummy(tx, 100);
    I2S_write(&rb, tx, RB_SAMPLES_PER_SLOT);
    bool read_blocked = !DMA_read(&rb, rx);
    printf(read_blocked ? "  PASS - read correctly blocked\n"
                        : "  FAIL - read should have been blocked!\n");

    // TEST 3: Read unlocks after write reaches slot 3
    printf("\n[TEST 3] Read unlocks after write reaches slot 3...\n");
    fill_dummy(tx, 200);
    I2S_write(&rb, tx, RB_SAMPLES_PER_SLOT);
    fill_dummy(tx, 300);
    I2S_write(&rb, tx, RB_SAMPLES_PER_SLOT);
    bool read_ok = DMA_read(&rb, rx);
    printf(read_ok ? "  PASS - read unblocked\n"
                   : "  FAIL - read still blocked!\n");

    // TEST 4: Data integrity
    printf("\n[TEST 4] Data integrity check (slot 0 = seed 100)...\n");
    bool t4 = verify_dummy(rx, 100);
    printf(t4 ? "  PASS - data matches\n" : "  FAIL - data corrupted!\n");

    // TEST 5: Wrap-around
    printf("\n[TEST 5] Wrap-around - fill all 8 slots and read back...\n");
    rb_init(&rb);
    bool wrap_pass = true;
    for(int s = 0; s < RB_NUM_SLOTS; s++) {
        fill_dummy(tx, s * 1000);
        I2S_write(&rb, tx, RB_SAMPLES_PER_SLOT);
    }
    for(int s = 0; s < RB_NUM_SLOTS; s++) {
        if(DMA_read(&rb, rx)) {
            if(!verify_dummy(rx, s * 1000)) {
                printf("  FAIL at slot %d\n", s);
                wrap_pass = false;
            }
        }
    }
    printf(wrap_pass ? "  PASS - wrap-around clean\n"
                     : "  FAIL - wrap-around corrupted!\n");

    // TEST 6: Overflow protection
    printf("\n[TEST 6] Overflow - write 9 slots without reading...\n");
    rb_init(&rb);
    fill_dummy(tx, 999);
    bool overflow_caught = false;
    for(int s = 0; s < RB_NUM_SLOTS + 1; s++) {
        bool ok = I2S_write(&rb, tx, RB_SAMPLES_PER_SLOT);
        if(!ok) { overflow_caught = true; }
    }
    printf(overflow_caught ? "  PASS - overflow detected\n"
                           : "  FAIL - overflow not caught!\n");

    printf("\n========== TESTS COMPLETE ==========\n\n");
}

// ===============================================
// LEVEL 2 — FreeRTOS Two-Task Test
// ===============================================

#define WRITE_INTERVAL_MS    16
#define TOTAL_SLOTS_TO_WRITE 32

static ring_buffer_t rb_rtos;

static volatile int  slots_written = 0;
static volatile int  slots_read    = 0;
static volatile int  read_errors   = 0;
static volatile int  write_errors  = 0;
static volatile bool writer_done   = false;

static void writer_task(void *pvParams)
{
    int32_t tx[RB_SAMPLES_PER_SLOT];

    for(int s = 0; s < TOTAL_SLOTS_TO_WRITE; s++)
    {
        for(int i = 0; i < RB_SAMPLES_PER_SLOT; i++)
            tx[i] = (s * 1000) + i;

        if(I2S_write(&rb_rtos, tx, RB_SAMPLES_PER_SLOT))
        {
            slots_written++;
            printf("[WRITER] Slot %d written (slots_filled=%d)\n",
                   s, rb_slots_available(&rb_rtos));
        }
        else
        {
            write_errors++;
            printf("[WRITER] WARNING: Overflow at slot %d!\n", s);
        }

        vTaskDelay(pdMS_TO_TICKS(WRITE_INTERVAL_MS));
    }

    writer_done = true;
    printf("[WRITER] Done. Total written: %d, errors: %d\n",
           slots_written, write_errors);
    vTaskDelete(NULL);
}

static void reader_task(void *pvParams)
{
    int32_t rx[RB_SAMPLES_PER_SLOT];
    int expected_seed = 0;

    while(!writer_done || rb_slots_available(&rb_rtos) > 0)
    {
        if(DMA_read(&rb_rtos, rx))
        {
            bool slot_ok = true;
            for(int i = 0; i < RB_SAMPLES_PER_SLOT; i++)
            {
                int32_t expected = (expected_seed * 1000) + i;
                if(rx[i] != expected)
                {
                    printf("[READER] MISMATCH slot %d at [%d]: "
                           "expected %ld got %ld\n",
                           expected_seed, i,
                           (long)expected, (long)rx[i]);
                    read_errors++;
                    slot_ok = false;
                    break;
                }
            }
            if(slot_ok)
                printf("[READER] Slot %d verified OK (slots_filled=%d)\n",
                       expected_seed, rb_slots_available(&rb_rtos));

            slots_read++;
            expected_seed++;
        }
        else
        {
            vTaskDelay(pdMS_TO_TICKS(1));
        }
    }

    printf("[READER] Done. Total read: %d, errors: %d\n",
           slots_read, read_errors);
    vTaskDelete(NULL);
}

void test_rb_rtos(void)
{
    printf("\n========== LEVEL 2: FreeRTOS TASK TEST ==========\n");
    printf("Writer: %d slots at %dms intervals\n",
           TOTAL_SLOTS_TO_WRITE, WRITE_INTERVAL_MS);
    printf("Reader: starts after 2-slot delay, verifies data\n\n");

    rb_init(&rb_rtos);
    slots_written = 0;
    slots_read    = 0;
    read_errors   = 0;
    write_errors  = 0;
    writer_done   = false;

    xTaskCreate(writer_task, "writer", 4096, NULL, 5, NULL);
    xTaskCreate(reader_task, "reader", 4096, NULL, 5, NULL);

    while(!writer_done || rb_slots_available(&rb_rtos) > 0)
        vTaskDelay(pdMS_TO_TICKS(100));

    vTaskDelay(pdMS_TO_TICKS(200));

    printf("\n========== LEVEL 2 RESULTS ==========\n");
    printf("Slots written : %d / %d  %s\n", slots_written, TOTAL_SLOTS_TO_WRITE,
           slots_written == TOTAL_SLOTS_TO_WRITE ? "PASS" : "FAIL");
    printf("Slots read    : %d / %d  %s\n", slots_read, TOTAL_SLOTS_TO_WRITE,
           slots_read == TOTAL_SLOTS_TO_WRITE ? "PASS" : "FAIL");
    printf("Write errors  : %d  %s\n", write_errors,
           write_errors == 0 ? "PASS" : "FAIL");
    printf("Read errors   : %d  %s\n", read_errors,
           read_errors == 0 ? "PASS" : "FAIL");

    bool all_pass = (slots_written == TOTAL_SLOTS_TO_WRITE) &&
                    (slots_read    == TOTAL_SLOTS_TO_WRITE) &&
                    (write_errors  == 0) &&
                    (read_errors   == 0);

    printf("\nLevel 2 overall: %s\n", all_pass ? "PASS" : "FAIL");
    printf("=====================================\n\n");
}