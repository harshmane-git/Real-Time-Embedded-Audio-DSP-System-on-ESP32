#include <stdio.h>
#include <string.h>
#include "ring_buffer.h"
#include "test_ring_buffer.h"

static ring_buffer_t rb;

// Helper: fill a slot with known pattern
static void fill_dummy(int32_t *buf, int32_t seed)
{
    for(int i = 0; i < RB_SAMPLES_PER_SLOT; i++)
        buf[i] = seed + i;
}

// Helper: verify pattern
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

    // -----------------------------------------------
    // TEST 1: Init state
    // -----------------------------------------------
    printf("\n[TEST 1] Init state...\n");
    rb_init(&rb);
    bool t1 = (rb_slots_available(&rb) == 0);
    printf(t1 ? "  PASS\n" : "  FAIL\n");

    // -----------------------------------------------
    // TEST 2: Read before 2-slot delay (should block)
    // -----------------------------------------------
    printf("\n[TEST 2] Read before 2-slot delay (must return false)...\n");
    fill_dummy(tx, 100);
    I2S_write(&rb, tx, RB_SAMPLES_PER_SLOT);
    bool read_blocked = !DMA_read(&rb, rx);
    printf(read_blocked ? "  PASS — read correctly blocked\n"
                        : "  FAIL — read should have been blocked!\n");

    // -----------------------------------------------
    // TEST 3: Read unlocks after write reaches slot 3
    // -----------------------------------------------
    printf("\n[TEST 3] Read unlocks after write reaches slot 3...\n");
    fill_dummy(tx, 200);
    I2S_write(&rb, tx, RB_SAMPLES_PER_SLOT);
    fill_dummy(tx, 300);
    I2S_write(&rb, tx, RB_SAMPLES_PER_SLOT);
    bool read_ok = DMA_read(&rb, rx);
    printf(read_ok ? "  PASS — read unblocked\n"
                   : "  FAIL — read still blocked!\n");

    // -----------------------------------------------
    // TEST 4: Data integrity — verify slot 0 content
    // -----------------------------------------------
    printf("\n[TEST 4] Data integrity check (slot 0 = seed 100)...\n");
    bool t4 = verify_dummy(rx, 100);
    printf(t4 ? "  PASS — data matches\n" : "  FAIL — data corrupted!\n");

    // -----------------------------------------------
    // TEST 5: Wrap-around (fill all 8 slots, read all)
    // -----------------------------------------------
    printf("\n[TEST 5] Wrap-around — fill all 8 slots and read back...\n");
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
    printf(wrap_pass ? "  PASS — wrap-around clean\n"
                     : "  FAIL — wrap-around corrupted!\n");

    // -----------------------------------------------
    // TEST 6: Overflow protection
    // -----------------------------------------------
    printf("\n[TEST 6] Overflow — write 9 slots without reading...\n");
    rb_init(&rb);
    fill_dummy(tx, 999);
    bool overflow_caught = false;
    for(int s = 0; s < RB_NUM_SLOTS + 1; s++) {
        bool ok = I2S_write(&rb, tx, RB_SAMPLES_PER_SLOT);
        if(!ok) { overflow_caught = true; }
    }
    printf(overflow_caught ? "  PASS — overflow detected\n"
                           : "  FAIL — overflow not caught!\n");

    printf("\n========== TESTS COMPLETE ==========\n\n");
}