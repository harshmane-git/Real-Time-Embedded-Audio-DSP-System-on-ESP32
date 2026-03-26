// ─────────────────────────────────────────────────────────────
//  Audio DSP Ring Buffer — ESP-IDF main.c
//
//  Architecture:
//  - Task 1: i2s_writer_task  → simulates I2S_write (sine wave)
//  - Task 2: dma_reader_task  → simulates DMA_read  (copies to filters)
//  - Both tasks run on FreeRTOS with 16ms period
//  - Serial output via ESP_LOGI
//
//  Validation checks:
//  - Overflow count == 0
//  - Underflow count == 0
//  - RMS of copied buffers ≈ 0.7071 (sine wave RMS)
//  - All 3 filter buffers identical after copy
// ─────────────────────────────────────────────────────────────

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include "ring_buffer.h"

// ── Tags for ESP_LOGI ────────────────────────────────────────
#define TAG_MAIN    "MAIN"
#define TAG_WRITE   "I2S_WRITE"
#define TAG_READ    "DMA_READ"
#define TAG_RESULT  "RESULT"

// ── Sine wave config ─────────────────────────────────────────
#define SINE_FREQ_HZ     1000.0f
#define SAMPLE_RATE_F    16000.0f
#define TWO_PI_F         6.28318530718f

// ── Test config ──────────────────────────────────────────────
#define SLOT_PERIOD_MS   16          // 256 samples @ 16kHz = 16ms
#define TOTAL_SLOTS      50          // stop after 50 slots read

// ── Shared state ─────────────────────────────────────────────
static RingBuffer        rb;
static FilterWorkBuffers work;
static SemaphoreHandle_t work_mutex;     // protect work buffers

static volatile uint32_t slots_written = 0;
static volatile uint32_t slots_read    = 0;
static volatile bool     test_done     = false;

// ─────────────────────────────────────────────────────────────
//  Utility: generate one slot of sine wave
// ─────────────────────────────────────────────────────────────
static float sine_phase = 0.0f;

static void generate_sine_slot(float *buf, uint16_t len) {
    float phase_inc = TWO_PI_F * SINE_FREQ_HZ / SAMPLE_RATE_F;
    for (int i = 0; i < len; i++) {
        buf[i]      = sinf(sine_phase);
        sine_phase += phase_inc;
        if (sine_phase >= TWO_PI_F) sine_phase -= TWO_PI_F;
    }
}

// ─────────────────────────────────────────────────────────────
//  Utility: compute RMS of a buffer
// ─────────────────────────────────────────────────────────────
static float compute_rms(const float *buf, uint16_t len) {
    float sum = 0.0f;
    for (int i = 0; i < len; i++) sum += buf[i] * buf[i];
    return sqrtf(sum / len);
}

// ─────────────────────────────────────────────────────────────
//  Utility: print ring buffer slot states
// ─────────────────────────────────────────────────────────────
static void print_rb_state(const char *label) {
    char state_str[64] = {0};
    for (int i = 0; i < RB_NUM_SLOTS; i++) {
        char c;
        switch (rb.slots[i].state) {
            case SLOT_FREE:     c = '.'; break;
            case SLOT_WRITTEN:  c = 'W'; break;
            case SLOT_READING:  c = 'R'; break;
            case SLOT_CONSUMED: c = 'C'; break;
            default:            c = '?'; break;
        }
        state_str[i*2]   = c;
        state_str[i*2+1] = ' ';
    }
    ESP_LOGI(TAG_MAIN, "[%s] slots: %s | wr=%d rd=%d gap=%d | OV=%lu UV=%lu",
        label, state_str,
        rb.write_ptr, rb.read_ptr,
        (rb.write_ptr - rb.read_ptr + RB_NUM_SLOTS) % RB_NUM_SLOTS,
        rb.overflow_count, rb.underflow_count);
}

// ─────────────────────────────────────────────────────────────
//  Task 1: I2S Writer (simulates hardware I2S_write)
//  Runs every 16ms — generates sine wave, writes to ring buffer
// ─────────────────────────────────────────────────────────────
static void i2s_writer_task(void *arg) {
    float temp[RB_SLOT_SAMPLES];
    TickType_t last_wake = xTaskGetTickCount();

    ESP_LOGI(TAG_WRITE, "I2S writer task started");

    while (!test_done) {
        // Wait for next 16ms window
        vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(SLOT_PERIOD_MS));

        generate_sine_slot(temp, RB_SLOT_SAMPLES);
        bool ok = rb_write(&rb, temp, RB_SLOT_SAMPLES);

        if (ok) {
            slots_written++;
            int slot_just_written = (rb.write_ptr - 1 + RB_NUM_SLOTS) % RB_NUM_SLOTS;
            ESP_LOGI(TAG_WRITE, "slot %d written  (total: %lu)",
                     slot_just_written, slots_written);
        } else {
            ESP_LOGW(TAG_WRITE, "OVERFLOW — buffer full, sample dropped! "
                     "(overflow count: %lu)", rb.overflow_count);
        }
    }

    vTaskDelete(NULL);
}

// ─────────────────────────────────────────────────────────────
//  Task 2: DMA Reader (simulates DMA_read)
//  Waits for 2-slot gap, then copies slot to all 3 filter bufs
// ─────────────────────────────────────────────────────────────
static void dma_reader_task(void *arg) {
    TickType_t last_wake = xTaskGetTickCount();

    // Wait for pre-fill (2 slots written first)
    ESP_LOGI(TAG_READ, "DMA reader task started — waiting for 2-slot gap...");
    while (!rb_is_safe_to_read(&rb)) {
        vTaskDelay(pdMS_TO_TICKS(1));
    }
    ESP_LOGI(TAG_READ, "2-slot gap established, starting reads");

    while (!test_done) {
        vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(SLOT_PERIOD_MS));

        if (xSemaphoreTake(work_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
            bool ok = rb_read(&rb, &work);
            xSemaphoreGive(work_mutex);

            if (ok) {
                slots_read++;
                int slot_just_read = (rb.read_ptr - 1 + RB_NUM_SLOTS) % RB_NUM_SLOTS;

                // Compute RMS to verify sine wave copied correctly
                float rms = compute_rms(work.lpf_buf, RB_SLOT_SAMPLES);

                // Verify all 3 filter buffers are identical
                bool match = (memcmp(work.lpf_buf, work.bpf_buf,
                                     RB_SLOT_SAMPLES * sizeof(float)) == 0) &&
                             (memcmp(work.lpf_buf, work.hpf_buf,
                                     RB_SLOT_SAMPLES * sizeof(float)) == 0);

                ESP_LOGI(TAG_READ,
                    "slot %d read  (total: %lu)  RMS=%.4f  buffers_match=%s",
                    slot_just_read, slots_read, rms, match ? "YES" : "NO");

                if (!match) {
                    ESP_LOGE(TAG_READ, "FILTER BUFFER MISMATCH — copy error!");
                }

                // Print full state every 10 slots
                if (slots_read % 10 == 0) {
                    print_rb_state("snapshot");
                }

                // Stop after TOTAL_SLOTS
                if (slots_read >= TOTAL_SLOTS) {
                    test_done = true;
                }

            } else {
                if (!rb_is_safe_to_read(&rb)) {
                    ESP_LOGW(TAG_READ, "gap < 2 slots — waiting");
                } else {
                    ESP_LOGW(TAG_READ, "UNDERFLOW — no data ready "
                             "(underflow count: %lu)", rb.underflow_count);
                }
            }
        }
    }

    vTaskDelete(NULL);
}

// ─────────────────────────────────────────────────────────────
//  app_main
// ─────────────────────────────────────────────────────────────
void app_main(void) {
    ESP_LOGI(TAG_MAIN, "==============================================");
    ESP_LOGI(TAG_MAIN, " Audio DSP Ring Buffer Test — ESP-IDF");
    ESP_LOGI(TAG_MAIN, "==============================================");
    ESP_LOGI(TAG_MAIN, " Slots      : %d", RB_NUM_SLOTS);
    ESP_LOGI(TAG_MAIN, " Slot size  : %d samples x 4 bytes = %d bytes",
             RB_SLOT_SAMPLES, RB_SLOT_SAMPLES * 4);
    ESP_LOGI(TAG_MAIN, " Total RAM  : %d bytes = %d KB",
             RB_NUM_SLOTS * RB_SLOT_SAMPLES * 4,
             RB_NUM_SLOTS * RB_SLOT_SAMPLES * 4 / 1024);
    ESP_LOGI(TAG_MAIN, " Sample rate: %d Hz", RB_SAMPLE_RATE);
    ESP_LOGI(TAG_MAIN, " Slot period: %d ms", SLOT_PERIOD_MS);
    ESP_LOGI(TAG_MAIN, " Test sine  : %.0f Hz", SINE_FREQ_HZ);
    ESP_LOGI(TAG_MAIN, "==============================================");

    // Initialise ring buffer and mutex
    rb_init(&rb);
    work.ready  = false;
    work_mutex  = xSemaphoreCreateMutex();

    // Pre-fill 2 slots to establish safety gap before reader starts
    ESP_LOGI(TAG_MAIN, "Pre-filling 2 slots to establish safety gap...");
    float temp[RB_SLOT_SAMPLES];
    for (int i = 0; i < 2; i++) {
        generate_sine_slot(temp, RB_SLOT_SAMPLES);
        rb_write(&rb, temp, RB_SLOT_SAMPLES);
        slots_written++;
    }
    print_rb_state("after pre-fill");

    // Start tasks
    // Writer: higher priority, pinned to core 0 (like real I2S DMA)
    // Reader: same priority, pinned to core 1 (like real DSP processing)
    xTaskCreatePinnedToCore(i2s_writer_task, "i2s_write", 4096, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(dma_reader_task, "dma_read",  4096, NULL, 5, NULL, 1);

    // Wait for test to finish then print results
    while (!test_done) {
        vTaskDelay(pdMS_TO_TICKS(100));
    }

    vTaskDelay(pdMS_TO_TICKS(200));  // let last log lines flush

    ESP_LOGI(TAG_RESULT, "==============================================");
    ESP_LOGI(TAG_RESULT, " TEST COMPLETE");
    ESP_LOGI(TAG_RESULT, "==============================================");
    ESP_LOGI(TAG_RESULT, " Slots written : %lu", slots_written);
    ESP_LOGI(TAG_RESULT, " Slots read    : %lu", slots_read);
    ESP_LOGI(TAG_RESULT, " Overflows     : %lu", rb.overflow_count);
    ESP_LOGI(TAG_RESULT, " Underflows    : %lu", rb.underflow_count);
    ESP_LOGI(TAG_RESULT, "----------------------------------------------");
    ESP_LOGI(TAG_RESULT, " Overflows  == 0 : %s",
             rb.overflow_count  == 0 ? "PASS" : "FAIL");
    ESP_LOGI(TAG_RESULT, " Underflows == 0 : %s",
             rb.underflow_count == 0 ? "PASS" : "FAIL");
    ESP_LOGI(TAG_RESULT, " RMS ~ 0.7071    : check logs above");
    ESP_LOGI(TAG_RESULT, " Buffers match   : check logs above");
    ESP_LOGI(TAG_RESULT, "==============================================");
}
