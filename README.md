# Real-Time Embedded Audio DSP System on ESP32

Real-time embedded audio DSP system on ESP32 with block-based floating-point processing, I2S integration, modular C architecture, and real-time performance validation.

---

## Overview

This project implements a real-time floating-point digital audio processing pipeline on ESP32 using the ESP-IDF (CMake-based) framework.

The system processes mono audio at **16 kHz** in **256-sample blocks** using time-domain DSP techniques.

The objective is to design, implement, and evaluate a modular embedded audio system while analyzing latency, computational complexity, and real-time constraints.

---

## DSP Processing Pipeline
Mic (I2S)
->
Gain
->
3-Band IIR Equalizer (Biquad)
->
Delay (Circular Buffer)
->
Limiter
->
Speaker (I2S)


All processing is implemented in the **time domain using difference equations**.  
No FFT-based processing is used in the main audio path.

---

## System Specifications

- **Sampling Rate:** 16 kHz  
- **Block Size:** 256 samples  
- **Processing Type:** Floating-point  
- **Audio Format:** Mono  
- **Preset Switching:** Safe update at block boundaries  
- **Memory Policy:** No dynamic memory allocation during processing  

---
## Project Structure

| Folder | Description |
|--------|------------|
| reference_model/ | Python DSP reference implementation |
| esp32_firmware/  | ESP-IDF C implementation |
| host_validation/ | Numerical comparison scripts |
| docs/            | Architecture and performance analysis |

---

## Performance Analysis

The system includes evaluation of:

- **Theoretical Latency:**  
  BlockSize / SamplingRate = 256 / 16000 = 16 ms  

- Measured end-to-end latency  
- CPU load estimation  
- Memory usage analysis  
- Real-time deadline validation  

---

## Learning Outcomes

- Application of sampling theorem in embedded systems  
- IIR filter implementation without FFT  
- Block-based real-time audio processing  
- Floating-point DSP on resource-constrained hardware  
- Modular embedded system architecture design  
