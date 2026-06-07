# Audio Configuration Encryption and Decryption System

## 1. Objective

The objective of this project is to develop a secure audio configuration management system for DSP (Digital Signal Processing) modules. The system centralizes all user-configurable audio parameters, encrypts them before storage, and decrypts them when required by the DSP engine.

This prevents direct access to tuning parameters and provides a structured mechanism for configuration management.

---

## 2. Problem Statement

Initially, DSP settings were distributed across multiple source files such as:

* gain.c
* delay.c
* equalizer.c

This approach had several disadvantages:

* Difficult configuration management
* Lack of centralized control
* Easy modification of tuning parameters
* No protection of proprietary audio tuning data

To overcome these issues, a centralized encrypted configuration mechanism was designed.

---

## 3. System Architecture

The system consists of the following components:

### 3.1 config.h

Contains all user-tunable DSP parameters.

Example parameters:

* Global Gain
* Delay Time
* Equalizer Gains
* Limiter Threshold
* Feature Enable Flags

### 3.2 audio_config.h

Defines the audio_config_t structure which stores all configuration parameters in a single memory block.

### 3.3 config_generator.c

Responsible for:

* Reading configuration values
* Packing them into audio_config_t
* Encrypting the configuration data
* Generating config.bin

### 3.4 config.bin

Encrypted binary file containing all DSP configuration parameters.

### 3.5 config_loader.c

Responsible for:

* Reading config.bin
* Decrypting configuration data
* Reconstructing audio_config_t
* Providing configuration parameters to the DSP system

---

## 4. Configuration Structure

The configuration parameters are stored using the following structure:

audio_config_t

Fields include:

* enable_eq
* enable_delay
* enable_limiter
* global_gain_db
* low_gain_db
* mid_gain_db
* high_gain_db
* delay_seconds
* limiter_threshold

This structure acts as a centralized container for all DSP settings.

---

## 5. Encryption Process

### Step 1: Load Configuration

Configuration values are read from config.h.

### Step 2: Populate Structure

Values are copied into an audio_config_t object.

### Step 3: Convert Structure into Byte Stream

The structure memory is viewed as a sequence of bytes using:

(uint8_t *)&config

### Step 4: XOR Encryption

Each byte is encrypted using:

data[i] ^= 0xAA;

The encryption key used is:

0xAA

### Step 5: Generate Binary File

Encrypted bytes are written into:

config.bin

using fwrite().

---

## 6. Decryption Process

### Step 1: Open Binary File

config.bin is opened using fopen().

### Step 2: Load Encrypted Bytes

Encrypted bytes are copied into an audio_config_t object using fread().

### Step 3: XOR Decryption

The same XOR operation is applied:

data[i] ^= 0xAA;

Due to the reversible property of XOR:

A XOR B XOR B = A

the original configuration is restored.

### Step 4: Recover Configuration

All DSP parameters become available again inside audio_config_t.

---


## 7. Conclusion

A complete encrypted audio configuration management system was successfully implemented. Configuration parameters are centrally maintained, securely stored in encrypted binary form, and correctly restored during system initialization. This architecture improves maintainability, protects proprietary tuning data, and provides a scalable foundation for future DSP configuration management.

