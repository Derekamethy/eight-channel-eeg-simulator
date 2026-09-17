# Design Decisions

This note records the main engineering choices behind the desktop proof of concept. A production implementation would require hardware-specific requirements, interfaces and verification criteria.

## Why Python?

Python supports rapid engineering prototyping, has a strong scientific and signal-processing ecosystem, integrates easily with serial interfaces, and is suitable for internal engineering tools. It also keeps GUI development, data processing and test automation in one environment.

## Why synthetic EEG?

The generated dataset is deterministic, repeatable, contains no patient data, is safe to demonstrate, and includes a known test event for verification. The synthetic rhythmic event is not claimed to be a clinically valid seizure.

## Why MockSimulatorDevice?

The physical simulator does not exist in this prototype. `MockSimulatorDevice` lets the desktop software be developed and tested independently, while preserving the same application-facing interface that a future serial device can implement. It also makes automated state-transition and protocol tests possible.

## Why Device Abstraction?

```text
GUI
 ↓
SimulatorDevice
 ├─ MockSimulatorDevice
 └─ SerialSimulatorDevice
```

The GUI depends on behaviour, not on a COM port or a specific transport. This separates presentation, protocol and hardware concerns and allows the device implementation to change without rewriting the application layer.

## Why use an MCU buffer?

PC scheduling and USB timing are non-deterministic, while a physical waveform must be generated at a deterministic sample rate.

```text
PC / USB
    ↓
buffer
    ↓
MCU hardware timer
    ↓
DAC
```

The desktop should transfer configuration and waveform data ahead of time; the MCU should own sample timing.

## Why Timer / DMA?

A hardware timer provides a deterministic sample interval. DMA or a tightly controlled interrupt path reduces CPU jitter, transfers repetitive samples efficiently, and supports synchronous multi-channel waveform updates.

## Why a virtual DAC?

The virtual model demonstrates the intended digital target-to-code mapping before a converter and analogue architecture are selected. It does not model physical output accuracy. The real transfer function depends on the DAC, reference, output topology, analogue scaling network and calibration.

## Why separate Desktop Software and Firmware?

### Desktop

- File parsing and test selection
- Waveform visualisation
- Amplitude and channel configuration
- Logging and automation

### Firmware

- Deterministic timing
- Hardware and buffer control
- Synchronous DAC updates
- Device state, errors and status reporting

## Why verification matters?

The simulator is intended as repeatable regression and verification infrastructure, not only as a signal generator.

```text
Known EEG
    ↓
Simulator
    ↓
acquisition device
    ↓
Observed result
    ↓
Expected vs actual
```

Known inputs and expected outcomes make failures reproducible and support root-cause debugging across desktop software, firmware and hardware boundaries.

## Why the Signal Chain view?

The view separates host software, embedded timing, analogue generation, acquisition device acquisition and downstream detection into explicit boundaries.

The same boundaries also define useful verification checkpoints and help isolate faults during development.

## Why three-level verification?

An overall pass/fail result cannot identify where a mismatch begins. The three checkpoints separate simulator-output fidelity, acquisition fidelity and event timing. If Level 1 fails, investigation starts in the simulator output path; if Level 1 passes but Level 2 fails, it moves to the interface/acquisition path; if both waveform levels pass but Level 3 fails, the detection stage becomes the focus. The present checks use fixed-seed software mocks only, so they demonstrate the verification method without claiming physical or clinical validation.
