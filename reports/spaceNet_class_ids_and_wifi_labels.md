# SpaceNet advanced class IDs and WiFi6 label definition

Audit date: 2026-11-18. Scope: the AutoDL SpaceNet advanced package and the
0-5 classes used by the modulation-separability experiment.

## 1. What the downloaded package contains

`/root/autodl-fs/SpaceNet_Dataset/advanced/{train,test}` contains only paired
`<stem>.bin` + `<stem>.json` files. There is no class-name table, README or
generator script inside the downloaded folder. Each JSON has numeric `class`
values and physical time-frequency boxes:

```json
{
  "signals": [
    {"signal_id": 0, "start_frequency": 2417.0, "end_frequency": 2437.0,
     "start_time": 0.0, "end_time": 51.0, "class": 2}
  ],
  "observation_range": [2400.0, 2480.0]
}
```

`.bin` is float16 little-endian interleaved I/Q (`I0,Q0,I1,Q1,...`). The
observation bandwidth is `B = observation_range[1] - observation_range[0]`
(MHz), the complex sample rate is `Fs = B`, and `N = B * T` (up to the
one-sample rounding already documented in `stage0_audit.json`). The detector
uses the established 0.7 ms Hann STFT; labels are normalized time-frequency
boxes, not protocol names.

## 2. Verified 14-class mapping

The numeric mapping was cross-checked against the public SpaceNet advanced
schema in `JAVAdcc/project1-yolo-baseline` at commit
`590a2e56374b64ff894901e7b4e1f5ec612a3e1e`
(`reports/advanced_dataset_schema.md`, `reports/class_mapping.json`:
https://github.com/JAVAdcc/project1-yolo-baseline) and against the
SpaceNet/EMSpec literature (arXiv:2610.12142, Table S4: "WiFi6 denotes the six
combinations of 20/40-MHz bandwidth and QPSK/16QAM/64QAM modulation"). It is
exact:

| id | class | nominal bandwidth | notes |
| ---: | --- | ---: | --- |
| 0 | WIFI 20MHz QPSK | 20 MHz | WiFi6, 20 MHz channel |
| 1 | WIFI 20MHz 16QAM | 20 MHz | WiFi6, 20 MHz channel |
| 2 | WIFI 20MHz 64QAM | 20 MHz | WiFi6, 20 MHz channel |
| 3 | WIFI 40MHz QPSK | 40 MHz | WiFi6, 40 MHz channel |
| 4 | WIFI 40MHz 16QAM | 40 MHz | WiFi6, 40 MHz channel |
| 5 | WIFI 40MHz 64QAM | 40 MHz | WiFi6, 40 MHz channel |
| 6 | BLE LE1M | 1 MHz | Bluetooth LE, 1 Msym/s |
| 7 | BLE LE2M | 2 MHz | Bluetooth LE, 2 Msym/s |
| 8 | Zigbee | 2 MHz | IEEE 802.15.4 |
| 9 | LoRa 250KHZ | 0.25 MHz | LoRa, 250 kHz bandwidth |
| 10 | SRRC QPSK | 10 MHz | SRRC-shaped single carrier |
| 11 | SRRC 16QAM | 10 MHz | SRRC-shaped single carrier |
| 12 | AM | 0.1-0.2 MHz | analog amplitude modulation |
| 13 | FM | 0.1-0.2 MHz | analog frequency modulation |

For 0-5 the bandwidth is an exact design value (20/40 MHz). For 6-13 the
table lists the nominal/median bandwidth; some instances vary (e.g. class 9 is
mostly 250 kHz, classes 10/11 are mostly 10 MHz, classes 12/13 are 0.1-0.2 MHz).

Local data checks agree with this table:

- classes 0-2 have label bandwidth exactly 20.000 MHz (n = 7,475 train+test);
- classes 3-5 have label bandwidth exactly 40.000 MHz (n = 3,324);
- class 6 is 1.000 MHz; classes 7/8 are 2.000 MHz; class 9 is 0.250 MHz;
- classes 10/11 are 10.000 MHz at the median; classes 12/13 are 0.1-0.2 MHz.

The label bandwidths are design values, not measured occupied bandwidth, so
the numeric IDs are the authoritative label. The downloaded package itself
does not spell out the names; the table above is the verified external
mapping, not an inference from the detector.

## 3. What "WiFi6" means here

The six WiFi classes are **not** six protocols. They are the Cartesian product

```text
channel bandwidth: 20 MHz, 40 MHz
subcarrier modulation: QPSK, 16QAM, 64QAM
```

so 0/1/2 are the three 20 MHz variants and 3/4/5 are the three 40 MHz
variants, in ascending modulation order. This is the definition used by the
SpaceNet/EMSpec vocabulary ("WiFi6 denotes the six combinations of 20/40-MHz
bandwidth and QPSK/16QAM/64QAM") and matches the exact 20/40 MHz label
bandwidths in the local files. The task is therefore a **within-bandwidth
three-class modulation problem**, not a WiFi-vs-BLE-vs-Zigbee problem.

Consequences for the representation experiment:

- A 20 MHz classifier should only see classes 0/1/2; a 40 MHz classifier only
  3/4/5. Class 2 vs 3 is a bandwidth change, not a modulation change, so
  cross-bandwidth confusion is not the target here.
- 20 MHz WiFi can appear inside 20/30/40/50/80 MHz observation canvases;
  40 MHz WiFi appears only in 40/50/80 MHz canvases. The local crop must use
  the signal bandwidth, not the observation bandwidth.
- The 20/40 MHz label bandwidth is much wider than the 0.7 ms coarse-view
  frequency resolution after resizing; this is exactly why local fine power
  and local complex STFT are compared before deciding on a dual-channel
  representation.

## 4. Waveform-generation caveat

The local package has no generator note. The JSON schema, exact design
bandwidths, balanced class counts and the interleaved float16 format are
consistent with the SpaceNet/CSRD synthetic generation family, but this audit
does not claim that every sample is OTA-measured. For the classifier
experiment the relevant facts are the numeric label, the exact design
bandwidth, and the IQ sample rate; all three are verified above.

Test remains sealed. No original IQ/JSON was modified.
