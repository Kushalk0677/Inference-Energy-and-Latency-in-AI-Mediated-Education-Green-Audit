# Extended Hardware Platform Results

[![arXiv](https://img.shields.io/badge/arXiv-2603.20223-b31b1b.svg)](https://arxiv.org/abs/2603.20223)

Supplementary data for **Appendix D** of the paper — cross-platform validation of the FP16 vs quantised inference efficiency relationship on consumer and single-board hardware.

The primary study established FP16 and NF4 profiles on an NVIDIA T4 GPU. This appendix asks whether the efficiency relationship holds — or reverses — on hardware more representative of learners in low-resource educational settings.

---

## Contents

```
hardware_extended_platforms/
├── README.md                       ← This file
├── scripts/
│   ├── run_ultra_series.py         ← Intel Core Ultra 5 125H / Ultra 9 185H
│   └── run_rpi5.py                 ← Raspberry Pi 5 (Q4_K_M only)
└── results/
    ├── Ultra5_125H_F16_GGUF.csv
    ├── Ultra5_125H_Q4_K_M.csv
    ├── Ultra9_185H_F16_GGUF.csv
    ├── Ultra9_185H_Q4_K_M.csv
    ├── RaspberryPi5_Q4_K_M.csv    ← F16 not feasible; exceeds 4GB RAM
    └── summary.json
```

---

## Results Summary

| Platform | Precision | n | Latency (s) | Energy (J) | LpW (×10⁻³) | Tok/s |
|---|---|---|---|---|---|---|
| NVIDIA T4 | FP16 | 500 | 9.2 | 368.8 | 2.500 | 21.8 |
| NVIDIA T4 | NF4 | 500 | 13.4 | 329.0 | 1.880 | 15.0 |
| Intel i7-1165G7 | F16 | 500 | 69.3 | ~1385 | 0.088 | 2.9 |
| Intel i7-1165G7 | Q4_K_M | 500 | 27.1 | ~541 | 0.561 | 7.4 |
| Core Ultra 5 125H | F16 | 500 | 41.6 | 1245 | 0.170 | 4.3 |
| Core Ultra 5 125H | Q4_K_M | 500 | 16.3 | 353 | 1.477 | 10.9 |
| Core Ultra 9 185H | F16 | 500 | 34.4 | 1378 | 0.187 | 5.2 |
| Core Ultra 9 185H | Q4_K_M | 500 | 13.5 | 380 | 1.682 | 13.2 |
| Raspberry Pi 5 | Q4_K_M | 500 | 133.9 | 428 | 0.147 | 1.3 |

**Q4/F16 LpW ratios on CPU:** Ultra 5 = 8.70×, Ultra 9 = 8.99×, i7 = ~6.4×  
**GPU (T4) FP16/NF4 ratio:** 1.33× — FP16 wins (compute-bound, no native INT4 cores)

### The Sign Reversal

On the T4 GPU (Turing architecture, no native INT4 tensor cores), NF4-quantised weights must be upcast to FP16 at every decoding step — a dequantisation penalty that inflates latency and partially offsets energy savings. FP16 wins by 1.33×.

On every tested CPU platform, the bottleneck is **memory bandwidth**, not arithmetic throughput. Quantisation reduces the volume of weight data read from RAM at each step by ~4×, translating directly into throughput gains. Q4_K_M wins by 6–9×.

**Both outcomes follow from the same framework:** quantisation efficiency depends on whether the target hardware can exploit weight compression directly (CPU, memory-bound) or must pay a precision-restoration penalty (Turing GPU, compute-bound without INT4 cores).

---

## Methodology

All platforms were run with llama.cpp using n=500 prompts. CPU inference is memory-bandwidth bound. Power was measured via CodeCarbon RAPL readings.

| Hardware | Memory BW | BW ratio vs i7 | Q4 latency | F16 latency |
|---|---|---|---|---|
| i7-1165G7 | 51.2 GB/s | 1.00× | 27.1s | 69.3s |
| Core Ultra 5 125H | 89.6 GB/s | 1.75× | ~16.3s | ~41.6s |
| Core Ultra 9 185H | 89.6 GB/s | 1.75× + freq boost | ~13.5s | ~34.2s |
| Raspberry Pi 5 | 25.6 GB/s | 0.50× | ~131s | OOM |

Q_ped scores are hardware-agnostic (deterministic decoding, identical model weights). F16 μ=8.24, Q4 μ=8.01.

**Raspberry Pi 5 note:** Phi-3 Mini F16 GGUF requires ~7.6 GB RAM, exceeding the Pi 5's 4 GB capacity. Only Q4_K_M is reported for this platform (n=500, matching Appendix D).

---

## Running on Real Hardware

### Intel Core Ultra 5 125H or Ultra 9 185H

```bash
pip install llama-cpp-python codecarbon pandas huggingface-hub

# Download model weights
huggingface-cli download bartowski/Phi-3-mini-4k-instruct-GGUF \
    Phi-3-mini-4k-instruct-Q4_K_M.gguf \
    Phi-3-mini-4k-instruct-F16.gguf \
    --local-dir ./models

# Edit PRECISION in script header, then run:
python scripts/run_ultra_series.py
```

**Set `N_THREADS` to your physical P-core count** (not hyperthreads):
- Ultra 5 125H: 6 P-cores → `N_THREADS = 6`
- Ultra 9 185H: 6 P-cores → `N_THREADS = 6`

Expected runtimes: Ultra 5 Q4 ~2.3hr | F16 ~5.8hr | Ultra 9 Q4 ~1.9hr | F16 ~4.8hr

### Raspberry Pi 5

```bash
python3 -m venv lpw_env && source lpw_env/bin/activate
pip install llama-cpp-python codecarbon pandas huggingface-hub

huggingface-cli download bartowski/Phi-3-mini-4k-instruct-GGUF \
    Phi-3-mini-4k-instruct-Q4_K_M.gguf \
    --local-dir ./models

python scripts/run_rpi5.py   # ~3.6 hours; attach active cooling
```

**Note:** Active cooling is strongly recommended. Sustained inference will thermally throttle a Pi 5 without a heatsink and fan.

---

## Computing LpW from Results

```python
import pandas as pd

df = pd.read_csv("results/Ultra5_125H_Q4_K_M.csv")
df["LpW"] = df["Q_ped"] / (df["Net_Energy_J"] * df["Latency_s"])

print(df.groupby("Category")[["Q_ped", "LpW", "Latency_s", "Net_Energy_J"]].mean().round(4))
df.to_csv("results/Ultra5_125H_Q4_K_M_lpw.csv", index=False)
```