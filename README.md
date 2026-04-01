# Inference Energy and Latency in AI-Mediated Education: A Learning-per-Watt Analysis of Edge and Cloud Models

[![arXiv](https://img.shields.io/badge/arXiv-2603.20223-b31b1b.svg)](https://arxiv.org/abs/2603.20223)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Kushal Khemani** · Billabong High International School, Hadapsar, Pune, India  
`kushal.khemani@gmail.com`

---

## Overview

This repository contains all code, data, and scoring sheets for the empirical study comparing FP16 and 4-bit NF4 inference of **Microsoft Phi-3 Mini (4k-instruct)** on an NVIDIA T4 GPU, evaluated across 500 secondary school educational prompts.

The paper introduces **Learning-per-Watt (LpW)** — a metric that quantifies pedagogical value delivered per unit of energy expended over the learner's waiting window:

$$\text{LpW}_i = \frac{Q_{\text{ped},i}}{E_{\text{net},i} \times L_i}$$

where $Q_{\text{ped}}$ is pedagogical quality (1–10), $E_{\text{net}}$ is net AI-attributable energy in Joules, and $L$ is response latency in seconds.

### Key Results (KV-cache enabled, n = 500 prompts, NVIDIA T4)

| Configuration | Latency (s) | Energy (J) | Q_ped | LpW (×10⁻³) |
|---|---|---|---|---|
| FP16 (edge, T4) | 9.2 | 368.8 | 8.24 | 2.50 |
| NF4 (edge, T4) | 13.4 | 329.0 | 8.05 | 1.88 |
| FP16 / NF4 ratio | 1.46× faster | 10.8% higher | +0.19 pts | **1.33× higher** |

> **Core finding:** The FP16–NF4 efficiency gap is *inference-regime dependent* — 1.33× under realistic KV-cache-enabled deployment versus 7.4× under cache-disabled benchmarking (the configuration most commonly used in offline evaluation). Stateless benchmarks overstate the FP16 advantage by more than fivefold.

---

## Repository Structure

```
.
├── code/
│   ├── colab/
│   │   ├── kvtrue_FP16.py          # Primary T4 experiment (FP16, use_cache=True)
│   │   ├── kvtrue_NF4.py           # Primary T4 experiment (NF4, use_cache=True)
│   │   └── backup_download.py      # Checkpoint download utility
│   ├── local/
│   │   ├── local_colab.py          # Raw data collection variant (Colab, no scoring)
│   │   └── local_pc.py             # Windows-compatible version of local_colab.py
│   ├── laptop_benchmark.py         # CPU inference benchmark (llama.cpp, F16 vs Q4_K_M)
│   └── cloud_scoring.py            # Cloud LLM AI scoring via Anthropic API
│
├── hardware_extended_platforms/    # Appendix D — cross-platform validation
│   ├── README.md                   # Methodology, results table, setup instructions
│   ├── scripts/
│   │   ├── run_ultra_series.py     # Intel Core Ultra 5 125H / Ultra 9 185H
│   │   └── run_rpi5.py             # Raspberry Pi 5
│   └── results/
│       ├── Ultra5_125H_F16_GGUF.csv
│       ├── Ultra5_125H_Q4_K_M.csv
│       ├── Ultra9_185H_F16_GGUF.csv
│       ├── Ultra9_185H_Q4_K_M.csv
│       ├── RaspberryPi5_Q4_K_M.csv   # Q4_K_M only (F16 not feasible — exceeds 4GB RAM)
│       └── summary.json
│
├── data/
│   ├── prompts/
│   │   └── LpW_500_Questions.xlsx  # 500 educational prompts (100 × 5 categories)
│   ├── figures/
│   │   ├── fig_lpw_dist.png        # Figure 1 — LpW distributions (FP16 vs NF4)
│   │   └── fig_sensitivity.png     # Appendix C — sensitivity & cloud scenario analysis
│   ├── scoring/
│   │   ├── ai_scorer_system_prompt.md      # System prompt used for GPT-4 / Claude / Gemini scoring
│   │   └── teacher_scoring_instructions.md # Rubric and instructions for human raters
│   ├── kvcache_true/               # PRIMARY STUDY — use_cache=True (KV-cache enabled)
│   │   ├── FP16/
│   │   │   ├── phi3_FP16_corrected_500prompts.csv   # Raw inference results
│   │   │   ├── phi3_FP16_chatgpt.xlsx               # GPT-4 scores
│   │   │   ├── phi3_FP16_claude_scored.xlsx         # Claude 3.5 Sonnet scores
│   │   │   ├── phi3_FP16_gemini_scored.xlsx         # Gemini 1.5 Pro scores
│   │   │   └── phi3_FP16_teacher_scored.xlsx        # 10-teacher panel scores
│   │   └── NF4/                    # Same structure as FP16
│   ├── kvcache_false/              # SECONDARY STUDY — use_cache=False (Appendix C)
│   │   ├── FP16/
│   │   │   ├── checkpoints/        # 10 checkpoint CSVs (every 50 prompts)
│   │   │   └── evaluation_sheets/  # Scoring templates and completed sheets
│   │   └── NF4/
│   │       ├── checkpoints/        # 4 backup CSVs (session-split collection)
│   │       └── evaluation_sheets/
│   ├── windows/
│   │   ├── laptop_100_prompts.csv              # 100-prompt stratified sample
│   │   ├── phi3_F16_windows_100prompts.csv     # F16 results (Intel i7, Iris Xe)
│   │   └── phi3_Q4_K_M_windows_100prompts.csv  # Q4_K_M results
│   ├── cloud_comparison_results.csv  # Cloud LLM latency measurements
│   └── model_comparison_results.csv  # Appendix A model selection experiment
│
├── supplements/
│   ├── multimodel_lpw_supplement.docx          ← Multi-model LpW validation (Mistral 7B, TinyLlama, Phi-3)
│   └── math_stats_supplement.docx              ← Mathematical and statistical supplement
│
└── student_study/
    ├── research_study_app.html     # Web app used for student interaction study
    └── form.txt                    # Consent / data collection form text
```

---

## Quickstart

### T4 GPU (Google Colab)

```python
# Install dependencies (Colab cell 1 — restart runtime after)
!pip install -q "transformers==4.44.2" accelerate bitsandbytes codecarbon

# Then run either:
# code/colab/kvtrue_FP16.py   → FP16 inference (set USE_QUANTIZATION = False)
# code/colab/kvtrue_NF4.py    → NF4 inference (set USE_QUANTIZATION = True)
```

### Consumer Laptop / CPU (llama.cpp)

```bash
pip install llama-cpp-python codecarbon pandas huggingface-hub

# Download GGUF weights
huggingface-cli download bartowski/Phi-3-mini-4k-instruct-GGUF \
    Phi-3-mini-4k-instruct-Q4_K_M.gguf \
    Phi-3-mini-4k-instruct-fp16.gguf \
    --local-dir ./models

# Run benchmark (edit QUANTIZATION at top of script)
python code/laptop_benchmark.py
```

### Extended Hardware Platforms

See [`hardware_extended_platforms/README.md`](hardware_extended_platforms/README.md) for setup instructions and results for:
- Intel Core Ultra 5 125H
- Intel Core Ultra 9 185H  
- Raspberry Pi 5 (Q4_K_M only)

---

## Hardware Platform Summary

| Platform | Arch | Precision | n | Latency (s) | Energy (J) | LpW (×10⁻³) | Q4/F16 LpW ratio |
|---|---|---|---|---|---|---|---|
| NVIDIA T4 | GPU (Turing) | FP16 | 500 | 9.2 | 368.8 | 2.500 | — |
| NVIDIA T4 | GPU (Turing) | NF4 | 500 | 13.4 | 329.0 | 1.880 | 0.75× (F16 wins) |
| Intel i7-1165G7 | CPU | F16 | 500 | 69.3 | ~1385 | 0.088 | — |
| Intel i7-1165G7 | CPU | Q4_K_M | 500 | 27.1 | ~541 | 0.561 | **6.4× (Q4 wins)** |
| Core Ultra 5 125H | CPU | F16 | 500 | 41.6 | 1245 | 0.170 | — |
| Core Ultra 5 125H | CPU | Q4_K_M | 500 | 16.3 | 353 | 1.477 | **8.7× (Q4 wins)** |
| Core Ultra 9 185H | CPU | F16 | 500 | 34.4 | 1378 | 0.187 | — |
| Core Ultra 9 185H | CPU | Q4_K_M | 500 | 13.5 | 380 | 1.682 | **9.0× (Q4 wins)** |
| Raspberry Pi 5 | CPU | Q4_K_M | 500 | 133.9 | 428 | 0.147 | — |

> **The efficiency advantage of quantisation reverses between GPU and CPU.** On the T4 (no native INT4 cores), FP16 wins by 1.33×. On every tested CPU platform, Q4_K_M wins by 6–9×. The binding constraint shifts from dequantisation overhead (GPU) to memory bandwidth (CPU).

---

## Pedagogical Scoring

Responses were scored by a hybrid panel of **10 Cambridge International secondary school teachers** and **3 frontier AI systems** (GPT-4, Claude 3.5 Sonnet, Gemini 1.5 Pro) using a four-dimension rubric aggregated at 60/40 human–AI weighting.

| Dimension | Code | What is assessed |
|---|---|---|
| Conceptual Accuracy | CA | Factual correctness; absence of critical misconceptions |
| Clarity & Coherence | CC | Logical structure and readability |
| Scaffolding Quality | SQ | Progressive knowledge building; examples, analogies, step-by-step reasoning |
| Level Appropriateness | LA | Suitability for secondary school learners (ages 14–18) |

Full rubric and scoring instructions: [`data/scoring/teacher_scoring_instructions.md`](data/scoring/teacher_scoring_instructions.md)  
AI system prompt: [`data/scoring/ai_scorer_system_prompt.md`](data/scoring/ai_scorer_system_prompt.md)

---

## Citation

If you use this code, data, or the LpW metric in your work, please cite:

```bibtex
@article{khemani2026lpw,
  title     = {Inference Energy and Latency in AI-Mediated Education:
               A Learning-per-Watt Analysis of Edge and Cloud Models},
  author    = {Khemani, Kushal},
  journal   = {arXiv preprint arXiv:2603.20223},
  year      = {2026},
  url       = {https://arxiv.org/abs/2603.20223}
}
```

---

## Ethics

Teacher raters participated voluntarily in their professional capacity as subject-specialist educators. All 10 teachers were independent of the author and were not affiliated with Billabong High International School, Hadapsar. No personally identifiable data were collected from raters or students. Teacher scores are identified only by numeric IDs (Teacher 1–10).

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.