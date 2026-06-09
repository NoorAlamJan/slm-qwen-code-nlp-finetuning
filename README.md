# Impact of Code on Small Language Models (SLMs)

---

## Overview

This repository contains the full experimental pipeline for my MS thesis investigating how code data in pretraining affects the reasoning, language, and coding capabilities of Small Language Models (SLMs). The study replicates and extends the findings of Aryabumi et al. (2024) using **TinyLlama** as the base model, with controlled data mixture experiments and benchmark-driven evaluation.

The central research questions are:

- Does exposure to code data during fine-tuning improve reasoning on non-code tasks?
- What is the optimal mixture ratio of code vs. natural language data for SLMs?
- How do these effects compare to findings reported for larger models?

---

## Repository Structure

```
.
├── data/                   # Cached datasets (auto-created)
├── cache/                  # HuggingFace model/dataset cache
├── results/                # Training checkpoints and logs
├── slm_thesis.py           # Main training script
├── evaluate.py             # Benchmark evaluation script (coming soon)
├── notebooks/              # Analysis and visualization notebooks
└── README.md
```

---

## Experimental Setup

| Setting | Value |
|---|---|
| Base Model | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| Fine-tuning Method | LoRA (r=16, α=32) |
| Max Sequence Length | 512 tokens |
| Effective Batch Size | 16 (1 × 16 grad accum) |
| Learning Rate | 2e-4 |
| Epochs | 1 |
| Hardware | Apple M-series (MPS) / CUDA |

### Data Mixtures

| Run | Natural Language | Code |
|---|---|---|
| Baseline | 100% | 0% |
| Code-only | 0% | 100% |
| Mixed-30 | 70% | 30% |
| Mixed-50 | 50% | 50% |

Datasets used: `roneneldan/TinyStories`, `Salesforce/wikitext`, `codeparrot/github-code`

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/NoorAlamJan/slm-code-impact
cd slm-code-impact
```

### 2. Set up environment (Python 3.11 recommended)

```bash
python3.11 -m venv ml_env
source ml_env/bin/activate
pip install --upgrade pip
pip install transformers datasets peft accelerate bitsandbytes tqdm
```

### 3. Set your HuggingFace token

```bash
export HF_TOKEN=your_token_here
```

### 4. Run training

```bash
python slm_thesis.py
```

The script auto-detects your hardware:
- **NVIDIA GPU** → CUDA + bfloat16
- **Apple Silicon** → MPS + float32
- **CPU** → float32 fallback

---

## Results

*To be updated as experiments complete.*

| Run | MMLU | HumanEval | HellaSwag | Notes |
|---|---|---|---|---|
| Baseline | — | — | — | NL only |
| Code-only | — | — | — | Code only |
| Mixed-30 | — | — | — | 70/30 NL/Code |
| Mixed-50 | — | — | — | 50/50 NL/Code |

---

## Author

**Noor Alam**  
Data Science
[GitHub](https://github.com/NoorAlamJan) · [LinkedIn](https://linkedin.com/in/noor-alam-0a7122209)
