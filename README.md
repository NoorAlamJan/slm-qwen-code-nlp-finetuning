# Impact of Code on Small Language Models (SLMs)

---

## Overview

This repository contains the full experimental pipeline  investigating how **code data mixture ratios** affect the capabilities of Small Language Models (SLMs). Using **Qwen2.5-0.5B** as the base model, we run controlled fine-tuning experiments across 7 dataset mixtures and evaluate the impact on language understanding, reasoning, and coding ability.

The central research questions are:
- Does exposure to code data during fine-tuning improve reasoning on non-code tasks?
- What is the optimal mixture ratio of code vs. natural language data for SLMs?
- How do these effects compare to findings reported for larger models?

---

## Repository Structure

```
.
├── data/                     # Cached datasets (auto-created)
├── cache/                    # HuggingFace model/dataset cache
├── results/                  # Evaluation results (JSON)
├── models/                   # Fine-tuned checkpoints
│   └── qwen25_500m/          # Per-mixture model outputs
├── 02_qwen25_500m.ipynb      # Main training & evaluation notebook
└── README.md
```

---

## Experimental Setup

| Setting | Value |
|---|---|
| Base Model | Qwen/Qwen2.5-0.5B |
| Fine-tuning Method | Full fine-tune + LoRA (r=16, α=32) |
| Max Sequence Length | 512 tokens |
| Effective Batch Size | 16 (batch=1 × grad accum=16) |
| Learning Rate | 2e-4 |
| Epochs | 1 |
| Hardware | Apple Silicon (MPS / float16) / CUDA (bfloat16) |

---

## Datasets

| ID | Description | Source |
|---|---|---|
| D1 | NLP text (10k) | DKYoon/SlimPajama-6B |
| D2 | Python code (10k) | bigcode/the-stack |
| D3 | Java code (10k) | bigcode/the-stack |
| D4 | C++ code (10k) | bigcode/the-stack |
| D5 | Python ∪ Java ∪ C++ balanced (10k) | bigcode/the-stack |

### Mixture Datasets

| ID | Text % | Code % |
|---|---|---|
| M1 | 100% | 0% |
| M2 | 0% | 100% |
| M3 | 70% | 30% |
| M4 | 30% | 70% |
| M5 | 75% | 25% |
| M6 | 25% | 75% |
| M7 | 50% | 50% |

---

## Experiments

13 checkpoints trained in total:

- **NFT** — base model, no fine-tuning (evaluation only)
- **7 full fine-tune runs** — one per mixture (M1–M7)
- **4 pure-code runs** — Python, Java, C++, Union (D2–D5)
- **1 LoRA run** — M1 (text-only) with LoRA adapters

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/NoorAlamJan/slm-qwen-code-nlp-finetuning.git
cd slm-qwen-code-nlp-finetuning
```

### 2. Set up environment (Python 3.11 recommended)
```bash
python3.11 -m venv ml_env
source ml_env/bin/activate
pip install transformers datasets peft accelerate lm-eval bitsandbytes tqdm
```

### 3. Set your HuggingFace token
```bash
export HF_TOKEN=your_token_here
```

> You need to accept the [bigcode/the-stack](https://huggingface.co/datasets/bigcode/the-stack) terms on HuggingFace before downloading.

### 4. Run the notebook
Open `02_qwen25_500m.ipynb` in VS Code or JupyterLab and run cells top to bottom. The notebook auto-detects your hardware (MPS / CUDA / CPU).

---

## Results

*To be updated as experiments complete.*

| Checkpoint | MMLU | HumanEval | HellaSwag |
|---|---|---|---|
| NFT (base) | — | — | — |
| M1 (text 100%) | — | — | — |
| M2 (code 100%) | — | — | — |
| M3 (70/30) | — | — | — |
| M4 (30/70) | — | — | — |
| M5 (75/25) | — | — | — |
| M6 (25/75) | — | — | — |
| M7 (50/50) | — | — | — |
| LoRA (M1) | — | — | — |

---

## Author

**Noor Alam**  
Data Science  
[GitHub](https://github.com/NoorAlamJan) · [LinkedIn](https://linkedin.com/in/noor-alam-0a7122209)
