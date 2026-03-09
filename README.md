# Quantized Diffusion Models — DVC Pipeline

A modular, reproducible pipeline for training and evaluating **quantized diffusion models** on CIFAR-10.  
Implements **Post-Training Quantization (PTQ)**, **Quantization-Aware Training (QAT)** with LSQ, and **Timestep-Aware Mixed Precision** — all orchestrated by [DVC](https://dvc.org/).

---

## 📂 Project Structure

```
NNresearch/
├── src/
│   ├── config.py            # Shared config (reads params.yaml)
│   ├── data.py              # CIFAR-10 download & preparation
│   ├── model.py             # UNet / pretrained HF model creation
│   ├── quantization.py      # QuantConv2d, LSQ, TimestepAware, apply helpers
│   ├── train.py             # Training, sampling, baseline evaluation
│   ├── ptq.py               # PTQ experiments (8/4/1-bit)
│   ├── qat.py               # QAT 1-bit fine-tuning
│   ├── timestep_aware.py    # Timestep-aware mixed-precision experiments
│   └── evaluate.py          # Aggregate all results → final CSV
├── params.yaml              # All tuneable hyperparameters
├── dvc.yaml                 # DVC pipeline definition
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md                # This file
```

---

## 🚀 Quick Start

### 1. Clone & install dependencies

```bash
git clone <your-repo-url>
cd NNresearch
pip install -r requirements.txt
```

### 2. Initialize DVC (first time only)

```bash
dvc init
```

### 3. Run the full pipeline

```bash
dvc repro
```

This executes all stages in order:

```
prepare_data → train_baseline → ptq ─┐
                                      ├→ timestep_aware → evaluate
                                qat ──┘
```

### 4. Run a single stage

```bash
dvc repro <stage_name>
# e.g. dvc repro ptq
```

---

## 🔧 Pipeline Stages

| Stage | Script | Description |
|-------|--------|-------------|
| `prepare_data` | `src/data.py` | Downloads CIFAR-10 to `data/` |
| `train_baseline` | `src/train.py` | Loads pretrained DDPM (or trains from scratch), evaluates baseline FID |
| `ptq` | `src/ptq.py` | Post-Training Quantization at 8, 4, and 1-bit |
| `qat` | `src/qat.py` | Quantization-Aware Training with 1-bit LSQ |
| `timestep_aware` | `src/timestep_aware.py` | Timestep-dependent mixed precision (1/4/8-bit) |
| `evaluate` | `src/evaluate.py` | Aggregates all results into `quantization_final_results.csv` |

---

## ⚙️ Configuration

All hyperparameters live in **`params.yaml`**. DVC tracks changes to these parameters and will re-run only the affected stages when you modify them.

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_pretrained` | `true` | Use `google/ddpm-cifar10-32` or train from scratch |
| `batch_size` | `32` | Training / calibration batch size |
| `epochs` | `500` | Training epochs (only if `use_pretrained: false`) |
| `qat_epochs` | `20` | QAT fine-tuning epochs |
| `fid_num_samples` | `5000` | Number of generated images for FID evaluation |

---

## 📊 Outputs

All outputs are saved to `outputs/`:

| File | Description |
|------|-------------|
| `baseline_model.pth` | Baseline model weights |
| `baseline_metrics.json` | Baseline FID, latency, VRAM, model size |
| `ptq_results.json` | PTQ metrics for 8/4/1-bit |
| `ptq_Xbit.pth` | PTQ model checkpoints |
| `qat_results.json` | QAT 1-bit metrics |
| `qat_1bit.pth` | QAT model checkpoint |
| `timestep_aware_results.json` | TA experiment results |
| `quantization_final_results.csv` | **Final comparison table** |

---

## 📝 Methods

### Post-Training Quantization (PTQ)
Replaces Conv2d layers with quantized versions using symmetric uniform quantization. Supports 8-bit, 4-bit, and 1-bit (binary) weight quantization with optional activation quantization via calibration.

### Quantization-Aware Training (QAT)
Uses **Learned Step Size Quantization (LSQ)** with learnable scale parameters and straight-through estimators. Includes gradient clipping for training stability.

### Timestep-Aware Mixed Precision
Dynamically adjusts bit-width based on the diffusion timestep:
- **t/T < 0.3** → 1-bit (early denoising steps need less precision)
- **0.3 ≤ t/T < 0.7** → 4-bit
- **t/T ≥ 0.7** → 8-bit (late steps need more precision for fine details)

---

## 🔬 Requirements

- **Python** ≥ 3.8
- **GPU** recommended (CUDA) for training and evaluation
- See `requirements.txt` for full dependency list

---
