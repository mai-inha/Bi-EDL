# Bi-EDL: Bidirectional Evidential Deep Learning for Medical Image Classification

Bi-EDL is a uncertainty-aware chest X-ray classification framework that combines **bidirectional Multiple-Choice Question (MCQ) training** with **Evidential Deep Learning (EDL)** on top of the [CARZero](https://github.com/sqrtsqrtsqrt/CARZero) vision-language backbone. The model learns to classify 14 thoracic pathologies from the NIH ChestXray-14 dataset while producing calibrated, interpretable uncertainty estimates.

---

## Overview

Standard vision-language models for medical imaging produce overconfident predictions without meaningful uncertainty quantification. Bi-EDL addresses this by:

1. **Bidirectional MCQ Training** — Jointly training image-to-text (i2t) and text-to-image (t2i) selection tasks to strengthen cross-modal alignment.
2. **Positive-Negative Contrast (PNC)** — Each disease has a positive prompt (*"There is Atelectasis."*) and a negative prompt (*"There is no Atelectasis."*). Both logits are used at inference, with the PNC score derived from their softmax ratio.
3. **EDL Loss** — Models predictions as a Dirichlet distribution over binary (positive/negative) outcomes, enabling closed-form uncertainty computation: `U = 2 / (α_pos + α_neg)`.
4. **Uncertainty Benchmarking** — At inference, five uncertainty methods (MSP, Energy, MaxLogit, EDL, ODIN) are evaluated via Area Under the Risk-Coverage (AURC) curve.

---

## Architecture

```
Input Image ─────► ViT-B/16 Encoder ──────────────────────────────┐
                                                                   ▼
                                              Bidirectional Cross-Attention Fusion
                                               (i2t + t2i dual attention modules)
                                                                   │
Input Text ──────► BioClinicalMPBERT ──────────────────────────────┘
(28 prompts:                                                        │
 14 positive                                               (B, T=28) similarity matrix
 14 negative)                                                       │
                                              ┌────────────────────┼────────────────────┐
                                              ▼                    ▼                    ▼
                                        i2t MCQ Loss         t2i MCQ Loss          EDL Loss
                                      (image→text)         (text→image)     (Dirichlet uncertainty)
```

### Loss Function

Training uses a warmup-weighted combination of all three objectives:

```
loss = (1 - λ) × [w × L_i2t + (1 - w) × L_t2i] + λ × L_edl
```

- `λ` linearly increases from 0 to 1 over `cfg.train.lam` epochs — MCQ dominates early, EDL takes over later.
- `w = cfg.train.weight` (default: 0.5) balances i2t vs. t2i MCQ.
- `L_edl = L_match + edl_weight × L_KL` where `L_match` maximizes evidence for the correct class and `L_KL` regularizes toward a uniform Dirichlet.

### MCQ Formulation

**i2t MCQ** (Image → Text): Given an image, identify the correct disease prompt from 3 candidates (1 correct + 2 distractors drawn from semantically opposite prompts).

**t2i MCQ** (Text → Image): Given a disease prompt, identify the matching image from 3 candidates (1 correct + 2 images where that prompt does not apply).

Both tasks are randomly shuffled, and targets track the post-shuffle answer position.

---

## Dataset

[NIH ChestXray-14](https://nihcc.app.box.com/v/ChestXray-NIHCC) — 112,120 frontal-view chest X-ray images labeled with 14 thoracic diseases.

| Split | Source |
|---|---|
| Train | All images **not** in `ChestXray-14/test_list.txt` (90% train / 10% val split) |
| Test | Official test list from `ChestXray-14/test_list.txt` |

**Diseases:** Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia

---

## Installation

```bash
# Clone and enter the repository
git clone <repo-url>
cd Bi-EDL

# Install dependencies (PyTorch, Lightning, CARZero, etc.)
pip install torch torchvision
pip install pytorch-lightning transformers omegaconf
pip install scikit-learn pandas tqdm wandb

# Install CARZero backbone
pip install -e ./CARZero   # or follow CARZero's own install instructions
```

---

## Usage

### Training

```bash
bash train.sh
```

Or directly:

```bash
python train.py \
    --data_path /path/to/NIH \
    --cfg_path  configs/chest14_finetuning_llm_dqn_wo_self_atten_mlp_gl_Bi_EDL.yaml
```

Training logs and checkpoints are saved under `logs/<project>/<name>/<timestamp>/`.  
The best checkpoint (by `val/mean_auroc`) is saved at `logs/.../checkpoints/best/best_model.ckpt`.  
Metrics are tracked with Weights & Biases.

### Inference & Uncertainty Evaluation

```bash
bash inference.sh
```

Or directly:

```bash
python inference.py \
    --ckpt_path  checkpoints/best_model.ckpt \
    --cfg_path   configs/chest14_finetuning_llm_dqn_wo_self_atten_mlp_gl_Bi_EDL.yaml \
    --data_path  /path/to/NIH \
    --method     msp energy maxlogit edl odin \
    --device     cuda:0 \
    --batch_size 128 \
    --coverage   0.9
```

**Output:**

1. **Classification performance table** — Positive, Negative, and PNC AUROC per disease class.
2. **Uncertainty comparison table** — AURC, Risk@90, and R(1) for each selected method.

| Argument | Default | Description |
|---|---|---|
| `--ckpt_path` | required | Path to trained `.ckpt` checkpoint |
| `--cfg_path` | required | OmegaConf `.yaml` config file |
| `--data_path` | required | NIH dataset root directory |
| `--method` | all | Uncertainty methods: `msp energy maxlogit edl odin` |
| `--device` | `cuda:0` | Compute device |
| `--batch_size` | `32` | Inference batch size |
| `--odin_eps` | `0.001` | ODIN input perturbation magnitude |
| `--coverage` | `0.9` | Coverage point for Risk@coverage metric |
| `--per_label` | flag | Print per-class AURC breakdown |

---

## Uncertainty Methods

Bi-EDL benchmarks five uncertainty scoring methods. All scores are higher-is-more-uncertain.

| Method | Score formula | Notes |
|---|---|---|
| **MSP** | `1 - max(p_pos, p_neg)` | Maximum Softmax Probability baseline |
| **Energy** | `-log(exp(p) + exp(n))` | Free energy of the logit pair |
| **MaxLogit** | `-max(p, n)` | Logit-level analogue of MSP |
| **EDL** | `2 / (softplus(p) + 1 + softplus(n) + 1)` | Dirichlet vacuity; model's native uncertainty |
| **ODIN** | `1 - max_softmax(perturbed input)` | Input-preprocessing perturbation (Liang et al.) |

---

## Configuration

Key parameters in `configs/chest14_finetuning_llm_dqn_wo_self_atten_mlp_gl_Bi_EDL.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `train.weight` | `0.5` | Balance between i2t and t2i MCQ loss |
| `train.lam` | `50` | Number of epochs for EDL warmup ramp |
| `train.edl_weight` | `0.1` | KL regularization weight inside EDL loss |
| `train.seed` | `14` | Random seed |
| `lightning.trainer.lr` | `1e-5` | Learning rate |
| `lightning.trainer.precision` | `16-mixed` | Mixed-precision training |
| `model.CARZero.multi` | `true` | Use separate i2t/t2i fusion modules |
| `model.text.bert_type` | `Laihaoran/BioClinicalMPBERT` | Text encoder |
| `freeze.image/text/fusion` | `false` | Which modules to freeze during fine-tuning |

---

## Repository Structure

```
Bi-EDL/
├── train.py                    # Training entry point
├── inference.py                # Inference + uncertainty evaluation
├── utils.py                    # Metrics (AUROC, PNC, temperature scaling)
├── train.sh / inference.sh     # Convenience shell scripts
├── configs/
│   └── chest14_finetuning_llm_dqn_wo_self_atten_mlp_gl_Bi_EDL.yaml
├── finetune/
│   ├── finetuning_lightening.py   # MCQEDLLightModel (training logic)
│   ├── finetuning_dm.py           # NIHDataModule (data pipeline)
│   └── finetuning_dataset.py      # Dataset class
├── ChestXray-14/
│   └── test_list.txt              # Official NIH test split
├── checkpoints/                   # Trained model checkpoints
└── logs/                          # WandB logs and saved configs
```

---

## Evaluation Metrics

**Classification:** AUROC computed separately for positive prompts (disease presence), negative prompts (disease absence), and the PNC-fused prediction.

**Uncertainty:** Area Under the Risk-Coverage (AURC) curve — samples are rejected in ascending order of confidence; a lower AURC indicates that the model correctly abstains on hard/wrong predictions.

- **AURC**: Integral of the risk-coverage curve (lower is better).
- **R(1)**: Risk at full coverage (= error rate, equivalent to 1 - accuracy).
- **Risk@90**: Risk when covering 90% of samples.

---

## Citation

If you use Bi-EDL in your research, please cite the CARZero backbone and EDL literature upon which it is built.
