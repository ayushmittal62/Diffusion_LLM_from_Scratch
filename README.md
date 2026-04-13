<div align="center">

# 🌀 Diffusion Language Model from Scratch

### A masked discrete diffusion Transformer trained on TinyStories — built entirely from scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vYkaeq6v_5s9-8OTjCOB8ij_K8KFS70T)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-bf16%20supported-76B900?style=flat-square&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

<br/>

**Early diffusion step — noisy output being denoised into text**

![Diffusion Inference GIF]![download](https://github.com/user-attachments/assets/63ac7481-9505-451a-9615-8be638c1846a)



**After full denoising — clean output**

![Diffusion Final Output]![download (1)](https://github.com/user-attachments/assets/79488939-45d8-4691-a407-03d09b01c566)

</div>

---

## 📖 What is This?

This project implements a **Discrete Diffusion Language Model** — a fundamentally different approach to text generation compared to the GPT-style autoregressive models you're used to.

Instead of predicting tokens **left-to-right one at a time**, a diffusion LM generates by **iteratively denoising a fully masked sequence in parallel**:

```
Step 0:   [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]
Step 16:  Once   [MASK] a     [MASK] there  [MASK] [MASK] story
Step 32:  Once   upon   a     time   there  was    a      story
Step 64:  Once   upon   a     time,  there  was    a      little   story...
```

This gives the distinctive "shimmer" effect — text that looks like it's being *carved out of noise* — which is what you see in the GIFs above.

**This is an educational, from-scratch implementation** — every component (tokenizer, architecture, training loop, sampler, GIF renderer) is written by hand with no pretrained weights.

---

## 🧠 Core Concepts

### Autoregressive LMs (GPT-style) vs. Diffusion LMs

| | Autoregressive LM | Diffusion LM (this project) |
|---|---|---|
| **Generation direction** | Left → Right, token by token | All positions in parallel |
| **Architecture** | Causal (masked) Transformer | Bidirectional Transformer |
| **Generation speed** | O(n) sequential steps | Fixed diffusion steps T |
| **Key operation** | Predict next token | Denoise masked tokens |
| **Cool factor** | Streaming text | Text materializing from noise ✨ |

### The Diffusion Process

**Forward process (training):** Corrupt a clean sequence `x₀` by replacing a fraction `t/T` of tokens with `[MASK]` → produces noisy `x_t`.

**Reverse process (inference):** Start from all `[MASK]`, iteratively predict and commit the most confident tokens, re-mask the uncertain ones. Repeat for `T` steps.

**Training objective:** Cross-entropy loss, computed **only at masked positions**:

```
loss = CrossEntropy(model(x_t, t), x₀)   at masked positions only
```

---

## 🏗️ Architecture

### DiffusionTransformerLM

A bidirectional Transformer with an added time-step conditioning mechanism:

```
Input IDs  ──► Token Embeddings  ─┐
                                   ├──► + Position Embeddings
Timestep t ──► Time Embedding   ──┘
                    │
                    ▼
          TransformerEncoder
          (N bidirectional layers)
                    │
                    ▼
              LayerNorm
                    │
                    ▼
           LM Head (linear, weight-tied to token embs)
                    │
                    ▼
            Logits [B, L, V]
```

Key design choices:
- **Bidirectional attention** (not causal) — the model sees the full sequence including masked positions
- **Timestep embedding** — sinusoidal, added to every token position so the model knows "how noisy" the input is
- **Weight tying** — LM head shares weights with token embedding matrix (reduces params, improves generalization)
- **Pre-norm** (`norm_first=True`) — LayerNorm before attention/FFN for training stability
- **GELU activations** in feed-forward layers

### Tokenizer

A **Byte-level BPE tokenizer trained from scratch** on the TinyStories corpus — no pretrained GPT-2/BERT tokenizer used.

Special tokens:

| Token | Purpose |
|---|---|
| `[PAD]` | Padding |
| `[UNK]` | Unknown token |
| `[BOS]` | Begin of sequence |
| `[EOS]` | End of sequence |
| `[MASK]` | Diffusion noise token |
| `<\|user\|>` `<\|assistant\|>` `<\|end\|>` | Chat formatting |

---

## ⚙️ Configuration Profiles

The notebook ships with two pre-tuned run profiles:

| Parameter | `quick` | `budget_100` |
|---|---|---|
| Training examples | 50,000 | 1,000,000 |
| Vocabulary size | 8,000 | 26,000 |
| Model dim (`d_model`) | 384 | 512 |
| Layers (`n_layers`) | 6 | 10 |
| Attention heads | 6 | 8 |
| Feed-forward dim | 1,536 | 2,048 |
| Diffusion steps T | 64 | 128 |
| Training steps | 2,000 | 200,000 |
| Batch size | 32 | 32 |
| Gradient accumulation | 1 | 2 |
| Learning rate | 3e-4 | 2e-4 |
| Warmup steps | 200 | 1,000 |

The `budget_100` run was trained on an **NVIDIA B200** GPU with `torch 2.8.0+cu128` and `bf16` mixed precision.

---

## 🗂️ Notebook Structure

```
Section 0  — Choose run profile (quick / budget_100)
Section 1  — Install dependencies
Section 2  — Load TinyStories dataset (HuggingFace)
Section 3  — Train Byte-level BPE tokenizer from scratch
Section 4  — Build DiffusionTransformerLM architecture
Section 5  — Create TokenBlockDataset with chat formatting
Section 6  — Implement diffusion corruption + training loss
Section 7  — Full training loop (Accelerate + cosine LR + AdamW)
Section 8  — Diffusion sampler (progressive unmasking with confidence)
Section 9  — Render terminal-style inference GIF
Section 10 — OPTIONAL: Tour of the dLLM reference repo
```

---

## 🔬 Implementation Details

### Corruption Schedule

A **linear masking schedule** — at diffusion step `t`, a fraction `t/T` of non-special tokens are replaced with `[MASK]`:

```python
def mask_ratio_schedule(t, T):
    return t.float() / float(T)
```

BOS, EOS, and PAD tokens are **never masked**.

### Training Loop

- **Optimizer:** AdamW (`weight_decay=0.1`)
- **Scheduler:** Cosine decay with linear warmup
- **Mixed precision:** bf16 via HuggingFace Accelerate
- **Checkpoint:** saves `model.pt` + `config.json` + `tokenizer/` at end of training

### Diffusion Sampler

```
1. Encode prompt → fix prompt tokens
2. Initialize answer region as all [MASK]
3. For t = T → 1:
     a. Forward pass → logits at masked positions
     b. Sample top-k tokens
     c. Keep top-confidence fraction (1 - t/T) of predictions
     d. Re-mask the rest
4. Decode final sequence
```

The result: confident tokens "lock in" early, uncertain positions keep getting revised — which creates the visible denoising effect in the GIF.

### Chat Format

Each training sample is formatted as:

```
<|user|>
Write a short story.
<|assistant|>
{story text}
<|end|>
```

Tokens are packed into fixed-length blocks (`SEQ_LEN=256`) via a streaming `IterableDataset`.

---

## 🎬 Inference GIF

The notebook renders a **terminal-style animated GIF** showing the progressive denoising:

- Early frames: output is random garbage / `[MASK]` tokens
- Middle frames: structure emerges, words start appearing
- Final frames: coherent text has materialized

Two styles are exported:
- `inference.gif` — plain dark terminal style
- `inference_cool.gif` — cyan/green syntax-highlighted terminal aesthetic

---

## 🚀 Getting Started

### Requirements

```bash
pip install datasets tokenizers accelerate tqdm numpy einops imageio pillow transformers hf_transfer
```

- Python 3.10+
- PyTorch 2.x
- CUDA GPU recommended (Google Colab T4/V100/A100/B200 all work)

### Run in Colab

1. Open the notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vYkaeq6v_5s9-8OTjCOB8ij_K8KFS70T)
2. Go to **Runtime → Change runtime type → GPU**
3. Set `RUN_MODE = "quick"` for a fast end-to-end test, or `"budget_100"` for better quality
4. Run all cells

### Outputs

After training, the notebook saves:

```
checkpoints/
└── final/
    ├── model.pt        # Model weights
    ├── config.json     # DiffusionLMConfig
    └── tokenizer/      # BPE tokenizer files

inference.gif           # Terminal-style denoising animation
inference_cool.gif      # Styled version with color
```

---

## 🗺️ Concept Map: This Notebook ↔ dLLM Library

For those who want to scale this up, here's how the scratch implementations map to the production [`dLLM`](https://github.com/ZHZisZZ/dllm) library:

| This Notebook | dLLM Equivalent |
|---|---|
| `corrupt_with_mask(...)` | `MDLMTrainer` (masked noise corruption) |
| `diffusion_loss(...)` | MDLM training objective |
| `diffusion_generate(...)` | `MDLMSampler` (iterative unmasking) |
| Multi-turn chat loop | `examples/llada/chat.py` |

---

## 📚 References & Further Reading

- [**MDLM: Masked Diffusion Language Models**](https://arxiv.org/abs/2406.07524) — Sahoo et al., 2024
- [**LLaDA: Large Language Diffusion with mAsking**](https://arxiv.org/abs/2502.09992) — the architecture this closely mirrors
- [**TinyStories Dataset**](https://huggingface.co/datasets/roneneldan/TinyStories) — Eldan & Li, 2023
- [**dLLM reference repo**](https://github.com/ZHZisZZ/dllm) — production MDLM trainer + sampler

---

## 👨‍💻 Author

Built by **Ayush Mittal** — ML & Software Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ayush-mittal629/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/ayushmittal62)
[![Gmail](https://img.shields.io/badge/Gmail-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:ayushmittal629@gmail.com)

---

<div align="center">

*If this helped you understand diffusion LMs, drop a ⭐ — it keeps me building.*

</div>
