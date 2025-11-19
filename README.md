# GPT Language Modeling Suite

This repository contains a clean-room implementation of GPT-style decoder-only language models (GPT-2 and GPT-3 configurations) plus the utilities required to train, evaluate, and deploy them. It is designed as a pedagogical yet production-ready reference: every block is explicitly defined, the training loop exposes the key hyper-parameters (warmup, cosine decay, AdamW with GPT-3 betas, gradient clipping, weight decay), and both notebook- and script-based workflows are supported.

## Why GPT Still Matters

Large decoder-only transformers remain the backbone of modern generative AI workloads. Even midsized GPT-2/-3 models learn powerful representations of language when trained on web corpora. In our experiments, a 256-context GPT-2 “mini” model trained for 10 epochs on a subset of OpenWebText already internalizes conversational English:

> **Prompt:** `What's your name?`  
> **Model:** `What's your name? And if it's not an answer, you could see yourself in the future in order to find yourself.`

Even with tiny parameter counts, cosine LR decay + AdamW betas (0.9, 0.95) stabilize optimization so the model rapidly learns tense agreement, long-range context, and question-to-answer structure.

## Repository Layout

```
.
├── scripts/                 # CLI entrypoints (training & inference)
├── src/
│   ├── data/                # Tokenizer + dataset builders
│   ├── inference/           # Sampling utilities
│   ├── model/               # Attention, GPT blocks, GPT2/GPT3 classes
│   └── training/            # Optimizer presets, schedulers, main loop, AMP helpers
├── tests/                   # Pytest suite covering blocks, models, schedulers, training loop
├── training_showcase/       # Jupyter notebooks with exploratory runs
├── requirements.txt
├── Dockerfile
└── README.md
```

## Architecture Reference

Official GPT literature specifies tiered configurations. We provide representative presets (you can override them via the CLI):


<div align="center">

### **GPT-2 Tier**

| **Model** | **Layers** | **Heads** | **d<sub>model</sub>** | **Params** |
|----------|-----------:|-----------:|-----------------------:|-----------:|
| Small    | 12 | 12 | 768  | 117M |
| Medium   | 24 | 16 | 1024 | 345M |
| Large    | 36 | 20 | 1280 | 774M |
| XL       | 48 | 25 | 1600 | 1.5B |

</div>

---

<div align="center">

### **GPT-3 Tier**

| **Model** | **Layers** | **Heads** | **d<sub>model</sub>** | **Params** |
|----------|-----------:|-----------:|-----------------------:|-----------:|
| Ada      | 12 | 12 | 768   | 350M |
| Babbage  | 24 | 16 | 1024  | 1.3B |
| Curie    | 48 | 32 | 2048  | 6.7B |
| Davinci  | 96 | 96 | 12288 | 175B |

</div>

---

Translating these tiers into code is straightforward; for example, a mid-sized GPT-2 using our modules:

```python
model = GPT2(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layer=12,
    n_head=12,
    d_model=768,
    dropout=0.1,
).to(device)
```

An equivalent GPT-3 “Curie”-style configuration in this repo looks like:

```python
model = GPT3(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layer=48,
    n_head=32,
    d_model=2048,
    dropout=0.1,
    mlp_expansion=4,
    resid_dropout=0.1,
).to(device)
```

All parameters are exposed through `scripts/train.py`, so you can mix-and-match presets with your hardware budget.

## Installation

> **Note:** Training GPT-3-sized models is computationally expensive. Expect to need multiple high-memory GPUs for anything beyond the smallest configs.

1. **Python environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   On Linux GPU machines use the appropriate `torch` wheel (e.g. `pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.1`).

2. **Docker (optional)**
   ```bash
   docker build -t gpt-suite .
   docker run --gpus all -it --rm -v $PWD:/app gpt-suite bash
   ```
   The Dockerfile installs `requirements.txt`, sets `PYTHONPATH=/app`, and defaults to exposing the CLI scripts. Mount your dataset/tokenizer artifacts into `/app` when launching containers.

## Datasets

Two dataset modules are provided:

- **OpenWebText-10k (`src/data/load_small_data.py`)** – mirrors the GPT-2 training distribution with ~10k documents for quick experiments.
- **WikiText-103 (`src/data/load_large_data.py`)** – a larger, high-quality corpus suitable for scaling to deeper models.

Each loader handles tokenizer training/loading, dataset chunking, and dataloader creation.

## Running the Pipeline

### Training from CLI

```bash
python scripts/train.py \
  --dataset small \
  --model-version gpt3 \
  --block-size 256 \
  --n-layer 8 \
  --n-head 8 \
  --d-model 512 \
  --epochs 10 \
  --batch-size 32 \
  --val-checking \
  --output-dir checkpoints \
  --ckpt-name gpt3-mini.pt
```

Key switches:

- `--dataset {small,large}` toggles OpenWebText-10k vs WikiText-103 loaders.
- `--model-version {gpt2,gpt3}` automatically adjusts weight decay, betas, warmup, gradient clipping, and cosine scheduler settings.
- `--preview-every N` enables teacher-forced previews mid-training (requires tokenizer decode).
- `--amp --amp-dtype fp16` to enable autocast + GradScaler.

All arguments can be inspected via `python scripts/train.py --help`.

### Text Generation

```bash
python scripts/generate.py \
  --checkpoint checkpoints/gpt3-mini.pt \
  --tokenizer-path owt10k_tokenizer.json \
  --model-version gpt3 \
  --prompt "What's your name?" \
  --strategy topk \
  --top-k 50 \
  --max-new-tokens 64
```

Choose `--strategy greedy` for deterministic responses or adjust `--temperature` for higher entropy.

### Inference Utilities

Advanced sampling helpers live in `src/inference/generate_text.py` (top-p sampling, beam search, perplexity computation) and a verbose debugging interface sits in `src/inference/interactive_generate_debug.py`.

## Testing

Run the test suite (capture disabled to avoid tmpfs issues in some shells):

```bash
pytest -s --capture=no
```

Coverage summary:
- **Model blocks**: shape preservation + gradient flow.
- **Model classes**: weight tying and loss output.
- **Schedulers**: warmup/cosine behavior and preset overrides.
- **Training loop**: executes a CPU epoch and writes checkpoints.

## Contribution Guidelines

1. Fork / branch, keep commits atomic, and run `pytest -s --capture=no` before opening a PR.
2. Document architectural changes (new blocks, schedulers, etc.) and cite relevant papers.
3. Avoid committing large checkpoints or tokenizer artifacts—update `.gitignore` if needed.

## Disclaimer

Training GPT-scale models is resource-intensive. The provided scripts default to small/medium footprints, but replicating full GPT-3 tiers requires massive compute, careful data curation, and safety considerations. Use responsibly, respect dataset licenses, and ensure downstream deployments comply with ethical and legal standards.

---

If you use this codebase in academic or industrial work, please cite the original GPT-2 and GPT-3 papers and reference this repository in your acknowledgements. Happy modeling!

## References

```
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  journal={OpenAI Technical Report},
  year={2019}
}

@article{brown2020language,
  title={Language Models are Few-Shot Learners},
  author={Brown, Tom B. and Mann, Benjamin and Ryder, Nick and Subbiah, Melanie and Kaplan, Jared and Dhariwal, Prafulla and {others}},
  journal={Advances in Neural Information Processing Systems},
  year={2020}
}
```

## License

This project is released under the MIT License. See `LICENSE` (or include in your fork) for the full text before redistributing or deploying derived models.
