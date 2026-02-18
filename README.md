# GPT Language Modeling Suite

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
![Repo size](https://img.shields.io/github/repo-size/pablo-reyes8/implementing-gpt)
![Last commit](https://img.shields.io/github/last-commit/pablo-reyes8/implementing-gpt)
![Open issues](https://img.shields.io/github/issues/pablo-reyes8/implementing-gpt)
![Contributors](https://img.shields.io/github/contributors/pablo-reyes8/implementing-gpt)
![Forks](https://img.shields.io/github/forks/pablo-reyes8/implementing-gpt?style=social)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/implementing-gpt?style=social)


This repository contains a clean-room implementation of GPT-style decoder-only language models (GPT-2 and GPT-3 configurations) plus the utilities required to train, evaluate, and deploy them. It is designed as a pedagogical yet production-ready reference: every block is explicitly defined, the training loop exposes the key hyper-parameters (warmup, cosine decay, AdamW with GPT-3 betas, gradient clipping, weight decay), and both notebook- and script-based workflows are supported.

## Why GPT Still Matters

Large decoder-only transformers remain the backbone of modern generative AI workloads. Even midsized GPT-2/-3 models learn powerful representations of language when trained on web corpora. In our experiments, a 256-context GPT-2 “mini” model trained for 10 epochs on a subset of OpenWebText already internalizes conversational English:

> **Prompt:** `What's your name?`  
> **Model:** `What's your name? And if it's not an answer, you could see yourself in the future in order to find yourself.`

Even with tiny parameter counts, cosine LR decay + AdamW betas (0.9, 0.95) stabilize optimization so the model rapidly learns tense agreement, long-range context, and question-to-answer structure.

## Repository Layout

```
.
├── scripts/                 # CLI entrypoints (training, inference, compare, ablations, plotting)
├── src/
│   ├── data/                # Tokenizer + dataset builders
│   ├── inference/           # Sampling utilities
│   ├── model/               # Attention, GPT blocks, GPT2/GPT3 classes
│   ├── research/            # Reusable experiment runner + result exporters
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

Research variants included in this repo:
- `RMSNorm` (`--norm-type rmsnorm`)
- `SwiGLU` feed-forward (`--mlp-type swiglu`)
- RoPE positional encoding (`--pos-encoding rope`)
- PyTorch SDPA backend (`--attention-impl sdpa`)
- Gradient checkpointing (`--gradient-checkpointing`)

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

The project now ships multiple corpus loaders for GPT pretraining research:

- `small` -> OpenWebText-10K (`src/data/load_small_data.py`)
- `large` -> WikiText-103 (`src/data/load_large_data.py`)
- `wikitext2` -> WikiText-2 (`src/data/load_wikitext2_data.py`)
- `tinystories` -> TinyStories (`src/data/load_tinystories_data.py`)
- `openwebtext` -> OpenWebText full (`src/data/load_openwebtext_data.py`)
- `c4_en` -> C4 English (`src/data/load_c4_data.py`)
- `pile` -> The Pile (uncopyrighted) (`src/data/load_pile_data.py`)
- `redpajama` -> RedPajama 1T sample (`src/data/load_redpajama_data.py`)
- `local_jsonl` -> Local JSONL/TXT corpus (`src/data/load_local_jsonl_data.py`)

List available options with:

```bash
python scripts/list_datasets.py
```

Build and inspect dataloaders for any registered option:

```bash
python scripts/build_dataloaders.py --dataset tinystories --preview-batches 2
```

Datasets are not downloaded until you run a loader through `train.py`, `compare_models.py`, or `run_ablations.py`.

For local corpora (`local_jsonl`):

```bash
export LOCAL_TEXT_DATA_PATH=/path/to/corpus.jsonl
export LOCAL_TEXT_DATA_FORMAT=json
export LOCAL_TEXT_FIELD=text
```

## Running the Pipeline

### Training from CLI

```bash
python scripts/train.py \
  --dataset small \
  --model-version gpt3 \
  --model-preset small \
  --block-size 256 \
  --epochs 3 \
  --max-steps 2000 \
  --batch-size 32 \
  --norm-type rmsnorm \
  --mlp-type swiglu \
  --pos-encoding rope \
  --attention-impl sdpa \
  --gradient-checkpointing \
  --val-checking \
  --output-dir checkpoints \
  --run-name gpt3_research_baseline
```

Key switches:

- `--dataset ...` supports OpenWebText, WikiText, TinyStories, C4, Pile, RedPajama, and local corpora.
- `--model-version {gpt2,gpt3}` automatically adjusts weight decay, betas, warmup, gradient clipping, and cosine scheduler settings.
- `--max-steps N` enables fair compute-matched comparisons across architectures.
- `--norm-type`, `--mlp-type`, `--pos-encoding`, `--attention-impl` enable architecture ablations.
- `--gradient-checkpointing` and `--compile` help scale to larger configs on fixed VRAM.
- `--preview-every N` enables teacher-forced previews mid-training (requires tokenizer decode).
- `--amp --amp-dtype fp16` to enable autocast + GradScaler.

All arguments can be inspected via `python scripts/train.py --help`.

### Controlled GPT-2 vs GPT-3 Comparison

```bash
python scripts/compare_models.py \
  --dataset small \
  --models gpt2 gpt3 \
  --seeds 42 43 44 \
  --n-layer 8 \
  --n-head 8 \
  --d-model 512 \
  --epochs 3 \
  --max-steps 2000 \
  --val-checking \
  --experiment-name gpt2_vs_gpt3_equal_budget
```

This writes per-run artifacts plus aggregate `results.jsonl` and `results.csv` for plotting/statistics.

### Architecture Ablations

```bash
python scripts/run_ablations.py \
  --model-version gpt3 \
  --dataset small \
  --ablation-axis norm_type \
  --ablation-values layernorm rmsnorm \
  --seeds 42 43 \
  --n-layer 8 \
  --n-head 8 \
  --d-model 512 \
  --epochs 3 \
  --max-steps 2000 \
  --val-checking \
  --experiment-name norm_ablation_gpt3
```

Typical axes: `norm_type`, `mlp_type`, `pos_encoding`, `attention_impl`, `gradient_checkpointing`, `dropout`.

### Plotting Results

```bash
python scripts/plot_results.py \
  --results research_runs/compare/gpt2_vs_gpt3_equal_budget/results.jsonl \
  --kind compare \
  --metric val_loss_best \
  --output-dir research_runs/plots
```

For ablations:

```bash
python scripts/plot_results.py \
  --results research_runs/ablations/norm_ablation_gpt3/results.csv \
  --kind ablation \
  --metric val_loss_best \
  --output-dir research_runs/plots
```

### Text Generation

```bash
python scripts/generate.py \
  --checkpoint checkpoints/gpt3_research_baseline.last.pt \
  --tokenizer-path owt10k_tokenizer.json \
  --use-ckpt-config \
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

This project is released under the MIT License. See [LICENSE](LICENSE) (or include in your fork) for the full text before redistributing or deploying derived models.
