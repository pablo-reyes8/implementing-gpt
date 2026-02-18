# GPT Research Suite (From Scratch)

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A research-focused implementation of GPT-style decoder-only language models (GPT-2 / GPT-3 style) with:

- clean architecture code from scratch
- reproducible CLI workflows for train/compare/ablation/plot
- multiple dataset loaders (OpenWebText, WikiText, TinyStories, C4, Pile, RedPajama, local corpora)
- YAML-based experiment packaging
- simple Airflow orchestration for professional, reproducible pipelines

## What Is Included

- Core models: `GPT2`, `GPT3` from scratch in `src/model/`
- Research variants:
  - `RMSNorm`
  - `SwiGLU`
  - RoPE positional encoding
  - SDPA attention backend
  - gradient checkpointing
- Research CLIs:
  - `scripts/train.py`
  - `scripts/compare_models.py`
  - `scripts/run_ablations.py`
  - `scripts/plot_results.py`
  - `scripts/run_pipeline_config.py`

## Repository Layout

```text
.
├── configs/                      # YAML configs for train/compare/ablation/plot/pipeline
├── orchestration/
│   └── airflow/
│       ├── dags/                # Airflow DAGs
│       └── README.md            # Airflow usage notes
├── scripts/                     # Project CLIs
├── src/
│   ├── data/                    # Dataset registry + dataloaders
│   ├── inference/               # Generation/sampling utils
│   ├── model/                   # Attention, blocks, GPT2/GPT3
│   ├── research/                # Experiment runner + plotting + pipeline config runner
│   └── training/                # Training loop, optimizer, AMP helpers
├── tests/
├── requirements.txt
└── README.md
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- `matplotlib` is included for plotting.
- Datasets are downloaded only when you run a loader/experiment command.

## Dataset Options

List all registered datasets:

```bash
python scripts/list_datasets.py
```

Current dataset keys:

- `small`: OpenWebText-10K
- `large`: WikiText-103
- `wikitext2`: WikiText-2
- `tinystories`: TinyStories
- `openwebtext`: OpenWebText
- `c4_en`: C4 English
- `pile`: The Pile (uncopyrighted)
- `redpajama`: RedPajama 1T sample
- `local_jsonl`: local JSONL/TXT corpora

Inspect dataloader shapes quickly:

```bash
python scripts/build_dataloaders.py --dataset tinystories --preview-batches 2
```

For local corpora (`local_jsonl`), configure env vars:

```bash
export LOCAL_TEXT_DATA_PATH=/path/to/corpus.jsonl
export LOCAL_TEXT_DATA_FORMAT=json   # json | text
export LOCAL_TEXT_FIELD=text         # only for json
```

## CLI Workflows

### 1. Train

```bash
python scripts/train.py \
  --dataset small \
  --model-version gpt3 \
  --model-preset small \
  --block-size 256 \
  --batch-size 32 \
  --epochs 3 \
  --max-steps 2000 \
  --norm-type rmsnorm \
  --mlp-type swiglu \
  --pos-encoding rope \
  --attention-impl sdpa \
  --gradient-checkpointing \
  --val-checking \
  --run-name gpt3_research_small
```

### 2. Fair GPT-2 vs GPT-3 Comparison

```bash
python scripts/compare_models.py \
  --dataset small \
  --models gpt2 gpt3 \
  --seeds 42 43 44 \
  --n-layer 8 --n-head 8 --d-model 512 \
  --epochs 3 --max-steps 2000 \
  --val-checking \
  --experiment-name gpt2_vs_gpt3_small_fair
```

Outputs include `results.jsonl` and `results.csv`.

### 3. Ablations

```bash
python scripts/run_ablations.py \
  --model-version gpt3 \
  --dataset small \
  --ablation-axis norm_type \
  --ablation-values layernorm rmsnorm \
  --seeds 42 43 \
  --n-layer 8 --n-head 8 --d-model 512 \
  --epochs 3 --max-steps 2000 \
  --val-checking \
  --experiment-name norm_ablation_gpt3_small
```

### 4. Plot Results

Compare plot:

```bash
python scripts/plot_results.py \
  --results research_runs/compare/gpt2_vs_gpt3_small_fair/results.jsonl \
  --kind compare \
  --metric val_loss_best \
  --output-dir research_runs/plots
```

Ablation plot:

```bash
python scripts/plot_results.py \
  --results research_runs/ablations/norm_ablation_gpt3_small/results.jsonl \
  --kind ablation \
  --metric val_loss_best \
  --output-dir research_runs/plots
```

### 5. Generate Text

```bash
python scripts/generate.py \
  --checkpoint checkpoints/gpt3_research_small.last.pt \
  --tokenizer-path owt10k_tokenizer.json \
  --use-ckpt-config \
  --prompt "What's your name?" \
  --strategy topk \
  --top-k 50 \
  --max-new-tokens 64
```

## Config-Driven Reproducibility (YAML)

The repo includes ready-to-run YAML configs:

- `configs/train/gpt3_research_small.yaml`
- `configs/compare/gpt2_vs_gpt3_small_fair.yaml`
- `configs/ablation/norm_ablation_gpt3_small.yaml`
- `configs/plot/compare_plot.yaml`
- `configs/plot/ablation_plot.yaml`
- `configs/pipeline/research_small_repro.yaml`

Run whole pipeline with one command:

```bash
python scripts/run_pipeline_config.py --config configs/pipeline/research_small_repro.yaml
```

Dry-run (only print commands):

```bash
python scripts/run_pipeline_config.py --config configs/pipeline/research_small_repro.yaml --dry-run
```

Run specific step(s):

```bash
python scripts/run_pipeline_config.py \
  --config configs/pipeline/research_small_repro.yaml \
  --step compare \
  --step plot_compare
```

The runner stores execution reports in JSON under `run_dir`.

## Simple Airflow Orchestration

DAG location:

- `orchestration/airflow/dags/gpt_research_pipeline.py`

Default pipeline config used by DAG:

- `configs/pipeline/research_small_repro.yaml`

Environment variables:

```bash
export AIRFLOW_HOME=$PWD/.airflow
export PYTHONPATH=$PWD
export GPT_PIPELINE_CONFIG=$PWD/configs/pipeline/research_small_repro.yaml
export GPT_PYTHON_BIN=python
```

Then trigger DAG `gpt_research_pipeline` from your Airflow instance.

The DAG calls the same project CLIs through `scripts/run_pipeline_config.py`, keeping local and orchestrated behavior aligned.

## Testing

```bash
python3 -m pytest -s --capture=no
```

## References

```bibtex
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  journal={OpenAI Technical Report},
  year={2019}
}

@article{brown2020language,
  title={Language Models are Few-Shot Learners},
  author={Brown, Tom B. and Mann, Benjamin and Ryder, Nick and Subbiah, Melanie and Kaplan, Jared and Dhariwal, Prafulla and others},
  journal={Advances in Neural Information Processing Systems},
  year={2020}
}
```

## License

MIT. See `LICENSE`.
