# LLM Energy Benchmark (WSL2 + NVIDIA Edition)

Automated test framework for measuring the energy consumption of Docker-hosted
language models – optimized for **WSL2 with NVIDIA GPU**.

Measurement via **`nvidia-smi`** (GPU power draw in watts, every 500 ms),
no RAPL, no eBPF, no root privileges required.

---

## Prerequisites

| What | Where |
|---|---|
| Windows 11 + WSL2 | NVIDIA driver ≥ 530 on the Windows host |
| Python 3.10+ in WSL2 | `python3 --version` |
| Docker (Desktop or native) | `docker info` |
| nvidia-container-toolkit | for GPU passthrough in Docker |
| `nvidia-smi` in WSL2 | should be available automatically |

### Install nvidia-container-toolkit (one-time)
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo service docker restart
```

---

## Quick Start

```bash
# 1. One-time setup
bash setup.sh

# 2. Start benchmark
source .venv/bin/activate
python3 benchmark.py --models gemma3:4b llama3.2:3b phi4-mini:3.8b --runs 3

# 3. Generate plots (for current dataset)
python3 plot_results.py {timestamp}
```
### Reasoning-control run (see "Reasoning-Control Extension" below)
```bash
python3 benchmark.py \
  --models qwen3.5:2b gemma4:e2b \
  --prompts-file prompts_with_reasoning_tier.txt \
  --reasoning-modes on off \
  --models-config models_config.json \
  --gold-answers reasoning_prompts.csv \
  --runs 3

python3 plot_results.py {timestamp}       # existing plots, now split by on/off
python3 plot_reasoning.py results/{timestamp}/prompts_{timestamp}.csv
python3 analyze_reasoning.py results/{timestamp}/prompts_{timestamp}.csv
```
---

## Parameters

### Benchmarking
```bash
python3 benchmark.py
  --models  gemma3:4b llama3.2:3b   # any number of Ollama tags
  --runs    3                        # repetitions per model
  --idle-secs   20                   # seconds of idle measurement before each model
  --warmup-secs 10                   # extra sleep after model load
  --prompts-file prompts.txt         # custom prompts
  --no-docker                        # Ollama already running (no Docker management)
  --no-gpu                           # CPU-only Docker (no --gpus all)
  --out-dir results    
  --reasoning-modes on off           # which reasoning modes to run for
                                     # reasoning_capable models (default: both)
  --models-config models_config.json # per-model reasoning metadata (see below)
  --gold-answers reasoning_prompts.csv  # optional gold answers for accuracy scoring                            
                                     # output directory
```

### Plots
```bash

# Select a specific benchmark run via timestamp
python3 plot_results.py 20250522_143012

```
The timestamp corresponds to the suffix of the files in `results/` – `prompts_YYYYMMDD_HHMMSS.csv` and `timeseries_YYYYMMDD_HHMMSS.csv` are resolved automatically from it.

---

## Reasoning-Control Extension

Some models (e.g. Qwen3.5, Gemma 4) support an explicit "thinking" / extended
reasoning mode that can be toggled on or off per request. Left uncontrolled,
this turns "model size" and "reasoning mode" into a confounded variable - see
`REASONING_EXTENSION.md` for the full methodology and rationale.

### Setup

1. Copy `models_config.json.example` to `models_config.json` and mark which
   models are `reasoning_capable`, and how their toggle is triggered
   (`"api_param"` for Ollama's native `"think": true/false` field, or
   `"prompt_suffix"` for models/builds that require a prompt suffix like
   `/no_think` instead).
2. **Verify the toggle actually works** for your Ollama version before
   trusting the results - see the "Verifying the reasoning toggle" section in
   `REASONING_EXTENSION.md`. Not all models advertised as "reasoning" models
   support API-level toggling (e.g. `phi4-mini:3.8b` rejects `"think": true`
   outright in our testing - see Limitations).
3. Optionally extend `reasoning_prompts.csv` (format: `prompt,gold_answer,tier`)
   with your own scored prompts, or use `prompts_with_reasoning_tier.txt` as
   your `--prompts-file` to include the bundled Tier-4 reasoning prompts.

### Running

```bash
python3 benchmark.py \
  --models qwen3.5:2b gemma4:e2b phi4-mini:3.8b \
  --prompts-file prompts_with_reasoning_tier.txt \
  --reasoning-modes on off \
  --models-config models_config.json \
  --gold-answers reasoning_prompts.csv \
  --runs 3
```

Models marked `reasoning_capable` in `models_config.json` are run once per
requested mode (so twice, by default); all other models run exactly once,
tagged `reasoning_mode="n/a"` - identical to pre-extension behaviour.

### Analysis

- `python3 plot_results.py {timestamp}` - the existing plots (comparison bar,
  rankings, timeline, temperatures) now automatically split any model that
  has both `on` and `off` data into two paired entries/curves, using the same
  base colour with a lighter tint for "off" and a dashed line for "on".
- `python3 plot_reasoning.py results/{timestamp}/prompts_{timestamp}.csv` -
  reasoning-specific plots: Reasoning Overhead Factor (ROF), thinking- vs.
  answer-token energy split, a fair token-normalised comparison, and (if
  gold answers were scored) an energy-vs-accuracy trade-off plot.
- `python3 analyze_reasoning.py results/{timestamp}/prompts_{timestamp}.csv` -
  the same metrics as text/CSV output, plus an optional mixed-effects model
  (`pip install statsmodels`).

### Merging multiple runs

If you need to re-run individual models (e.g. after fixing a
`models_config.json` entry), merge the resulting `prompts_*.csv` and
`timeseries_*.csv` files before plotting - see `REASONING_EXTENSION.md` for
a ready-to-use merge script. `plot_results.py` and `plot_reasoning.py`
resolve files purely by the `results/{stamp}/` naming convention, so a merged
dataset needs to be placed under a `results/<name>/prompts_<name>.csv` (and
matching `timeseries_<name>.csv`) path to be plottable via `--stamp <name>`
or `python3 plot_results.py <name>`.

### Known limitations

- Ollama reports only a single `eval_count` (total generated tokens) per
  request; the split into `thinking_tokens_est` / `answer_tokens_est` is an
  approximation based on word-count share between the `thinking` and
  `response` fields, not the model's actual tokenizer. Energy is apportioned
  proportionally to this estimate.
- Timeseries rows collected with a `benchmark.py` version predating this
  extension lack the `reasoning_mode` column; `plot_results.py` falls back to
  treating them as untagged (`"n/a"`) rather than guessing, so timeline/
  temperature plots for such runs will not show an on/off split until the
  model is re-run.

## Output

```
results/
  prompts_YYYYMMDD_HHMMSS.csv       ← one entry per prompt execution
  timeseries_YYYYMMDD_HHMMSS.csv    ← raw nvidia-smi samples (500ms interval)
  system_info_YYYYMMDD_HHMMSS.json  ← hardware documentation of the run

plots/
  timeline_gemma3_4b.png          ← averaged power curve: idle vs. load
  timeline_llama3.2_3b.png
  timeline_qwen3.5_2b_on.png      ← reasoning-mode variants get separate timelines
  timeline_qwen3.5_2b_off.png
  comparison_bar.png              ← bar chart model comparison (on/off shown as paired bars)
  temperatures.png                ← CPU and GPU temperature curve (on/off shown as paired curves)
  rankings.png                    ← per-metric model rankings
  summary_table.csv               ← numerical summary

  # via plot_reasoning.py, only for models with both reasoning modes:
  reasoning_overhead.png          ← energy per prompt, on vs. off, with ROF annotated
  reasoning_split.png             ← thinking- vs. answer-token energy share
  reasoning_fair_compare.png      ← J/answer-token (on) vs. J/out-token (off)
  reasoning_accuracy.png          ← accuracy vs. energy-per-correct-answer (if scored)
  reasoning_summary.csv
```

### `prompts_*.csv` columns

| Column | Description |
|---|---|
| `run_id` | repetition number, or `0` for idle |
| `phase` | `idle` or `load` |
| `duration_ms` | response time |
| `tokens_per_sec` | generation speed |
| `gpu_mean_w` | average GPU power during this prompt |
| `gpu_peak_w` | peak GPU power |
| `gpu_energy_j` | energy = mean_w × duration (joules) |

Additional columns added by the reasoning-control extension:

| Column | Description |
|---|---|
| `reasoning_mode` | `"on"`, `"off"`, or `"n/a"` (model never reasoning-capable) |
| `reasoning_toggle_method` | `"api_param"` or `"prompt_suffix"` |
| `thinking_tokens_est` / `answer_tokens_est` | approximate token split (see Limitations) |
| `energy_thinking_j` / `energy_answer_j` | energy apportioned to each split |
| `j_per_answer_token` | energy cost of the final answer only, excluding thinking overhead |
| `correct` | `True`/`False` if scored against `reasoning_prompts.csv`, else empty |

---

## Understanding the Plots

### `timeline_<model>.png`
- **Dashed line**: idle baseline (GPU power without inference)
- **Colored curve**: averaged load power across all runs
- **Shaded area**: ±1σ between runs
- **Δ annotation**: extra consumption compared to idle
- For models with both reasoning modes recorded, this becomes two separate
  files (`timeline_<model>_on.png` / `timeline_<model>_off.png`) rather than
  one averaged curve mixing both modes together.
![timeline chart](example/plots/20260526_120833/timeline_gemma3_4b.png)

### `comparison_bar.png`
- **Top**: idle (light blue) / mean load / peak load per model in watts
- **Bottom**: energy per prompt in joules + tokens/s
- Models with both reasoning modes appear as two adjacent bars, e.g.
  `qwen3.5:2b (off)` / `qwen3.5:2b (on)` - both use the same base colour,
  with the "off" bar in a lighter tint, so paired bars are easy to spot at
  a glance without needing to read the x-axis labels closely.
![comparison chart](example/plots/20260526_120833/comparison_bar.png)

### `temperatures.png`
- **Top**: GPU temperature over the entire run duration per model
- **Bottom**: CPU temperature (if available – depends on lm-sensors or WSL2 thermal zone)
- **Shaded areas**: load phases per model
- CPU temperature may remain empty in WSL2 if neither lm-sensors nor `/sys/class/thermal` is available
- **Dashed curve** = reasoning mode `on`, **solid curve** = `off`, same base
  colour per model. If a model's `timeseries_*.csv` predates the
  reasoning-control extension (no `reasoning_mode` column), that variant is
  skipped with a console note rather than silently merging on/off samples -
  re-run the model to populate it.
![temperatures chart](example/plots/20260526_120833/temperatures.png)


### `reasoning_overhead.png`
Bar pairs (on vs. off) of mean GPU energy per prompt, per model - only for
models with both reasoning modes recorded. Each pair is annotated with the
**Reasoning Overhead Factor (ROF)** = `E[on] / E[off]`, the single most
important number for judging how expensive a model's thinking mode is in
practice.

### `reasoning_split.png`
Stacked bars (reasoning mode `on` only) showing how much of the per-prompt
energy went into thinking tokens vs. the final answer tokens, with the
thinking-token share annotated as a percentage. Apportionment is
approximate - see "Known limitations" above.

### `reasoning_fair_compare.png`
A stricter, token-normalised comparison: J per output-token with reasoning
`off`, vs. J per **answer**-token with reasoning `on` (i.e. excluding the
thinking-token overhead already shown separately in `reasoning_split.png`).
This isolates whether reasoning mode changes the cost of producing the
answer itself, independent of how many extra thinking tokens it generates.

### `reasoning_accuracy.png`
Only produced if `reasoning_prompts.csv` gold answers were supplied to
`benchmark.py`. Plots accuracy on the scored Tier-4 prompts against energy
per correct answer, connecting each model's `on` and `off` point - shows
whether the extra energy of reasoning mode is "paid back" in higher
accuracy, or wasted.

---

## Finding Models

All available tags: https://ollama.com/library

Recommended candidates for comparison:

| Tag | Size | Type |
|---|---|---|
| `gemma3:1b` | ~0.8 GB | Google, very small |
| `gemma3:4b` | ~3 GB | Google, mid-range |
| `llama3.2:1b` | ~1 GB | Meta, very small |
| `llama3.2:3b` | ~2 GB | Meta, mid-range |
| `phi4-mini:3.8b` | ~2.5 GB | Microsoft |
| `mistral:7b` | ~4 GB | Mistral, larger |
| `qwen2.5:3b` | ~2 GB | Alibaba |
