#!/usr/bin/env python3
"""
plot_results.py – Visualise LLM energy benchmark results (WSL2/NVIDIA edition)
===============================================================================
Reads the two CSV files written by benchmark.py and produces:

  plots/timeline_<model>.png    – averaged GPU power curve (idle vs load)
  plots/comparison_bar.png      – per-model bar chart (power + energy)
  plots/summary_table.csv       – numeric summary

Usage:
    python3 plot_results.py results/prompts_*.csv results/timeseries_*.csv
    python3 plot_results.py                        # auto-finds latest files
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
import json

# ── Style ─────────────────────────────────────────────────────────────────────
MODEL_COLORS = [
    "#1565C0", "#AD1457", "#2E7D32", "#F57F17",
    "#4527A0", "#00695C", "#BF360C", "#37474F",
    "#C2185B", "#E65100", "#6A1B9A"
]
IDLE_COLOR  = "#90CAF9"
GRID_ALPHA  = 0.35

def lighten(color: str, amount: float = 0.45):
    """Blends a color towards white. Used so that the 'off' bar of a
    reasoning on/off pair uses a lighter tint of the same base color as
    the 'on' bar - visually pairing them while keeping them distinct."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": GRID_ALPHA,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
})

PLOTS_DIR       = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

SMOOTH_WINDOW = 50


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_prompts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["duration_ms", "tokens_per_sec", "prompt_tokens", "gen_tokens",
                "gpu_mean_w", "gpu_peak_w", "gpu_energy_j"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    # reasoning-control extension: older CSVs (Paper 1) won't have this column -
    # default to "n/a" so all downstream logic treats them exactly as before.
    if "reasoning_mode" not in df.columns:
        df["reasoning_mode"] = "n/a"
    df["reasoning_mode"] = df["reasoning_mode"].fillna("n/a")
    return df


def load_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["gpu_w"] = pd.to_numeric(df["gpu_w"], errors="coerce").fillna(0.0)
    df["t_mono"] = pd.to_numeric(df["t_mono"], errors="coerce")
    df["gpu_temp_c"] = pd.to_numeric(df.get("gpu_temp_c", 0), errors="coerce")
    df["cpu_temp_c"] = pd.to_numeric(df.get("cpu_temp_c", 0), errors="coerce")
    # reasoning-control extension: same backward-compat fallback as load_prompts()
    if "reasoning_mode" not in df.columns:
        df["reasoning_mode"] = "n/a"
    df["reasoning_mode"] = df["reasoning_mode"].fillna("n/a")
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# Reasoning-mode variant expansion
# ═══════════════════════════════════════════════════════════════════════════════

def build_model_variants(df_p: pd.DataFrame) -> list[dict]:
    """Expands each model into one or two plottable 'variants'.

    - Model has both reasoning_mode='on' and 'off' rows -> two variants,
      labelled '<model> (on)' / '<model> (off)', so every existing plot
      (comparison bar, rankings, summary table, timeline) shows the
      reasoning-overhead effect directly alongside all other models.
    - Otherwise (mode is always 'n/a', or the toggle never produced an
      'on' row, e.g. a model that doesn't support thinking) -> exactly
      one variant, labelled with the bare model name, filtered to
      whatever rows exist. This reproduces the pre-extension behaviour
      1:1 for Paper-1-style CSVs and for non-reasoning-capable models.

    Returns a list of dicts: {label, model, mode} where mode is None
    (no filtering beyond model name) or "on"/"off".
    """
    variants = []
    for model, modes in df_p.groupby("model")["reasoning_mode"].apply(set).items():
        if {"on", "off"} <= modes:
            variants.append({"label": f"{model} (off)", "model": model, "mode": "off"})
            variants.append({"label": f"{model} (on)",  "model": model, "mode": "on"})
        else:
            variants.append({"label": model, "model": model, "mode": None})
    return variants


# ═══════════════════════════════════════════════════════════════════════════════
# Per-model stats
# ═══════════════════════════════════════════════════════════════════════════════

def model_stats(df_p: pd.DataFrame, model: str, mode: str | None = None) -> dict:
    m = df_p[df_p["model"] == model]
    if mode is not None:
        m = m[m["reasoning_mode"] == mode]
    idle = m[m["phase"] == "idle"]
    load = m[m["phase"] == "load"]

    idle_w  = idle["gpu_mean_w"].mean() if not idle.empty else 0.0
    load_w  = load["gpu_mean_w"].mean() if not load.empty else 0.0
    peak_w  = load["gpu_peak_w"].max()  if not load.empty else 0.0
    total_j = load["gpu_energy_j"].sum()
    n       = len(load)
    mean_dur = load["duration_ms"].mean() if n else 0.0
    mean_tps = load["tokens_per_sec"].mean() if n else 0.0
    total_tokens = (load["prompt_tokens"] + load["gen_tokens"]).sum()
    total_out_tok = load["gen_tokens"].sum()

    return dict(
        idle_w=idle_w, load_w=load_w, peak_w=peak_w,
        total_j=total_j,
        energy_per_prompt_j=total_j / n if n else 0.0,
        n_prompts=n,
        mean_dur_ms=mean_dur,
        mean_tps=mean_tps,
        j_per_token=total_j / total_tokens if total_tokens > 0 else 0.0,
        j_per_out_token=total_j / total_out_tok if total_out_tok > 0 else 0.0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1 – Timeline per model
# ═══════════════════════════════════════════════════════════════════════════════

def plot_timeline(df_ts: pd.DataFrame, df_p: pd.DataFrame,
                  model: str, color: str, stamp: str,
                  mode: str | None = None, label: str | None = None) -> None:
    """
    Averaged power-over-time curve for one model (optionally one
    reasoning_mode variant of that model).
    Strategy: re-index all runs onto a common relative time axis, then average.
    """
    label = label or model
    m_ts = df_ts[df_ts["model"] == model].copy()
    if mode is not None:
        m_ts = m_ts[m_ts["reasoning_mode"] == mode]
    m_p = df_p[df_p["model"] == model].copy()

    idle_ts = m_ts[m_ts["phase"] == "idle"]
    load_ts = m_ts[~m_ts["phase"].isin(["idle"])]

    if m_ts.empty:
        print(f"  No timeseries data for {model}, skipping timeline.")
        return

    # ── Build averaged load trace ──────────────────────────────────────────
    # Group by run, normalise t_mono to start at 0, interpolate onto common grid
    run_ids = sorted(set(
        ph.split("_")[0] for ph in load_ts["phase"].unique() if ph.startswith("run")
    ))

    all_interp: list[np.ndarray] = []
    max_dur = 0.0

    for rid in run_ids:
        run_mask = load_ts["phase"].str.startswith(rid)
        seg = load_ts[run_mask].sort_values("t_mono")
        if seg.empty:
            continue
        t_rel = seg["t_mono"].values - seg["t_mono"].iloc[0]
        w = seg["gpu_w"].values
        max_dur = max(max_dur, t_rel[-1])
        all_interp.append((t_rel, w))

    if not all_interp or max_dur == 0:
        print(f"  No load timeseries for {model}")
        return

    # Common time grid
    POLL_INTERVAL = 0.5  # matches benchmark.py
    grid = np.linspace(0, max_dur, num=int(max_dur / POLL_INTERVAL + 1))

    stacked = []
    for t_rel, w in all_interp:
        interp_w = np.interp(grid, t_rel, w)
        stacked.append(interp_w)

    mean_trace = np.mean(stacked, axis=0)
    std_trace = np.std(stacked, axis=0) if len(stacked) > 1 else np.zeros_like(mean_trace)

    # Idle stats
    idle_mean = idle_ts["gpu_w"].mean() if not idle_ts.empty else 0.0

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))

    # Idle baseline band
    ax.axhspan(0, idle_mean, alpha=0.12, color=IDLE_COLOR, label=None)
    ax.axhline(idle_mean, color=IDLE_COLOR, linewidth=2, linestyle="--",
               label=f"Idle baseline  {idle_mean:.1f} W")

    # Shaded std band
    if len(stacked) > 1:
        ax.fill_between(grid, mean_trace - std_trace, mean_trace + std_trace,
                        color=color, alpha=0.18, label="±1 σ across runs")

    # Mean trace
    load_mean_w = float(np.mean(mean_trace))
    ax.plot(grid, mean_trace, color=color, linewidth=2,
            label=f"Load (avg {load_mean_w:.1f} W)")

    # Annotate delta
    delta = load_mean_w - idle_mean
    ax.annotate(
        f"Δ = {delta:+.1f} W vs idle",
        xy=(grid[len(grid) // 2], load_mean_w),
        xytext=(grid[len(grid) // 2], load_mean_w + max(2, delta * 0.4)),
        fontsize=9, color=color,
        arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        ha="center",
    )

    ax.set_xlabel("Time (seconds, relative to first prompt)")
    ax.set_ylabel("GPU Power Draw (W)")
    ax.set_ylim(bottom=0)
    label_safe = label.replace(":", "_").replace(" ", "_").replace("(", "").replace(")", "")
    n_runs = len(stacked)
    ax.set_title(
        f"GPU Power Timeline — {label}  ({n_runs} run{'s' if n_runs != 1 else ''} averaged)\n"
        f"Idle: {idle_mean:.1f} W  │  Load avg: {load_mean_w:.1f} W  │  Δ = {delta:+.1f} W"
    )
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    # Save PNG and PDF
    out_png = PLOTS_DIR / f"{stamp}/timeline_{label_safe}.png"
    out_pdf = PLOTS_DIR / f"{stamp}/timeline_{label_safe}.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")  # PDF speichern
    plt.close(fig)
    print(f"  Saved {out_png} and .pdf")


def plot_temperatures(df_ts: pd.DataFrame, variants: list[str], colors: list[str], stamp: str) -> None:
    """
    Dual-panel temperature plot for all models combined.
    Top: GPU temperature, Bottom: CPU temperature.
    """
    import math

    has_gpu_temp = df_ts["gpu_temp_c"].notna().any() and (df_ts["gpu_temp_c"] != 0).any()
    has_cpu_temp = df_ts["cpu_temp_c"].notna().any() and (df_ts["cpu_temp_c"] != 0).any()

    if not has_gpu_temp and not has_cpu_temp:
        print("  No temperature data available, skipping temperature plot.")
        return

    n_panels = sum([has_gpu_temp, has_cpu_temp])
    fig, axes = plt.subplots(n_panels, 1, figsize=(13, 4 * n_panels), squeeze=False)
    panel = 0

    for temp_col, title, unit in [
        ("gpu_temp_c", "GPU Temperature", "°C"),
        ("cpu_temp_c", "CPU Temperature", "°C"),
    ]:
        if temp_col == "gpu_temp_c" and not has_gpu_temp:
            continue
        if temp_col == "cpu_temp_c" and not has_cpu_temp:
            continue

        ax = axes[panel][0]

        for i, v in enumerate(variants):
            m_ts = df_ts[df_ts["model"] == v["model"]].copy()
            if v["mode"] is not None:
                m_ts = m_ts[m_ts["reasoning_mode"] == v["mode"]]
            if m_ts.empty:
                print(f"    (no timeseries data tagged reasoning_mode={v['mode']!r} "
                      f"for {v['model']} - skipping {v['label']} in temperature plot; "
                      f"re-run this model with the current benchmark.py to fix)")
                continue

            # Normalize time to 0
            t0 = m_ts["t_mono"].min()
            m_ts["t_rel"] = m_ts["t_mono"] - t0

            temps = pd.to_numeric(m_ts[temp_col], errors="coerce")
            valid = temps.notna() & temps.gt(0)
            if not valid.any():
                continue

            temps_smoothed = (
                temps[valid]
                .reset_index(drop=True)
                .rolling(window=SMOOTH_WINDOW, center=True, min_periods=1)
                .mean()
            )

            color = colors[i % len(colors)]
            linestyle = "--" if v["mode"] == "on" else "-"

            ax.plot(
                m_ts.loc[valid, "t_rel"],
                temps_smoothed,
                color=color,
                linewidth=1.4,
                alpha=0.85,
                linestyle=linestyle,
                label=v["label"],
            )

            ax.plot(m_ts.loc[valid, "t_rel"], temps[valid],
                    color=color, alpha=0.12, linewidth=0.7, linestyle=linestyle)

            # Shade load phases
            load_mask = ~m_ts["phase"].isin(["idle"])
            if load_mask.any():
                load_times = m_ts.loc[load_mask, "t_rel"]
                if not load_times.empty:
                    ax.axvspan(
                        load_times.min(), load_times.max(),
                        alpha=0.07, color=color,
                    )

        ax.set_xlabel("Time (seconds, relative)")
        ax.set_ylabel(f"Temperature ({unit})")
        ax.set_title(title)
        ax.legend(fontsize=9)
        panel += 1

    plt.suptitle("CPU / GPU Temperature over Benchmark Run",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    # Save PNG and PDF
    out_png = PLOTS_DIR / f"{stamp}/temperatures.png"
    out_pdf = PLOTS_DIR / f"{stamp}/temperatures.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_png} and .pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2 – Model comparison bar chart
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comparison(models: list[str], stats: dict[str, dict], colors: list[str], stamp: str) -> None:
    n = len(models)
    x = np.arange(n)
    labels = [m.replace(":", "\n") for m in models]

    fig = plt.figure(figsize=(max(9, n * 2.6), 10))
    gs = GridSpec(3, 1, figure=fig, hspace=0.5)

    # ── Top: Power comparison (idle vs load vs peak) ───────────────────────
    ax1 = fig.add_subplot(gs[0])
    w = 0.25

    idle_vals = [stats[m]["idle_w"] for m in models]
    load_vals = [stats[m]["load_w"] for m in models]
    peak_vals = [stats[m]["peak_w"] for m in models]

    b_idle = ax1.bar(x - w, idle_vals, w, label="Idle mean (W)",
                     color=IDLE_COLOR, edgecolor="white", linewidth=0.8)
    b_load = ax1.bar(x, load_vals, w, label="Load mean (W)",
                     color=colors[:n], edgecolor="white", linewidth=0.8)
    b_peak = ax1.bar(x + w, peak_vals, w, label="Load peak (W)",
                     color=colors[:n], alpha=0.45, edgecolor=colors[:n],
                     linewidth=1.2, linestyle="--")

    # Value labels
    for bars, vals in [(b_idle, idle_vals), (b_load, load_vals), (b_peak, peak_vals)]:
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.4,
                     f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("GPU Power (W)")
    ax1.set_title("Idle / Load / Peak GPU Power per Model")
    ax1.legend(fontsize=9)

    # ── Second: Energy per prompt ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])

    epj = [stats[m]["energy_per_prompt_j"] for m in models]
    bars = ax2.bar(x, epj, width=0.5, color=colors[:n], edgecolor="white", linewidth=0.8)

    for bar, v, m in zip(bars, epj, models):
        tps = stats[m]["mean_tps"]
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.05,
                 f"{v:.1f} J\n{tps:.0f} tok/s",
                 ha="center", va="bottom", fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Energy per Prompt (J)")
    ax2.set_title("Average GPU Energy per Prompt")

    # ── Third: Energy per token ──────────────────────────────────────────

    ax3 = fig.add_subplot(gs[2])
    jpot = [stats[m]["j_per_out_token"] for m in models]
    bars = ax3.bar(x, jpot, width=0.5, color=colors[:n], edgecolor="white", linewidth=0.8)
    for bar, v, m in zip(bars, epj, models):
        joul = stats[m]["j_per_out_token"]
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.05,
                 f"{joul:.2f} J/tok",
                 ha="center", va="bottom", fontsize=8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=9)
    ax3.set_ylabel("J / Output-Token")
    ax3.set_title("GPU Energy per Output Token")

    # ──────────────────────────────────────────

    plt.suptitle("LLM Energy Benchmark — Model Comparison",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    # Save PNG and PDF
    out_png = PLOTS_DIR / f"{stamp}/comparison_bar.png"
    out_pdf = PLOTS_DIR / f"{stamp}/comparison_bar.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_png} and .pdf")


RANKING_METRICS = [
    ("load_w", "Load avg (W)", "lower_is_better"),
    ("peak_w", "Peak (W)", "lower_is_better"),
    ("j_per_out_token", "J / Output-Token", "lower_is_better"),
    ("energy_per_prompt_j", "J / Prompt", "lower_is_better"),
    ("mean_tps", "Throughput (tok/s)", "higher_is_better"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3 – Rankings
# ═══════════════════════════════════════════════════════════════════════════════

def plot_rankings(models, stats, colors, stamp):
    n_metrics = len(RANKING_METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, max(3, len(models) * 0.8 + 1.5)))

    for ax, (key, label, direction) in zip(axes, RANKING_METRICS):
        vals = [(m, stats[m][key]) for m in models]
        sorted_vals = sorted(vals, key=lambda x: x[1],
                             reverse=(direction == "higher_is_better"))

        mdl_labels = [v[0].replace(":", "\n") for v in sorted_vals]
        bar_vals = [v[1] for v in sorted_vals]
        bar_colors = [colors[models.index(v[0]) % len(colors)] for v in sorted_vals]

        bars = ax.barh(mdl_labels, bar_vals, color=bar_colors, edgecolor="white")
        ax.invert_yaxis()  # Bester oben

        # Wert-Labels
        for bar, v in zip(bars, bar_vals):
            ax.text(bar.get_width() * 1.02, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}", va="center", fontsize=8)

        # Gewinner/Verlierer markieren
        ax.get_yticklabels()[0].set_color("green")
        ax.get_yticklabels()[-1].set_color("red")

        best_label = "↓ best" if direction == "lower_is_better" else "↑ best"
        ax.set_title(f"{label}\n{best_label}", fontsize=9)
        ax.set_xlabel(label, fontsize=8)

    plt.suptitle("Model Rankings", fontsize=13, fontweight="bold")
    plt.tight_layout()

    # Save PNG and PDF
    out_png = PLOTS_DIR / f"{stamp}/rankings.png"
    out_pdf = PLOTS_DIR / f"{stamp}/rankings.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_png} and .pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary CSV
# ═══════════════════════════════════════════════════════════════════════════════

def save_summary(models: list[str], stats: dict[str, dict], stamp: str) -> None:
    import csv as csv_mod
    rows = []
    for m in models:
        s = stats[m]
        rows.append({
            "model":                m,
            "idle_gpu_w":           round(s["idle_w"],              2),
            "load_gpu_mean_w":      round(s["load_w"],              2),
            "load_gpu_peak_w":      round(s["peak_w"],              2),
            "delta_w":              round(s["load_w"] - s["idle_w"], 2),
            "total_gpu_energy_j":   round(s["total_j"],             2),
            "energy_per_prompt_j":  round(s["energy_per_prompt_j"], 2),
            "mean_duration_ms":     round(s["mean_dur_ms"],          0),
            "mean_tokens_per_sec":  round(s["mean_tps"],             1),
            "n_prompts_measured":   s["n_prompts"],
            "j_per_token": round(s["j_per_token"], 4),
            "j_per_out_token": round(s["j_per_out_token"], 4),
        })

    out = PLOTS_DIR / f"{stamp}/summary_table.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv_mod.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved {out}")
    print()
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_files(stamp: str) -> tuple[Path, Path]:
    """Baut aus einem Zeitstempel (z.B. 20250522_143012) die beiden Pfade."""
    prompt_path = Path(f"results/{stamp}") / f"prompts_{stamp}.csv"
    ts_path     = Path(f"results/{stamp}") / f"timeseries_{stamp}.csv"
    for p in (prompt_path, ts_path):
        if not p.exists():
            print(f"File not found: {p}")
            sys.exit(1)
    return prompt_path, ts_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stamp", nargs="?", default=None,
        help="Zeitstempel-Suffix, z.B. 20250522_143012 – "
             "lässt du ihn weg, wird automatisch die neueste Datei gewählt"
    )
    args = parser.parse_args()

    if not args.stamp:
        print(f"Stamp not found")
        sys.exit(1)

    prompt_path, ts_path = resolve_files(args.stamp)

    print(f"Prompt CSV   : {prompt_path}")
    print(f"Timeseries   : {ts_path}")

    df_p  = load_prompts(prompt_path)
    df_ts = load_timeseries(ts_path)

    dir = Path(f"plots/{args.stamp}")
    dir.mkdir(exist_ok=True)

    # Show system info if available
    sys_info_path = prompt_path.parent / f"system_info_{prompt_path.stem.replace('prompts_', '')}.json"
    if sys_info_path.exists():
        with open(sys_info_path, encoding="utf-8") as f:
            si = json.load(f)
        gpu_names = ", ".join(g["name"] for g in si.get("gpu", [])) or "n/a"
        print(f"System : {si['os']['node']}  |  {si['cpu']['brand']}")
        print(f"GPU    : {gpu_names}")
        print(f"RAM    : {si.get('ram_total_gb', 'n/a')} GB\n")

    real_models = [m for m in df_p["model"].unique() if pd.notna(m)]
    if not real_models:
        print("No model data found.")
        sys.exit(1)

    # ── Reasoning-control extension: expand into on/off variants where
    # applicable. For CSVs without reasoning-mode data (Paper 1 runs, or
    # models that were never reasoning_capable) this is a 1:1 passthrough
    # of the previous behaviour - one label per model, no filtering. ──────
    variants = build_model_variants(df_p)
    labels = [v["label"] for v in variants]
    print(f"Models: {real_models}")
    n_split = sum(1 for v in variants if v["mode"] is not None)
    if n_split:
        print(f"  -> {n_split} of these are shown as on/off reasoning-mode pairs\n")
    else:
        print()

    # Assign each REAL model a base colour; "off" variants get a lightened
    # tint of the same colour so paired bars are visually grouped.
    base_color_map: dict[str, str] = {}
    for i, m in enumerate(real_models):
        base_color_map[m] = MODEL_COLORS[i % len(MODEL_COLORS)]

    colors = []
    for v in variants:
        base = base_color_map[v["model"]]
        colors.append(lighten(base) if v["mode"] == "off" else base)

    stats_map: dict[str, dict] = {}
    for v, color in zip(variants, colors):
        stats_map[v["label"]] = model_stats(df_p, v["model"], v["mode"])
        print(f"── Timeline: {v['label']}")
        plot_timeline(df_ts, df_p, v["model"], color, args.stamp,
                      mode=v["mode"], label=v["label"])

    print("\n── Comparison chart")
    plot_comparison(labels, stats_map, colors, args.stamp)

    print("\n── Temperature plot")
    plot_temperatures(df_ts, variants, colors, args.stamp)
    
    print("\n── Summary")
    save_summary(labels, stats_map, args.stamp)

    print("\n── Rankings")
    plot_rankings(labels, stats_map, colors, args.stamp)

    print(f"\nAll plots in: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()