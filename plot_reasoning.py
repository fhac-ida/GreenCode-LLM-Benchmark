#!/usr/bin/env python3
"""
plot_reasoning.py – Visualise the reasoning-control extension results
=======================================================================
Companion script to plot_results.py, kept deliberately SEPARATE so that
plot_results.py (and its output, used for Paper 1) stays untouched and
reproducible.

Reads a prompts_<stamp>.csv produced by benchmark.py's reasoning-control
extension (i.e. containing reasoning_mode/thinking_tokens_est/... columns)
and produces, only for models with both reasoning_mode="on" and "off" rows:

  plots/<stamp>/reasoning_overhead.png   – energy per prompt, on vs off,
                                            with the Reasoning Overhead
                                            Factor (ROF) annotated
  plots/<stamp>/reasoning_split.png      – thinking- vs answer-token energy
                                            share per model (mode="on")
  plots/<stamp>/reasoning_fair_compare.png – J/answer-token (on) vs
                                            J/out-token (off): the fair,
                                            token-normalised comparison
  plots/<stamp>/reasoning_accuracy.png   – accuracy vs energy-per-correct
                                            answer, on vs off (only if
                                            --gold-answers scoring was used)
  plots/<stamp>/reasoning_summary.csv    – numeric summary table

Usage
-----
    python3 plot_reasoning.py results/<stamp>/prompts_<stamp>.csv
    python3 plot_reasoning.py results/merged_prompts_reasoning_study.csv --stamp merged
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Style (matches plot_results.py so figures look consistent) ──────────────
MODEL_COLORS = [
    "#1565C0", "#AD1457", "#2E7D32", "#F57F17",
    "#4527A0", "#00695C", "#BF360C", "#37474F",
    "#C2185B", "#E65100", "#6A1B9A"
]
ON_COLOR  = "#AD1457"   # reasoning on
OFF_COLOR = "#1565C0"   # reasoning off
GRID_ALPHA = 0.35

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


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

NUMERIC_COLS = [
    "gpu_energy_j", "gen_tokens", "thinking_tokens_est", "answer_tokens_est",
    "energy_thinking_j", "energy_answer_j", "j_per_answer_token",
    "j_per_out_token", "duration_ms",
]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["phase"] == "load"].copy()
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def reasoning_capable_models(df: pd.DataFrame) -> list[str]:
    """Models that have BOTH reasoning_mode='on' and 'off' rows present -
    i.e. models the reasoning-control extension actually toggled, as
    opposed to models that failed to toggle (see phi4-mini case) or
    models that were never reasoning_capable (mode='n/a')."""
    modes_per_model = df.groupby("model")["reasoning_mode"].apply(set)
    return sorted(m for m, modes in modes_per_model.items() if {"on", "off"} <= modes)


# ═══════════════════════════════════════════════════════════════════════════
# Plot 1 – Reasoning Overhead: Energy per prompt, on vs off
# ═══════════════════════════════════════════════════════════════════════════

def plot_reasoning_overhead(df: pd.DataFrame, models: list[str], out_dir: Path) -> pd.DataFrame:
    rows = []
    for m in models:
        sub = df[df["model"] == m]
        e_on = sub[sub["reasoning_mode"] == "on"]["gpu_energy_j"].mean()
        e_off = sub[sub["reasoning_mode"] == "off"]["gpu_energy_j"].mean()
        rof = e_on / e_off if e_off > 0 else np.nan
        rows.append({"model": m, "energy_on_j": e_on, "energy_off_j": e_off, "rof": rof})
    summary = pd.DataFrame(rows).sort_values("rof", ascending=False)

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(models) + 2), 4.5))
    x = np.arange(len(summary))
    width = 0.35

    bars_off = ax.bar(x - width / 2, summary["energy_off_j"], width,
                       label="Reasoning off", color=OFF_COLOR, edgecolor="white")
    bars_on = ax.bar(x + width / 2, summary["energy_on_j"], width,
                      label="Reasoning on", color=ON_COLOR, edgecolor="white")

    for xi, row in zip(x, summary.itertuples()):
        top = max(row.energy_on_j, row.energy_off_j)
        ax.text(xi, top * 1.03, f"ROF ×{row.rof:.1f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(":", "\n") for m in summary["model"]], fontsize=9)
    ax.set_ylabel("Mean GPU Energy per Prompt (J)")
    ax.set_title("Reasoning Overhead: Energy per Prompt, On vs. Off\n"
                 "ROF = E[reasoning on] / E[reasoning off]")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    out_png = out_dir / "reasoning_overhead.png"
    out_pdf = out_dir / "reasoning_overhead.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_png} and .pdf")
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Plot 2 – Thinking vs. answer token/energy share (mode="on")
# ═══════════════════════════════════════════════════════════════════════════

def plot_reasoning_split(df: pd.DataFrame, models: list[str], out_dir: Path) -> pd.DataFrame:
    rows = []
    for m in models:
        sub = df[(df["model"] == m) & (df["reasoning_mode"] == "on")]
        e_think = sub["energy_thinking_j"].mean()
        e_ans = sub["energy_answer_j"].mean()
        t_think = sub["thinking_tokens_est"].mean()
        t_ans = sub["answer_tokens_est"].mean()
        rows.append({
            "model": m, "energy_thinking_j": e_think, "energy_answer_j": e_ans,
            "thinking_tokens": t_think, "answer_tokens": t_ans,
            "thinking_share_pct": 100 * t_think / (t_think + t_ans) if (t_think + t_ans) > 0 else 0,
        })
    summary = pd.DataFrame(rows).sort_values("thinking_share_pct", ascending=False)

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(models) + 2), 4.5))
    x = np.arange(len(summary))

    bars_think = ax.bar(x, summary["energy_thinking_j"], 0.5,
                         label="Thinking-token energy", color="#6A1B9A", edgecolor="white")
    bars_ans = ax.bar(x, summary["energy_answer_j"], 0.5,
                       bottom=summary["energy_thinking_j"],
                       label="Answer-token energy", color="#00695C", edgecolor="white")

    for xi, row in zip(x, summary.itertuples()):
        total = row.energy_thinking_j + row.energy_answer_j
        ax.text(xi, total * 1.02, f"{row.thinking_share_pct:.0f}% thinking",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(":", "\n") for m in summary["model"]], fontsize=9)
    ax.set_ylabel("Mean GPU Energy per Prompt (J), apportioned")
    ax.set_title("Thinking- vs. Answer-Token Energy Share (reasoning_mode=on)\n"
                 "Apportionment is proportional to approx. token count - see limitations")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    out_png = out_dir / "reasoning_split.png"
    out_pdf = out_dir / "reasoning_split.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_png} and .pdf")
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Plot 3 – Fair, token-normalised comparison: J/answer-token (on) vs J/out-token (off)
# ═══════════════════════════════════════════════════════════════════════════

def plot_fair_comparison(df: pd.DataFrame, models: list[str], out_dir: Path) -> pd.DataFrame:
    """Compares energy cost of producing the FINAL ANSWER only (mode=on,
    j_per_answer_token) against the plain per-token cost with reasoning off
    (j_per_out_token). This isolates whether reasoning changes the cost of
    generating the answer itself, separate from the thinking overhead
    already shown in Plot 1/2."""
    rows = []
    for m in models:
        sub = df[df["model"] == m]
        j_on_answer = sub[sub["reasoning_mode"] == "on"]["j_per_answer_token"].mean()
        j_off = sub[sub["reasoning_mode"] == "off"]["j_per_out_token"].mean()
        rows.append({"model": m, "j_per_answer_token_on": j_on_answer, "j_per_out_token_off": j_off})
    summary = pd.DataFrame(rows).sort_values("j_per_out_token_off")

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(models) + 2), 4.5))
    x = np.arange(len(summary))
    width = 0.35

    ax.bar(x - width / 2, summary["j_per_out_token_off"], width,
           label="Reasoning off: J / output-token", color=OFF_COLOR, edgecolor="white")
    ax.bar(x + width / 2, summary["j_per_answer_token_on"], width,
           label="Reasoning on: J / answer-token only\n(thinking tokens excluded)",
           color=ON_COLOR, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(":", "\n") for m in summary["model"]], fontsize=9)
    ax.set_ylabel("J / token")
    ax.set_title("Fair Comparison: Cost of the Final Answer Itself\n"
                 "(excludes thinking-token overhead from the 'on' bars)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    out_png = out_dir / "reasoning_fair_compare.png"
    out_pdf = out_dir / "reasoning_fair_compare.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_png} and .pdf")
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Plot 4 – Accuracy vs. energy-per-correct-answer (only if scored)
# ═══════════════════════════════════════════════════════════════════════════

def plot_accuracy_energy(df: pd.DataFrame, models: list[str], out_dir: Path) -> pd.DataFrame | None:
    if "correct" not in df.columns:
        return None
    scored = df[df["correct"].isin([True, False, "True", "False"])].copy()
    if scored.empty:
        print("  No scored (Tier-4) rows found - skipping accuracy/energy plot. "
              "Requires --gold-answers reasoning_prompts.csv to have been used.")
        return None
    scored["correct"] = scored["correct"].astype(str) == "True"
    scored = scored[scored["model"].isin(models)]
    if scored.empty:
        print("  No scored rows for reasoning-toggled models - skipping accuracy plot.")
        return None

    rows = []
    for (m, mode), g in scored.groupby(["model", "reasoning_mode"]):
        n_correct = g["correct"].sum()
        acc = g["correct"].mean()
        energy_per_correct = g["gpu_energy_j"].sum() / n_correct if n_correct > 0 else np.nan
        rows.append({"model": m, "reasoning_mode": mode, "accuracy": acc,
                     "energy_per_correct_j": energy_per_correct, "n_correct": n_correct, "n_total": len(g)})
    summary = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for m in models:
        sub = summary[summary["model"] == m]
        if len(sub) < 2:
            continue
        off_row = sub[sub["reasoning_mode"] == "off"].iloc[0]
        on_row = sub[sub["reasoning_mode"] == "on"].iloc[0]
        ax.plot([off_row["accuracy"], on_row["accuracy"]],
                [off_row["energy_per_correct_j"], on_row["energy_per_correct_j"]],
                marker="o", label=m, linewidth=1.5)
        ax.annotate("off", (off_row["accuracy"], off_row["energy_per_correct_j"]),
                    textcoords="offset points", xytext=(6, -4), fontsize=7)
        ax.annotate("on", (on_row["accuracy"], on_row["energy_per_correct_j"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)

    ax.set_xlabel("Accuracy on Tier-4 (scored) prompts")
    ax.set_ylabel("Energy per Correct Answer (J)")
    ax.set_title("Energy-Quality Trade-off: Reasoning On vs. Off")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()

    out_png = out_dir / "reasoning_accuracy.png"
    out_pdf = out_dir / "reasoning_accuracy.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_png} and .pdf")
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Visualise the reasoning-control extension results")
    parser.add_argument("prompts_csv", help="Path to a prompts_<stamp>.csv (or merged CSV) "
                                             "containing reasoning_mode columns")
    parser.add_argument("--stamp", default=None,
                         help="Output subfolder name under plots/. Defaults to the CSV's stem.")
    args = parser.parse_args()

    path = Path(args.prompts_csv)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    stamp = args.stamp or path.stem
    out_dir = Path("plots") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(path)
    models = reasoning_capable_models(df)

    if not models:
        print("No models with both reasoning_mode='on' and 'off' rows found in this CSV.")
        print("(Nothing to plot - check that benchmark.py was run with --reasoning-modes on off "
              "against reasoning_capable models, and that the toggle actually worked - see "
              "REASONING_EXTENSION.md.)")
        sys.exit(0)

    print(f"Reasoning-toggled models found: {models}\n")

    print("── Plot 1: Reasoning overhead (energy per prompt)")
    overhead_summary = plot_reasoning_overhead(df, models, out_dir)

    print("\n── Plot 2: Thinking vs. answer energy split")
    split_summary = plot_reasoning_split(df, models, out_dir)

    print("\n── Plot 3: Fair token-normalised comparison")
    fair_summary = plot_fair_comparison(df, models, out_dir)

    print("\n── Plot 4: Accuracy vs. energy-per-correct-answer")
    acc_summary = plot_accuracy_energy(df, models, out_dir)

    # ── Combined summary CSV ─────────────────────────────────────────────
    merged = overhead_summary.merge(split_summary, on="model").merge(fair_summary, on="model")
    out_csv = out_dir / "reasoning_summary.csv"
    merged.to_csv(out_csv, index=False)
    print(f"\n  Saved {out_csv}")
    print()
    print(merged.round(3).to_string(index=False))

    print(f"\nAll reasoning plots in: {out_dir}/")


if __name__ == "__main__":
    main()
