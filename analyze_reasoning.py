#!/usr/bin/env python3
"""
analyze_reasoning.py - Reasoning-mode analysis for the GreenCode LLM Benchmark
================================================================================
Reads a prompts_<stamp>.csv produced by the reasoning-control extension of
benchmark.py (i.e. run with --reasoning-modes on off against
reasoning_capable models) and reports:

  1. Reasoning Overhead Factor (ROF) per model:
         ROF = mean(energy_j | reasoning_mode=on) / mean(energy_j | reasoning_mode=off)
  2. Thinking vs. answer token/energy share per model, reasoning_mode=on
  3. Energy-per-correct-answer on the scored Tier-4 subset (requires
     reasoning_prompts.csv gold answers to have been supplied to benchmark.py)
  4. A simple mixed-model-style ANOVA-lite via statsmodels, if installed
     (optional - falls back to descriptive stats only)

Usage
-----
    python3 analyze_reasoning.py results/prompts_<stamp>.csv
"""

import sys
from pathlib import Path

import pandas as pd


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["phase"] == "load"].copy()
    numeric_cols = [
        "gpu_energy_j", "gen_tokens", "thinking_tokens_est", "answer_tokens_est",
        "energy_thinking_j", "energy_answer_j", "j_per_answer_token",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def reasoning_overhead_factor(df: pd.DataFrame) -> pd.DataFrame:
    capable = df[df["reasoning_mode"].isin(["on", "off"])]
    if capable.empty:
        print("No reasoning_mode=on/off rows found - nothing to compute. "
              "Did you run benchmark.py with --reasoning-modes on off "
              "against reasoning_capable models?")
        return pd.DataFrame()

    grouped = capable.groupby(["model", "reasoning_mode"])["gpu_energy_j"].mean().unstack()
    grouped["ROF"] = grouped.get("on") / grouped.get("off")
    return grouped.sort_values("ROF", ascending=False)


def thinking_answer_split(df: pd.DataFrame) -> pd.DataFrame:
    on_rows = df[df["reasoning_mode"] == "on"]
    if on_rows.empty:
        return pd.DataFrame()
    agg = on_rows.groupby("model").agg(
        mean_thinking_tokens=("thinking_tokens_est", "mean"),
        mean_answer_tokens=("answer_tokens_est", "mean"),
        mean_energy_thinking_j=("energy_thinking_j", "mean"),
        mean_energy_answer_j=("energy_answer_j", "mean"),
        mean_j_per_answer_token=("j_per_answer_token", "mean"),
    )
    agg["thinking_token_share_pct"] = (
        100 * agg["mean_thinking_tokens"]
        / (agg["mean_thinking_tokens"] + agg["mean_answer_tokens"])
    )
    return agg.sort_values("thinking_token_share_pct", ascending=False)


def energy_per_correct(df: pd.DataFrame) -> pd.DataFrame:
    scored = df[df["correct"].isin([True, False, "True", "False"])].copy()
    if scored.empty:
        print("No scored rows found (correct column empty for all rows). "
              "Pass --gold-answers reasoning_prompts.csv to benchmark.py "
              "and include the Tier-4 prompts in your prompt set to enable this.")
        return pd.DataFrame()
    scored["correct"] = scored["correct"].astype(str) == "True"

    rows = []
    for (model, mode), g in scored.groupby(["model", "reasoning_mode"]):
        n_correct = g["correct"].sum()
        total_energy = g["gpu_energy_j"].sum()
        accuracy = g["correct"].mean()
        energy_per_correct = total_energy / n_correct if n_correct > 0 else float("nan")
        rows.append({
            "model": model, "reasoning_mode": mode,
            "accuracy": accuracy, "n_correct": n_correct, "n_total": len(g),
            "energy_per_correct_j": energy_per_correct,
        })
    return pd.DataFrame(rows).sort_values(["model", "reasoning_mode"])


def try_mixed_model(df: pd.DataFrame):
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("\n(statsmodels not installed - skipping mixed-effects model. "
              "pip install statsmodels to enable this.)")
        return
    capable = df[df["reasoning_mode"].isin(["on", "off"])].copy()
    if capable.empty or capable["model"].nunique() < 2:
        print("\nNot enough reasoning-mode data across models for a mixed model.")
        return
    try:
        m = smf.mixedlm(
            "j_per_out_token ~ reasoning_mode",
            data=capable,
            groups=capable["model"],
        ).fit()
        print("\n── Mixed-effects model: j_per_out_token ~ reasoning_mode, "
              "random intercept per model ──")
        print(m.summary())
    except Exception as e:
        print(f"\nMixed-model fit failed: {e}")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    path = Path(sys.argv[1])
    df = load(path)

    print(f"Loaded {len(df)} prompt-level rows from {path}\n")

    print("═" * 70)
    print("1) Reasoning Overhead Factor (ROF = E[on] / E[off], per model)")
    print("═" * 70)
    rof = reasoning_overhead_factor(df)
    if not rof.empty:
        print(rof.round(3).to_string())

    print("\n" + "═" * 70)
    print("2) Thinking- vs. answer-token/energy split (reasoning_mode=on)")
    print("═" * 70)
    split = thinking_answer_split(df)
    if not split.empty:
        print(split.round(3).to_string())

    print("\n" + "═" * 70)
    print("3) Energy per correct answer (Tier-4 scored subset)")
    print("═" * 70)
    epc = energy_per_correct(df)
    if not epc.empty:
        print(epc.round(3).to_string(index=False))

    try_mixed_model(df)


if __name__ == "__main__":
    main()
