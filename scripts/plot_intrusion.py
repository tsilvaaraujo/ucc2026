#!/usr/bin/env python3
"""Generate figures for the EnerFL results section.

Reads the same paired on/off runs as ``analyze_intrusion.py`` and produces:
  * ``intrusion.pdf`` - total run time per model with the profiler overhead
                        highlighted in a separate colour.
  * ``stages.pdf``    - mean time spent in each pipeline stage per model (donuts).
  * ``cost_energy.pdf``  - per-model energy (J).
  * ``cost_carbon.pdf``  - per-model carbon (g CO2).
  * ``cost_scatter.pdf`` - per-run energy vs. run time.

Usage:
    python scripts/plot_intrusion.py results --outdir graphs

``<results_root>`` is the folder containing ``metrics/`` and ``logs/{cnn,lstm,llm}/``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics as st
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analyze_intrusion import FAMILIES, collect

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#222222",
    "axes.labelcolor": "#222222",
    "figure.dpi": 130,
})

ORDER = ["cnn", "lstm", "llm"]
SHORT = {"cnn": "CNN", "lstm": "LSTM", "llm": "LLM"}
DATASET_TOKEN = {"cnn": ("FEMNIST", "EfficientNet"),
                 "lstm": ("Sentiment140", "LSTM"),
                 "llm": ("Alpaca", "OpenLLaMA")}


def _run_totals(path: str) -> dict[str, float]:
    energy = emissions = comm = total_time = None
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("total_energy_joules"):
                energy = float(row["total_energy_joules"])
            if row.get("total_emissions_kg"):
                emissions = float(row["total_emissions_kg"])
            if row.get("cumulative_bytes_total"):
                comm = float(row["cumulative_bytes_total"])
            if row.get("total_time"):
                total_time = float(row["total_time"])
    return {"energy": energy, "emissions": emissions,
            "comm": comm, "time": total_time}


COLORS = {"cnn": "#4C72B0", "lstm": "#55A868", "llm": "#C44E52"}
BASELINE_COLOR = "#B0B7C6"
OVERHEAD_COLOR = "#C44E52"

STAGE_SLICES = [
    ("Client setup", ["configure_train_time", "configure_evaluate_time"]),
    ("Local training", ["client_computation_train"]),
    ("Model transfer", ["network_overhead_train"]),
    ("Aggregation", ["weight_aggregation_train_time", "metric_aggregation_evaluate_time"]),
    ("Evaluation", ["send_receive_evaluate_time", "server_evaluation_time"]),
]
STAGE_COLORS = ["#937860", "#4C72B0", "#DD8452", "#8172B3", "#55A868"]


def _fmt_time(secs: float) -> str:
    if secs >= 3600:
        return f"{secs / 3600:.1f} h"
    if secs >= 60:
        return f"{secs / 60:.0f} min"
    return f"{secs:.0f} s"


def _fmt_secs(secs: float) -> str:
    if secs >= 1000:
        return f"{secs:,.0f} s"
    if secs >= 10:
        return f"{secs:.0f} s"
    if secs >= 1:
        return f"{secs:.1f} s"
    return f"{secs:.2f} s"


def _fmt_minutes(mins: float) -> str:
    if mins >= 100:
        return f"{mins:,.0f} min"
    if mins >= 10:
        return f"{mins:.0f} min"
    return f"{mins:.1f} min"


def _cost_rows(root: Path) -> dict[str, list[dict[str, float]]]:
    """Per-repetition cost totals per family (profiler-on runs only)."""
    out: dict[str, list[dict[str, float]]] = {k: [] for k in ORDER}
    for key in ORDER:
        tokens = DATASET_TOKEN[key]
        for path in glob.glob(str(root / "metrics" / "*.csv")):
            name = os.path.basename(path)
            if not any(tok in name for tok in tokens):
                continue
            t = _run_totals(path)
            if t["energy"] is None:
                continue
            out[key].append({
                "energy_j": t["energy"],
                "carbon_g": (t["emissions"] or 0.0) * 1000.0,
                "comm_gb": (t["comm"] or 0.0) / 1e9,
                "time_min": (t["time"] or 0.0) / 60.0,
            })
    return out


def _round_time_stats(root: Path) -> dict[str, dict[str, list[float]]]:
    """Mean/std of per-round wall-clock time across profiler-on runs."""
    out: dict[str, dict[str, list[float]]] = {}
    for key in ORDER:
        tokens = DATASET_TOKEN[key]
        per_round: dict[int, list[float]] = {}
        for path in glob.glob(str(root / "metrics" / "*.csv")):
            if not any(tok in os.path.basename(path) for tok in tokens):
                continue
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    try:
                        val = float(row["round_total_time"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    per_round.setdefault(int(row["round"]), []).append(val)
        rounds = sorted(per_round)
        out[key] = {
            "rounds": [float(r) for r in rounds],
            "means": [st.mean(per_round[r]) for r in rounds],
            "stds": [st.pstdev(per_round[r]) if len(per_round[r]) > 1 else 0.0
                     for r in rounds],
        }
    return out


def _intrusion_summary(root: Path) -> dict[str, dict[str, float]]:
    """Median baseline/profiled time (s) and median overhead (%) per family."""
    families = collect(root)
    out: dict[str, dict[str, float]] = {}
    for key in ORDER:
        offs = [p.t_off for p in families[key].pairs.values() if p.t_off is not None]
        ons = [p.t_on for p in families[key].pairs.values() if p.t_on is not None]
        pcts = [p.overhead_pct for p in families[key].pairs.values()
                if p.overhead_pct is not None]
        out[key] = {
            "off": st.median(offs) if offs else 0.0,
            "on": st.median(ons) if ons else 0.0,
            "pct": st.median(pcts) if pcts else 0.0,
        }
    return out


def _stage_breakdown(root: Path) -> dict[str, list[float]]:
    """Mean per-run time (s) in each pipeline stage per family."""
    means, _ = _stage_stats(root)
    return means


def _stage_stats(root: Path) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Mean and std of per-run time (s) in each pipeline stage per family."""
    cols = [c for _, group in STAGE_SLICES for c in group]
    means: dict[str, list[float]] = {}
    stds: dict[str, list[float]] = {}
    for key in ORDER:
        tokens = DATASET_TOKEN[key]
        per_run: list[dict[str, float]] = []
        for path in glob.glob(str(root / "metrics" / "*.csv")):
            if not any(tok in os.path.basename(path) for tok in tokens):
                continue
            totals = {c: 0.0 for c in cols}
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    for c in cols:
                        v = row.get(c)
                        if v:
                            try:
                                totals[c] += float(v)
                            except ValueError:
                                pass
            per_run.append(totals)
        m_slices = []
        s_slices = []
        for _, group in STAGE_SLICES:
            vals = [sum(r[c] for c in group) for r in per_run]
            m_slices.append(st.mean(vals) if vals else 0.0)
            s_slices.append(st.stdev(vals) if len(vals) > 1 else 0.0)
        means[key] = m_slices
        stds[key] = s_slices
    return means, stds


def _intrusion_raw(root: Path) -> dict[str, list[float]]:
    """Raw list of profiler overhead (%) per family."""
    families = collect(root)
    return {key: [p.overhead_pct for p in families[key].pairs.values()
                  if p.overhead_pct is not None] for key in ORDER}


def _intrusion_stats(root: Path) -> dict[str, dict[str, float]]:
    """Mean/std/min/max/median of the profiler overhead (%) per family."""
    raw = _intrusion_raw(root)
    out: dict[str, dict[str, float]] = {}
    for key in ORDER:
        pcts = raw[key]
        if not pcts:
            out[key] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                        "median": 0.0, "n": 0}
            continue
        out[key] = {
            "mean": st.mean(pcts),
            "std": st.stdev(pcts) if len(pcts) > 1 else 0.0,
            "min": min(pcts),
            "max": max(pcts),
            "median": st.median(pcts),
            "n": len(pcts),
        }
    return out


def _draw_intrusion(ax, raw, keys, positions, rng, scatter_s=18,
                    mean_hw=0.22, median_ms=5, axhline=True):
    """Draw violin + jittered runs + mean/median markers on a given axis."""
    data = [raw[k] for k in keys]
    parts = ax.violinplot(data, positions=positions, widths=0.72,
                          showextrema=False, showmeans=False,
                          showmedians=False)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(COLORS[keys[i]])
        body.set_edgecolor(COLORS[keys[i]])
        body.set_alpha(0.28)
        body.set_linewidth(1.0)
    for pos, key in zip(positions, keys):
        vals = raw[key]
        jitter = rng.normal(0.0, 0.045, len(vals))
        ax.scatter([pos + j for j in jitter], vals, s=scatter_s,
                   color=COLORS[key], alpha=0.85, edgecolor="white",
                   linewidth=0.4, zorder=3)
        mean = st.mean(vals)
        median = st.median(vals)
        ax.hlines(mean, pos - mean_hw, pos + mean_hw, color=COLORS[key],
                  linewidth=2.0, zorder=4)
        ax.plot(pos, median, marker="D", markersize=median_ms, color="white",
                markeredgecolor=COLORS[key], markeredgewidth=1.2, zorder=5)
    if axhline:
        ax.axhline(0.0, color="#8a8a8a", linewidth=0.8, linestyle="--", zorder=0)


def plot_intrusion(root: Path, outdir: Path) -> None:
    raw = _intrusion_raw(root)
    fig, axes = plt.subplots(1, len(ORDER), figsize=(3.9, 3.1))
    rng = np.random.default_rng(0)
    for ax, key in zip(axes, ORDER):
        _draw_intrusion(ax, raw, [key], [0], rng, scatter_s=16,
                        mean_hw=0.34, median_ms=5)
        ax.set_title(SHORT[key], fontsize=12, fontweight="bold",
                     color=COLORS[key], pad=4)
        ax.set_xticks([])
        ax.set_xlim(-0.75, 0.75)
        ax.margins(y=0.20)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, color="#c9c9c9",
                zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Profiler intrusion (%)")

    handles = [
        Line2D([0], [0], color="#4a4a4a", linewidth=2.0, label="mean"),
        Line2D([0], [0], marker="D", linestyle="", markersize=5,
               markerfacecolor="white", markeredgecolor="#4a4a4a",
               label="median"),
        Line2D([0], [0], marker="o", linestyle="", markersize=5,
               markerfacecolor="#7a7a7a", markeredgecolor="white",
               label="run"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=10.5,
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02),
               handlelength=1.4, columnspacing=1.1)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(outdir / "intrusion.pdf")
    fig.savefig(outdir / "intrusion.png", dpi=130)
    plt.close(fig)


def plot_stages(root: Path, outdir: Path) -> None:
    means, stds = _stage_stats(root)
    labels = [name for name, _ in STAGE_SLICES]
    n_stage = len(labels)
    all_vals = [v for k in ORDER for v in means[k] if v > 0]
    left = min(all_vals) * 0.55
    right = max(all_vals) * 2.4

    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    y = 0.0
    gap = 1.0
    centers = []
    for key in ORDER:
        vals = means[key]
        ys = []
        for si in range(n_stage):
            ax.plot([left, vals[si]], [y, y], color=STAGE_COLORS[si],
                    linewidth=2.6, alpha=0.45, solid_capstyle="round",
                    zorder=1)
            ax.scatter(vals[si], y, s=58, color=STAGE_COLORS[si],
                       edgecolor="white", linewidth=0.9, zorder=3)
            ax.annotate(_fmt_secs(vals[si]), xy=(vals[si], y),
                        xytext=(7, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=7, color="#333333")
            ys.append(y)
            y += 1.0
        centers.append(sum(ys) / len(ys))
        y += gap
    ax.set_xscale("log")
    ax.set_xlim(left, right)
    ax.set_ylim(-0.8, y - gap + 0.2)
    ax.invert_yaxis()
    ax.set_yticks(centers)
    ax.set_yticklabels([SHORT[k] for k in ORDER], fontweight="bold")
    ax.set_xlabel("Mean time per run (s), log scale")
    ax.grid(axis="x", linestyle=":", linewidth=0.5, color="#c9c9c9", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=7,
                      markerfacecolor=STAGE_COLORS[i], markeredgecolor="white",
                      label=labels[i]) for i in range(n_stage)]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, ncol=n_stage,
              loc="upper center", bbox_to_anchor=(0.5, 1.12),
              columnspacing=1.2, handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "stages.pdf")
    fig.savefig(outdir / "stages.png", dpi=130)
    plt.close(fig)


def _tint(hex_color: str, amount: float) -> tuple[float, float, float]:
    """Blend a colour towards white; amount=0 keeps it, amount=1 is white."""
    rgb = matplotlib.colors.to_rgb(hex_color)
    return tuple(c + (1.0 - c) * amount for c in rgb)


def plot_runtime(root: Path, outdir: Path) -> None:
    """Median run time per workload with the profiler off and on."""
    summ = _intrusion_summary(root)
    bar_h = 0.34
    fig, ax = plt.subplots(figsize=(3.9, 3.1))
    centers = list(range(len(ORDER)))
    max_on = max(summ[k]["on"] for k in ORDER)
    for i, key in enumerate(ORDER):
        off_s = summ[key]["off"]
        on_s = summ[key]["on"]
        pct = summ[key]["pct"]
        color = COLORS[key]
        ax.barh(i + bar_h / 2 + 0.02, off_s, bar_h, facecolor=_tint(color, 0.30),
                edgecolor=color, linewidth=0.8, hatch="////", zorder=2)
        ax.barh(i - bar_h / 2 - 0.02, on_s, bar_h, facecolor=color,
                edgecolor="white", linewidth=0.5, zorder=2)
        ax.annotate(_fmt_secs(on_s), xy=(on_s, i), xytext=(6, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=10.5, color="#333333",
                    annotation_clip=False)
        ax.annotate(f"{pct:+.2f}%", xy=(on_s, i), xytext=(6, -11),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=10.5, fontweight="bold", color=color,
                    annotation_clip=False)
    ax.set_xscale("log")
    ax.set_xlim(right=max_on * 4.0)
    ax.set_xlabel("Median run time (s), log scale")
    ax.set_yticks(centers)
    ax.set_yticklabels([SHORT[k] for k in ORDER], fontweight="bold")
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, color="#c9c9c9", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    handles = [
        Patch(facecolor="#dedede", edgecolor="#555555", linewidth=0.8,
              hatch="////", label="profiler off"),
        Patch(facecolor="#dedede", edgecolor="#555555", linewidth=0.8,
              label="profiler on"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=11,
              loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.13),
              handlelength=1.4, columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout()
    fig.savefig(outdir / "runtime.pdf")
    fig.savefig(outdir / "runtime.png", dpi=130)
    plt.close(fig)


def plot_rounds(root: Path, outdir: Path) -> None:
    """Per-round wall-clock time across the federated rounds of each run."""
    stats = _round_time_stats(root)
    fig, ax = plt.subplots(figsize=(3.9, 3.1))
    for key in ORDER:
        rounds = stats[key]["rounds"]
        means = np.array(stats[key]["means"])
        stds = np.array(stats[key]["stds"])
        color = COLORS[key]
        ax.fill_between(rounds, means - stds, means + stds, color=color,
                        alpha=0.18, linewidth=0, zorder=1)
        ax.plot(rounds, means, color=color, linewidth=1.6, marker="o",
                markersize=5, markeredgecolor="white", markeredgewidth=0.7,
                label=SHORT[key], zorder=3)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Round time (s), log scale")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, color="#c9c9c9", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=11, loc="center right",
              bbox_to_anchor=(1.0, 0.70), handlelength=1.4, labelspacing=0.4)
    fig.tight_layout()
    fig.savefig(outdir / "rounds.pdf")
    fig.savefig(outdir / "rounds.png", dpi=130)
    plt.close(fig)


def _fmt_mean_std(vals: list[float], digits: int = 2) -> str:
    if not vals:
        return "n/a"
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.{digits}f} +/- {s:.{digits}f}"


def print_stats(root: Path) -> None:
    """Dump per-workload cost and intrusion statistics for the results table."""
    rows = _cost_rows(root)
    intr = _intrusion_stats(root)
    print("\n=== Cost per run (mean +/- std) ===")
    for key in ORDER:
        r = rows[key]
        n = len(r)
        energy = [x["energy_j"] for x in r]
        carbon = [x["carbon_g"] for x in r]
        comm = [x["comm_gb"] for x in r]
        tmin = [x["time_min"] for x in r]
        print(f"\n{SHORT[key]} (n={n})")
        print(f"  energy (kJ)  : {_fmt_mean_std([e / 1000 for e in energy])}")
        print(f"  carbon (g)   : {_fmt_mean_std(carbon, 3)}")
        print(f"  comm (GB)    : {_fmt_mean_std(comm, 1)}")
        print(f"  time (min)   : {_fmt_mean_std(tmin, 1)}")
    print("\n=== Profiler intrusion (%) ===")
    for key in ORDER:
        s = intr[key]
        print(f"{SHORT[key]:5s} n={s['n']:2d}  mean={s['mean']:+.2f}  "
              f"std={s['std']:.2f}  median={s['median']:+.2f}  "
              f"min={s['min']:+.2f}  max={s['max']:+.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()
    outdir = args.outdir or (args.root.parent / "graphs")
    outdir.mkdir(parents=True, exist_ok=True)
    plot_intrusion(args.root, outdir)
    plot_runtime(args.root, outdir)
    plot_stages(args.root, outdir)
    plot_rounds(args.root, outdir)
    print_stats(args.root)
    print(f"\nfigures written to {outdir}")


if __name__ == "__main__":
    main()
