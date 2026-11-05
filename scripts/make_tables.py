#!/usr/bin/env python3
"""Regenerate the LaTeX results tables (cost, breakdown, intrusion).

Usage:
    python scripts/make_tables.py results --outdir tables

``<results_root>`` is the folder containing ``metrics/`` and
``logs/{cnn,lstm,llm}/`` (same layout as ``plot_intrusion.py``).

Data sources
  * Coordinator energy/carbon, traffic, run time and per-round server metrics:
    the profiler-on metrics CSVs.
  * Per-client energy: the ``Train metrics`` / ``Eval metrics`` lines of the
    profiler-on logs (FedAvg-averaged over the cohort). The CSV only keeps the
    evaluate-phase value because it overwrites the train-phase one.
  * Profiler intrusion: paired on/off times from ``analyze_intrusion.collect``.

Conventions
  * "x +/- s" is the mean and sample std across runs (one value per run;
    per-round quantities are averaged over the rounds of each run first).
  * Federation energy = coordinator + n_clients * per-client energy.
  * Federation carbon = federation energy * coordinator carbon intensity.
  * Link utilization assumes a 25 Gbit/s interface (Grid'5000 gros nodes).
  * Intrusion: Hodges-Lehmann estimates of the paired differences, exact
    Wilcoxon signed-rank tests, Holm-adjusted across the three workloads.
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import os
import re
import statistics as st
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from analyze_intrusion import collect

ORDER = ["cnn", "lstm", "llm"]
SHORT = {"cnn": "CNN", "lstm": "LSTM", "llm": "LLM"}
DATASET_TOKEN = {"cnn": "FEMNIST", "lstm": "Sentiment140", "llm": "Alpaca"}
N_CLIENTS = 64
LINE_RATE_BPS = 25e9  # bits per second
RUN_ID_RE = re.compile(r"rin(\d+)r(\d+)on")
LOG_ID_RE = re.compile(r"in(\d+)r(\d+)on")
METRICS_RE = re.compile(r"(Train|Eval) metrics: (\{.*\})")


def _f(row: dict, col: str) -> float | None:
    v = row.get(col)
    if v in (None, ""):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _ms(vals: list[float]) -> tuple[float, float]:
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def _num(x: float, d: int) -> str:
    return rf"\num{{{x:.{d}f}}}"


def _pm(vals: list[float], d: int) -> str:
    m, s = _ms(vals)
    return f"{_num(m, d)} $\\pm$ {_num(s, d)}"


def _mpm(vals: list[float], d: int | None = None) -> str:
    """Math-mode ``$m \\pm s$``; d=None picks 0 decimals for values >= 1000."""
    m, s = _ms(vals)
    if d is None:
        d = 0 if abs(m) >= 1000 else 1
    return f"${m:.{d}f} \\pm {s:.{d}f}$"


def _hodges_lehmann(x: list[float]) -> float:
    x = np.asarray(x)
    i, j = np.triu_indices(len(x))
    return float(np.median((x[i] + x[j]) / 2.0))


def _hl_ci(x: list[float]) -> tuple[float, float, float]:
    """Exact distribution-free CI for the HL estimator (Walsh averages).

    Uses the largest critical value C with P(W <= C) <= 0.025 under the
    exact signed-rank null, i.e. the closest achievable level >= 95 %.
    Returns (lower, upper, confidence level).
    """
    n = len(x)
    w = np.sort([(x[i] + x[j]) / 2.0 for i in range(n) for j in range(i, n)])
    dist = np.zeros(n * (n + 1) // 2 + 1)
    dist[0] = 1.0
    for i in range(1, n + 1):
        nxt = dist.copy()
        nxt[i:] += dist[:-i]
        dist = nxt
    cdf = np.cumsum(dist) / 2.0 ** n
    c = int(np.max(np.where(cdf <= 0.025)[0]))
    return float(w[c]), float(w[len(w) - 1 - c]), float(1.0 - 2.0 * cdf[c])


def _holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def _csv_runs(root: Path, key: str) -> dict[int, list[dict]]:
    out = {}
    for path in glob.glob(str(root / "metrics" / f"*{DATASET_TOKEN[key]}*.csv")):
        m = RUN_ID_RE.search(os.path.basename(path))
        if m:
            with open(path, newline="") as fh:
                out[int(m.group(2))] = list(csv.DictReader(fh))
    return out


def _client_energy_per_run(root: Path, key: str) -> dict[int, float]:
    """Per-client energy (J) per run: sum of train+eval over all rounds."""
    out = {}
    for path in (root / "logs" / key).glob("*.out"):
        m = LOG_ID_RE.search(path.name)
        if not m:
            continue
        total, n = 0.0, 0
        for line in path.read_text(errors="replace").splitlines():
            mm = METRICS_RE.search(line)
            if mm:
                e = ast.literal_eval(mm.group(2)).get("client_energy_joules")
                if e is not None:
                    total += float(e)
                    n += 1
        if n:
            out[int(m.group(2))] = total
    return out


def _round_rows(rows: list[dict]) -> list[dict]:
    """Federated rounds only (drops the initial-evaluation row of the LLM)."""
    return [r for r in rows if _f(r, "client_computation_train") is not None]


def _run_mean(rows: list[dict], col: str) -> float:
    return st.mean(v for r in _round_rows(rows) if (v := _f(r, col)) is not None)


def table_cost(root: Path) -> str:
    lines = []
    for key in ORDER:
        runs = _csv_runs(root, key)
        client = _client_energy_per_run(root, key)
        coord, per_client, fed, carbon, sent, recv, time = ([] for _ in range(7))
        for rep, rows in sorted(runs.items()):
            last = rows[-1]
            e_coord = _f(last, "total_energy_joules")
            if rep not in client:
                raise SystemExit(f"{key} rep {rep}: no client metrics in logs")
            e_fed = e_coord + N_CLIENTS * client[rep]
            intensity = _f(last, "total_emissions_kg") / e_coord  # kg/J
            coord.append(e_coord / 1e3)
            per_client.append(client[rep] / 1e3)
            fed.append(e_fed / 1e6)
            carbon.append(e_fed * intensity * 1e3)
            sent.append(_f(last, "cumulative_bytes_sent") / 1e9)
            recv.append(_f(last, "cumulative_bytes_received") / 1e9)
            time.append(_f(last, "total_time"))
        name = f"{SHORT[key]:<4}"
        lines.append(
            f"{name} & {_pm(coord, 2)} & {_pm(per_client, 2)} & "
            f"{_num(st.mean(fed), 2)} & {_num(st.mean(carbon), 1)} & "
            f"{_pm(sent, 1)} & {_pm(recv, 1)} & {_pm(time, 0)} \\\\")
    body = "\n".join(lines)
    return rf"""\begin{{table*}}[t]
\centering
\caption{{Mean per-run cost by workload ($\pm$ one standard deviation).}}
\label{{tab:cost}}
\footnotesize
\begin{{tabular}}{{lrrrrrrr}}
\toprule
Workload & Coordinator & Per client & Federation & Carbon & Sent & Recv.\ & Time \\
         & (kJ)        & (kJ)       & (MJ)       & (g CO\textsubscript{{2}}) & (GB) & (GB) & (s) \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def table_breakdown(root: Path) -> str:
    cols: dict[str, list[str]] = {k: [] for k in range(8)}
    for key in ORDER:
        runs = list(_csv_runs(root, key).values())
        comp = [_run_mean(r, "client_computation_train") for r in runs]
        net = [_run_mean(r, "network_overhead_train") for r in runs]
        strag = [_run_mean(r, "straggler_index_train") for r in runs]
        c2c = [_run_mean(r, "comm_to_compute_ratio_train") for r in runs]
        eff = [_run_mean(r, "round_efficiency") for r in runs]
        thr = [_run_mean(r, "bandwidth_utilization_bytes_per_sec_train") / 1e6
               for r in runs]
        cols[0].append(_mpm(comp))
        cols[1].append(_mpm(net))
        cols[2].append(_mpm(strag))
        cols[3].append(_num(st.mean(strag) / st.mean(comp) * 100, 1))
        cols[4].append(_mpm(c2c, 3))
        cols[5].append(_mpm(eff, 3))
        cols[6].append(_mpm(thr, 1))
        cols[7].append(_num(st.mean(thr) * 1e6 * 8 / LINE_RATE_BPS * 100, 2))
    labels = [
        r"Client computation (\si{\second})",
        r"Network overhead (\si{\second})  ",
        r"Straggler spread (\si{\second})  ",
        r"\quad as \% of computation       ",
        r"Communication to computation     ",
        r"Round efficiency                 ",
        r"Throughput (\si{\mega\byte\per\second})",
        r"Link utilization (\%)            ",
    ]
    body = "\n".join(f"{labels[i]} & " + " & ".join(cols[i]) + r" \\"
                     for i in range(8))
    return rf"""\begin{{table*}}[t]
\centering
\caption{{Per-round decomposition of the training phase of a federated round.}}
\label{{tab:breakdown}}
\footnotesize
\begin{{tabular}}{{lrrr}}
\toprule
Per round & CNN & LSTM & LLM \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def table_intrusion(root: Path) -> str:
    fams = collect(root)
    stats = []
    for key in ORDER:
        runs = _csv_runs(root, key)
        n_rounds = int(_f(next(iter(runs.values()))[-1], "num_rounds"))
        pairs = [p for p in fams[key].pairs.values() if p.overhead_pct is not None]
        pct = [p.overhead_pct for p in pairs]
        sec = [p.overhead_s for p in pairs]
        p = wilcoxon(sec, alternative="two-sided", method="exact").pvalue
        lo, hi, _ = _hl_ci(sec)
        stats.append({"n": len(pairs), "median": st.median(pct),
                      "hl_pct": _hodges_lehmann(pct),
                      "hl_round": _hodges_lehmann(sec) / n_rounds,
                      "ci": (lo / n_rounds, hi / n_rounds), "p": p})
    for s, padj in zip(stats, _holm([s["p"] for s in stats])):
        s["p_holm"] = padj
    lines = []
    for key, s in zip(ORDER, stats):
        p_txt = f"{s['p_holm']:.3f}"
        if s["p_holm"] < 0.05:
            p_txt = rf"\textbf{{{p_txt}}}"
        lines.append(
            f"{SHORT[key]:<4} & {s['n']:>2} & ${s['median']:+.2f}$ & "
            f"${s['hl_pct']:+.2f}$ & ${s['hl_round']:+.2f}$ & "
            f"$[{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}]$ & {p_txt} \\\\")
    body = "\n".join(lines)
    return rf"""\begin{{table}}[t]
\centering
\caption{{Profiler intrusion on end-to-end run time.}}
\label{{tab:intrusion}}
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrrrrcr}}
\toprule
Workload & $n$ & Median & $\Delta t$ & $\Delta t$/round & 95\,\% CI & $p_\text{{Holm}}$ \\
         &     & (\%)   & (\%)       & (\si{{\second}})   & (\si{{\second}}) &  \\
\midrule
{body}
\bottomrule
\end{{tabular}}

\vspace{{2pt}}
{{\footnotesize $\Delta t$: Hodges--Lehmann estimate of $T_\text{{on}}-T_\text{{off}}$.
CI: exact Walsh-average interval for $\Delta t$/round (95.1\,\% for $n=10$).
$p_\text{{Holm}}$: exact Wilcoxon signed-rank $p$-value, Holm-adjusted.}}
\end{{table}}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()
    outdir = args.outdir or (args.root.parent / "tables")
    outdir.mkdir(parents=True, exist_ok=True)
    for name, fn in (("cost", table_cost), ("breakdown", table_breakdown),
                     ("intrusion", table_intrusion)):
        tex = fn(args.root)
        (outdir / f"{name}.tex").write_text(tex, encoding="utf-8")
        print(f"===== {name}.tex\n{tex}")
    print(f"tables written to {outdir}")


if __name__ == "__main__":
    main()
