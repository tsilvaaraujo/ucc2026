#!/usr/bin/env python3
"""Analyze EnerFL profiler intrusion (overhead) from paired on/off runs.

For each repetition the experiment runs the same federated workload twice:
  * ``on``  - profiler enabled  -> total run time is written to the metrics CSV
              (``total_time`` column, populated on the final round).
  * ``off`` - baseline, no profiler -> no CSV is produced, so the total run
              time is read from the log line ``Strategy execution finished in Xs``.

The intrusion is the extra wall-clock time the profiler adds relative to the
baseline:  intrusion% = (T_on - T_off) / T_off * 100.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

LOG_ID_RE = re.compile(r"in(\d+)r(\d+)(on|off)", re.IGNORECASE)
CSV_ID_RE = re.compile(r"rin(\d+)r(\d+)(on|off)", re.IGNORECASE)
STRATEGY_RE = re.compile(r"Strategy execution finished in ([\d.]+)s")

FAMILIES = {"cnn": "CNN (EfficientNet-B0 / FEMNIST)",
            "lstm": "LSTM (Sentiment140)",
            "llm": "LLM (OpenLLaMA / Alpaca-GPT4)"}


@dataclass
class Pair:
    rep: int
    t_on: float | None = None
    t_off: float | None = None
    energy_on: float | None = None
    emissions_on: float | None = None

    @property
    def overhead_s(self) -> float | None:
        if self.t_on is None or self.t_off is None:
            return None
        return self.t_on - self.t_off

    @property
    def overhead_pct(self) -> float | None:
        if self.t_on is None or self.t_off is None or self.t_off == 0:
            return None
        return (self.t_on - self.t_off) / self.t_off * 100.0


@dataclass
class Family:
    key: str
    pairs: dict[int, Pair] = field(default_factory=dict)

    def pair(self, rep: int) -> Pair:
        return self.pairs.setdefault(rep, Pair(rep=rep))


def _last_total_time(csv_path: Path) -> tuple[float | None, float | None, float | None]:
    """Return (total_time, total_energy_joules, total_emissions_kg) from a CSV."""
    total_time = energy = emissions = None
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("total_time"):
                try:
                    total_time = float(row["total_time"])
                except ValueError:
                    pass
            if row.get("total_energy_joules"):
                try:
                    energy = float(row["total_energy_joules"])
                except ValueError:
                    pass
            if row.get("total_emissions_kg"):
                try:
                    emissions = float(row["total_emissions_kg"])
                except ValueError:
                    pass
    return total_time, energy, emissions


def collect(root: Path) -> dict[str, Family]:
    families: dict[str, Family] = {k: Family(k) for k in FAMILIES}

    for csv_path in sorted((root / "metrics").glob("*.csv")):
        m = CSV_ID_RE.search(csv_path.name)
        if not m:
            continue
        rep, mode = int(m.group(2)), m.group(3).lower()
        if mode != "on":
            continue
        fam = _family_for_dataset(csv_path.name)
        if fam is None:
            continue
        t, energy, emissions = _last_total_time(csv_path)
        p = families[fam].pair(rep)
        p.t_on, p.energy_on, p.emissions_on = t, energy, emissions

    for fam_key in FAMILIES:
        for log_path in sorted((root / "logs" / fam_key).glob("*.out")):
            m = LOG_ID_RE.search(log_path.name)
            if not m:
                continue
            rep, mode = int(m.group(2)), m.group(3).lower()
            if mode != "off":
                continue
            text = log_path.read_text(errors="replace")
            sm = STRATEGY_RE.search(text)
            if sm:
                families[fam_key].pair(rep).t_off = float(sm.group(1))
    return families


def _family_for_dataset(name: str) -> str | None:
    low = name.lower()
    if "femnist" in low or "efficientnet" in low:
        return "cnn"
    if "sentiment140" in low or "lstm" in low:
        return "lstm"
    if "openllama" in low or "alpaca" in low:
        return "llm"
    return None


def _fmt(x: float | None, unit: str = "") -> str:
    return "n/a" if x is None else f"{x:.2f}{unit}"


def report(families: dict[str, Family]) -> None:
    all_pct: list[float] = []
    for key, label in FAMILIES.items():
        fam = families[key]
        rows = [fam.pairs[r] for r in sorted(fam.pairs)]
        print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
        print(f"{'rep':>3} | {'off (s)':>10} | {'on (s)':>10} | "
              f"{'Δ (s)':>9} | {'intrusion %':>11}")
        print("-" * 60)
        pcts: list[float] = []
        for p in rows:
            print(f"{p.rep:>3} | {_fmt(p.t_off):>10} | {_fmt(p.t_on):>10} | "
                  f"{_fmt(p.overhead_s):>9} | {_fmt(p.overhead_pct, '%'):>11}")
            if p.overhead_pct is not None:
                pcts.append(p.overhead_pct)
        if pcts:
            all_pct.extend(pcts)
            mean = statistics.mean(pcts)
            med = statistics.median(pcts)
            sd = statistics.stdev(pcts) if len(pcts) > 1 else 0.0
            print("-" * 60)
            print(f"paired reps: {len(pcts):>2}   "
                  f"mean: {mean:+.2f}%   median: {med:+.2f}%   "
                  f"std: {sd:.2f}%   min: {min(pcts):+.2f}%   max: {max(pcts):+.2f}%")
        else:
            print("  (no complete on/off pairs)")

    print(f"\n{'#' * 78}\nOVERALL\n{'#' * 78}")
    if all_pct:
        print(f"paired reps: {len(all_pct)}   "
              f"mean intrusion: {statistics.mean(all_pct):+.2f}%   "
              f"median: {statistics.median(all_pct):+.2f}%   "
              f"std: {statistics.stdev(all_pct):.2f}%")
    else:
        print("No paired runs found.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path,
                    help="folder containing metrics/ and logs/{cnn,lstm,llm}/")
    args = ap.parse_args()
    report(collect(args.root))


if __name__ == "__main__":
    main()
