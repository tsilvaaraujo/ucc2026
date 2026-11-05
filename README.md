# Per-Round Profiling of Energy, Carbon, and Communication in Federated Learning with EnerFL

Thiago da Silva Araújo<sup>1</sup>, Cláudia Brito<sup>2</sup> and Philippe Olivier Alexandre Navaux<sup>1</sup>

> <sup>1</sup> Informatics Institute, Federal University of Rio Grande do Sul (UFRGS), Porto Alegre, Brazil
<sup>2</sup> INESC TEC and University of Minho, Portugal

This repository contains the artifact of our UCC 2026 article: the code used to run the
experiments, the raw measurements, and the scripts that produce every figure and table
in the paper.

## Repository layout

```
ucc2026/
├── enerfl-benchmarking/     # submodule: experiment code (Flower apps, Singularity images, job scripts)
│   └── enerfl/              #   nested submodule: the EnerFL profiler, release v0.1.0
├── results/
│   ├── logs/{cnn,lstm,llm}/ # job logs of all 60 runs (profiler on and off)
│   └── metrics/             # per-round profiler CSVs of the 30 profiler-on runs
├── graphs/                  # Figures 2-4 (PDF and PNG)
├── tables/                  # Tables III-V (LaTeX)
└── scripts/
    ├── analyze_intrusion.py # paired on/off overhead per workload (console report)
    ├── plot_intrusion.py    # Figures 2a, 2b, 3, and 4
    ├── make_tables.py       # Tables III, IV, and V
    └── llm_token_replay/    # offline replay of the LLM client batches (Section IV-D)
```

## Getting the code

```bash
git clone --recursive https://github.com/tsilvaaraujo/ucc2026.git
# or, in an existing clone:
git submodule update --init --recursive
```

| Component | Repository | Version |
|---|---|---|
| EnerFL profiler | [tsilvaaraujo/enerfl](https://github.com/tsilvaaraujo/enerfl) | `v0.1.0` (`2238bf7`) |
| Experiment code | [tsilvaaraujo/enerfl-benchmarking](https://github.com/tsilvaaraujo/enerfl-benchmarking) | `1721148` |

## Experiments

All runs use synchronous FedAvg with 64 clients and one server, one process per node,
for five rounds, on the `gros` cluster of Grid'5000 (Nancy). Each workload has ten
pairs of runs, one with the profiler enabled and one without it, executed strictly one
at a time.

| Workload | Model | Dataset | Log folder | CSV prefix |
|---|---|---|---|---|
| CNN  | EfficientNet-B0 | FEMNIST       | `results/logs/cnn`  | `metrics_FEMNIST_FedAvg_EfficientNet-B0_c64_*` |
| LSTM | 2-layer LSTM    | Sentiment140  | `results/logs/lstm` | `metrics_Sentiment140_FedAvg_LSTM_c64_*` |
| LLM  | OpenLLaMA-7B (4-bit, LoRA r=32) | Alpaca-GPT4 | `results/logs/llm` | `metrics_Alpaca-GPT4_FedAvg_OpenLLaMA_c64_*` |

Run names follow `enerfl-in<i>r<rep><on|off>.<job_id>`: `<rep>` is the repetition (1-10),
`on`/`off` is the profiler mode, and runs `in<2k-1>` and `in<2k>` form pair `k`.
Profiler-on runs write a metrics CSV with the same job id; profiler-off runs only write a
log, from which the run time is read (`Strategy execution finished in Xs`).

The experiments are launched with `enerfl-benchmarking/scripts/run_intrusion_experiment.sh`
(see the `enerfl-benchmarking` README for building the Singularity images and choosing
the scheduler backend).

## Reproducing the figures and tables

The analysis only needs the files in `results/`:

```bash
pip install -r requirements.txt
cd scripts
python analyze_intrusion.py ../results                    # overhead per pair and workload
python plot_intrusion.py    ../results --outdir ../graphs  # Figures 2-4
python make_tables.py       ../results --outdir ../tables  # Tables III-V
```

| Paper | File |
|---|---|
| Figure 2a | `graphs/runtime.pdf` |
| Figure 2b | `graphs/intrusion.pdf` |
| Figure 3  | `graphs/stages.pdf` |
| Figure 4  | `graphs/rounds.pdf` |
| Table III | `tables/intrusion.tex` |
| Table IV  | `tables/cost.tex` |
| Table V   | `tables/breakdown.tex` |

### LLM token replay

`scripts/llm_token_replay/` replays, offline, the batches each LLM client trains on in a
round, and shows that per-batch padding explains the waiting observed on identical nodes
(Section IV-D). `lengths.json` is included, so steps 2-4 run without downloads:

```bash
cd scripts/llm_token_replay
python 1_tokenize_partitions.py lengths.json   # optional: rebuild lengths.json (downloads data and tokenizer)
python 2_match_checkpoint.py lengths.json      # reproduces the checkpoint token count (35,313)
python 3_check_flops.py lengths.json           # reproduces the checkpoint FLOP count
python 4_per_client_work.py lengths.json       # padded tokens per client vs. measured times
```

Step 1 additionally needs `torch`, `transformers`, `sentencepiece`, and `flwr-datasets`.

## Notes on the measurements

- **Energy is a CodeCarbon estimate.** The RAPL counters were not readable from the
  unprivileged jobs, so CodeCarbon 3.3.0 used its load-based model:
  CPU power = TDP x (0.1 + 0.9 u^3) with TDP = 125 W, plus 38 W of RAM. All energy and
  carbon values are model-based estimates of CPU and RAM energy (Section III-C).
- **Client energy covers only the `train` and `evaluate` tasks.** Client trackers do not
  run while a client waits, so client and federation energies are lower bounds.
- **LLM sequence length.** `llm_pipeline.py` assigns `max_seq_length = 512` after the
  `SFTConfig` is built, which has no effect; training used the TRL default of 1024 tokens.
  The paper reports 1024.
- **Model quality is not evaluated.** The study measures cost and overhead over five
  rounds and does not train to convergence; the LSTM evaluation loss stays at ln 2
  (chance level) in these runs.
