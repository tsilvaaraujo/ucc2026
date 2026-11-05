"""Tokenize the training partition of every LLM client, as the clients did.

Replicates the client data path of enerfl-benchmarking/llm_pipeline.py:
flwrlabs/alpaca-gpt4 split into 64 IID partitions (FederatedDataset default
shuffle, seed 42), an 80/20 train/test split with seed 42, the same prompt
template, and the OpenLLaMA slow tokenizer. Writes the token length (with BOS,
without EOS) of every training example of every client.

Usage:
    python 1_tokenize_partitions.py [lengths.json]

Downloads the dataset and tokenizer to the Hugging Face cache on first use.
"""

import json
import sys

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from transformers import AutoTokenizer

NUM_CLIENTS = 64


def fmt(ex):
    m = ("Below is an instruction that describes a task. "
         "Write a response that appropriately completes the request.")
    ins, inp = ex["instruction"], ex.get("input", "")
    resp = ex.get("response") or ex.get("output") or ""
    if inp:
        return f"{m}\n### Instruction:\n{ins}\n### Input:\n{inp}\n### Response: {resp}"
    return f"{m}\n### Instruction:\n{ins}\n### Response: {resp}"


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "lengths.json"
    tok = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b",
                                        use_fast=False, padding_side="right")
    fds = FederatedDataset(dataset="flwrlabs/alpaca-gpt4",
                           partitioners={"train": IidPartitioner(num_partitions=NUM_CLIENTS)})
    out = {}
    for pid in range(NUM_CLIENTS):
        d = fds.load_partition(pid, "train")
        if "output" in d.column_names:
            d = d.rename_column("output", "response")
        tr = d.train_test_split(test_size=0.2, seed=42)["train"]
        out[pid] = {"n": len(tr),
                    "len_bos": [len(tok(fmt(ex))["input_ids"]) for ex in tr]}
        print(pid, len(tr), flush=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main()
