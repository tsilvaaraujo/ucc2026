"""Find the sampler/tokenization variant that reproduces the token count of a
client checkpoint (num_tokens = 35313, partition 16): seed 42, EOS appended,
truncation at 1024 tokens (TRL default; the max_seq_length = 512 assignment in
llm_pipeline.py has no effect).

Usage: python 2_match_checkpoint.py lengths.json
"""
import json, sys, torch
L = {int(k): v for k, v in json.load(open(sys.argv[1])).items()}
seeds = {"g42": 42, "fresh_gen": torch.Generator().initial_seed()}
TARGET = 35313
for sname, seed in seeds.items():
    for eos in (0, 1):
        for trunc in (512, 1024, 10**9):
            hits = []
            for pid, d in L.items():
                g = torch.Generator(); g.manual_seed(seed)
                idx = torch.randperm(d["n"], generator=g).tolist()[:160]
                ntok = sum(min(d["len_bos"][i] + eos, trunc) for i in idx)
                if ntok == TARGET: hits.append(pid)
            print(f"{sname:9s} eos={eos} trunc={trunc:>10}: partitions matching 35313 -> {hits}")
