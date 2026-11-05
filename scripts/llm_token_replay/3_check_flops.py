"""Check the replay against the total_flos of the same checkpoint. The implied
parameter count is 6,624,120,832 = OpenLLaMA-7B without embeddings
(6,607,343,616) + LoRA adapters (16,777,216).

Usage: python 3_check_flops.py lengths.json
"""
import json, sys, torch
L = {int(k): v for k, v in json.load(open(sys.argv[1])).items()}
d = L[16]
g = torch.Generator(); g.manual_seed(42)
idx = torch.randperm(d["n"], generator=g).tolist()[:160]
lens = [min(d["len_bos"][i] + 1, 1024) for i in idx]
padded = sum(16 * max(lens[b*16:(b+1)*16]) for b in range(10))
print("real tokens", sum(lens), " padded tokens", padded)
N = 3744906967646208 / (6 * padded)
print(f"implied parameter count N = {N:,.1f}")
