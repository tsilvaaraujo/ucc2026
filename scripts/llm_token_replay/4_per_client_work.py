"""Padded tokens per client and round (10 steps x 16 examples, per-batch
padding) and comparison with the measured LLM local training times. The
measured values below come from results/metrics and results/logs/llm
(rounds 2-5, profiler-on runs).

Usage: python 4_per_client_work.py lengths.json
"""
import json, sys, statistics as st, torch
L = {int(k): v for k, v in json.load(open(sys.argv[1])).items()}
rows = []
for pid, d in sorted(L.items()):
    g = torch.Generator(); g.manual_seed(42)
    idx = torch.randperm(d["n"], generator=g).tolist()[:160]
    lens = [min(d["len_bos"][i] + 1, 1024) for i in idx]
    mx = [max(lens[b*16:(b+1)*16]) for b in range(10)]
    rows.append({"pid": pid, "real": sum(lens), "padded": sum(16*m for m in mx),
                 "quad": sum(16*m*m for m in mx), "trunc": sum(l == 1024 for l in lens)})
P = [r["padded"] for r in rows]; R = [r["real"] for r in rows]
mxP, mnP, meP = max(P), min(P), st.mean(P)
print(f"padded tokens/client: min {mnP} mean {meP:.0f} max {mxP}  (min/max {mnP/mxP:.3f}, mean/max {meP/mxP:.3f})")
print(f"real tokens/client:   min {min(R)} mean {st.mean(R):.0f} max {max(R)}")
print(f"examples truncated at 1024: {sum(r['trunc'] for r in rows)} of {64*160}; real tokens > 512: computed below")
fixed = 25.8 + 2.7 + 0.03 + 0.29
t_max = 3963.0 - fixed
t_min = t_max - 1066.0
t_mean = 3261.7
print(f"measured local training: min {t_min:.0f} mean {t_mean:.0f} max {t_max:.0f}  (min/max {t_min/t_max:.3f}, mean/max {t_mean/t_max:.3f})")
b = (t_max - t_min) / (mxP - mnP); a = t_max - b*mxP
print(f"linear fit t = {a:.0f} s + {b*1000:.2f} ms/token -> predicted mean {a + b*meP:.0f} s (measured {t_mean:.0f})")
print(f"predicted mean waiting max-mean: {b*(mxP-meP):.0f} s/round (measured ~680 incl. round 1)")
rows.sort(key=lambda r: r["padded"])
print("fastest:", [(r["pid"], r["padded"]) for r in rows[:3]], " slowest:", [(r["pid"], r["padded"]) for r in rows[-3:]])
over512 = 0
for pid, d in L.items():
    g = torch.Generator(); g.manual_seed(42)
    idx = torch.randperm(d["n"], generator=g).tolist()[:160]
    over512 += sum(d["len_bos"][i] + 1 > 512 for i in idx)
print(f"sampled examples longer than 512 tokens: {over512} of {64*160} ({over512/(64*160)*100:.1f} %)")
