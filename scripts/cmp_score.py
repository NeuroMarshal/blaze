# External test client (not part of the engine). import json, statistics, sys
b = {x["pos"]: x for x in map(json.loads, open(sys.argv[1]))}
r = {x["pos"]: x for x in map(json.loads, open(sys.argv[2]))}
pos = sorted(set(b) & set(r))
assert all(b[p]["target"] == r[p]["target"] for p in pos), "token mismatch"
d = [abs(b[p]["logprob"] - r[p]["logprob"]) for p in pos]
agree = sum(b[p]["argmax"] == r[p]["argmax"] for p in pos) / len(pos)
print(f"{len(pos)} positions: argmax agreement {agree:.3f}, mean |dlogprob| {statistics.mean(d):.4f}, median {statistics.median(d):.4f}, max {max(d):.3f}")
print(f"mean logprob: blaze {statistics.mean(b[p]['logprob'] for p in pos):.4f}, reference {statistics.mean(r[p]['logprob'] for p in pos):.4f}")
for lo in range(0, len(pos), 512):
    seg = pos[lo:lo + 512]
    print(f"  positions {seg[0]}-{seg[-1]}: agreement {sum(b[p]['argmax'] == r[p]['argmax'] for p in seg) / len(seg):.3f}, mean |d| {statistics.mean(abs(b[p]['logprob'] - r[p]['logprob']) for p in seg):.4f}")
