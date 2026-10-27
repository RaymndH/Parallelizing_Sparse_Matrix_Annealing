"""check a kissat/cadical solution: decode_cnf.py graph solver_output [q]"""
import sys, numpy as np
from common import read_graph, conflicts
N, e = read_graph(sys.argv[1]); q = int(sys.argv[3]) if len(sys.argv) > 3 else 5
vals = {}
for line in open(sys.argv[2]):
    if line.startswith("v"):
        for t in line.split()[1:]:
            t = int(t)
            if t: vals[abs(t)] = t > 0
col = np.array([next((c for c in range(q) if vals.get(v * q + c + 1)), 0) for v in range(N)])
print("conflicts after decoding:", conflicts(e, col))
