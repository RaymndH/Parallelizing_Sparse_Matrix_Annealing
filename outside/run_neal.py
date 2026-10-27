"""D-Wave neal simulated annealing on the one-hot QUBO. usage: run_neal.py graph num_sweeps [num_reads]"""
import sys, time, numpy as np, dimod, neal
from common import read_graph, conflicts
N, e = read_graph(sys.argv[1]); q = 5; sweeps = int(sys.argv[2]); reads = int(sys.argv[3]) if len(sys.argv) > 3 else 1
P = 2.0     # one-hot penalty weight
lin = {}; quad = {}
for v in range(N):
    for c in range(q):
        lin[v*q+c] = -P
        for c2 in range(c+1, q): quad[(v*q+c, v*q+c2)] = 2*P
for a, b in e.tolist():
    for c in range(q): quad[(a*q+c, b*q+c)] = 1.0
bqm = dimod.BinaryQuadraticModel(lin, quad, P*N, "BINARY")
t = time.time(); ss = neal.SimulatedAnnealingSampler().sample(bqm, num_reads=reads, num_sweeps=sweeps, seed=1); el = time.time() - t
best = ss.first.sample; x = np.array([[best[v*q+c] for c in range(q)] for v in range(N)])
bad_nodes = int((x.sum(1) != 1).sum()); col = x.argmax(1)
print(f"neal: {sweeps} sweeps in {el:.1f}s, nodes not one-hot {bad_nodes}, conflicts after decoding {conflicts(e, col)}")
