"""graph -> DIMACS CNF for q-colouring (direct encoding: >=1 colour per node, no shared colour on an edge;
node 0 fixed to colour 0 to break the colour symmetry). usage: graph_to_cnf.py graph out.cnf [q]"""
import sys, numpy as np
from common import read_graph
N, e = read_graph(sys.argv[1]); q = int(sys.argv[3]) if len(sys.argv) > 3 else 5
var = lambda v, c: v * q + c + 1
with open(sys.argv[2], "w") as f:
    f.write(f"p cnf {N*q} {N + len(e)*q + 1}\n")
    nodes = np.arange(N)
    np.savetxt(f, np.stack([var(nodes, c) for c in range(q)] + [np.zeros(N, np.int64)], 1), fmt="%d")
    for c in range(q):
        np.savetxt(f, np.stack([-var(e[:, 0], c), -var(e[:, 1], c), np.zeros(len(e), np.int64)], 1), fmt="%d")
    f.write(f"{var(0, 0)} 0\n")
