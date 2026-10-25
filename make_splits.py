"""Write 10 splits into independent sets using the notebook's networkx MIS peeling.
usage: python make_splits.py graph_file splits.txt   (one line of N class labels per split)"""
import re, sys
import numpy as np, networkx as nx
path, out = sys.argv[1], sys.argv[2]
with open(path) as f:
    head = f.readline(); m = re.search(r"N\s*=\s*(\d+)", head)
    N = int(m.group(1)) if m else int(head.split()[0])
    e = np.array(f.read().split(), dtype=np.int64).reshape(-1, 2)
G = nx.Graph(); G.add_nodes_from(range(N)); G.add_edges_from(e.tolist())
lab = np.zeros((10, N), np.int64)
for s in range(10):
    Gs = G.copy(); k = 0
    while Gs.number_of_nodes():
        mis = nx.maximal_independent_set(Gs, seed=s * 1000 + k)
        lab[s, list(mis)] = k; Gs.remove_nodes_from(mis); k += 1
    assert not (lab[s][e[:, 0]] == lab[s][e[:, 1]]).any()
np.savetxt(out, lab, fmt="%d")
print("sets per split:", [int(l.max()) + 1 for l in lab])
