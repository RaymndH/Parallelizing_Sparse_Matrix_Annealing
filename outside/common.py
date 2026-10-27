import re, numpy as np
def read_graph(path):
    with open(path) as f:
        head = f.readline(); m = re.search(r"N\s*=\s*(\d+)", head)
        N = int(m.group(1)) if m else int(head.split()[0])
        e = np.array(f.read().split(), dtype=np.int64).reshape(-1, 2)
    return N, e
def conflicts(e, col): return int((col[e[:, 0]] == col[e[:, 1]]).sum())
