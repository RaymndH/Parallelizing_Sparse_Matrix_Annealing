"""CP-SAT q-colouring. usage: run_cpsat.py graph time_limit_s [workers]"""
import sys, time, numpy as np
from ortools.sat.python import cp_model
from common import read_graph, conflicts
N, e = read_graph(sys.argv[1]); q = 5; tl = float(sys.argv[2]); workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
m = cp_model.CpModel(); x = [[m.NewBoolVar("") for c in range(q)] for v in range(N)]
for v in range(N): m.AddExactlyOne(x[v])
for a, b in e.tolist():
    for c in range(q): m.AddBoolOr([x[a][c].Not(), x[b][c].Not()])
m.Add(x[0][0] == 1)
s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = tl; s.parameters.num_workers = workers
t = time.time(); r = s.Solve(m); el = time.time() - t
if r in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    col = np.array([next(c for c in range(q) if s.Value(x[v][c])) for v in range(N)]); print(f"CP-SAT: {s.StatusName(r)} in {el:.1f}s, conflicts {conflicts(e, col)}")
else: print(f"CP-SAT: {s.StatusName(r)} after {el:.1f}s (no solution)")
