# Parallel simulated annealing for planted graph coloring

Simulated annealing for 5-coloring planted Erdős–Rényi graphs (average degree c = 18), parallelized by splitting the graph into independent sets. Nodes in one independent set share no edges, so all of them can be updated at once on a GPU with exactly the same result as updating them one at a time.

Graphs come from `plantstate.cpp` in [SimAnn](https://github.com/RaymndH/SimAnn) (`graph/N-1e3_c-18.0_q-5`, `graph/N-1e4_c-18.0_q-5`).

## Files

| File | What it is |
|---|---|
| `Sparse_Matrix.ipynb` | Original Colab notebook (first GPU version) |
| `anneal_gpu.py` | GPU annealer (torch): independent-set parallel updates, incremental `Aq`, all seeds batched, `--selftest` |
| `seq.c` | Sequential C baseline: random order, or set-by-set order from a splits file |
| `make_splits.py` | Builds 10 independent-set splits with the notebook's networkx MIS peeling |
| `N-1e3_c-18.0_q-5` | 1000-node planted graph |

## Running on atar

```bash
nvidia-smi                                                    # pick an idle GPU
python -c "import torch; print(torch.cuda.is_available())"   # needs a CUDA build of torch
python anneal_gpu.py --selftest --device cuda:0               # < 1 min, checks correctness
python anneal_gpu.py ../SimAnn/graph/N-1e4_c-18.0_q-5 --device cuda:0 --seeds 10 --dT 1e-5
```

`--selftest` checks that one batched GPU step equals the same moves done sequentially, that the incremental `Aq` matches a full recompute, and that a fresh planted 1000-node graph gets solved.

Sequential baseline:

```bash
gcc -O3 -march=native -o seq seq.c -lm
LIN=1 ./seq N-1e3_c-18.0_q-5 1.4 0 28000 10                   # random order, testcode.cpp schedule
python make_splits.py N-1e3_c-18.0_q-5 splits1e3.txt
LIN=1 ./seq N-1e3_c-18.0_q-5 1.4 0 28000 10 splits1e3.txt     # independent-set order
```

## Results so far (sequential C, 2-core sandbox)

| Graph | Schedule | Random order | Independent-set order |
|---|---|---|---|
| 10³ | linear 1.4 → 0, dT = 5e-5 (28k sweeps) | 9/10 solved, ~22,600 sweeps | 10/10 solved, ~22,700 sweeps |
| 10⁴ | linear 1.4 → 0, dT = 5e-5 (28k sweeps) | 0/10 (~1,500 conflicts left) | 1/10 |
| 10⁴ | geometric 0.6 → 0.15, 50k sweeps | 5/10, ~45,500 sweeps | 4/10, ~46,000 sweeps |

- The independent-set order solves as reliably as random order, so the parallel scheme is sound.
- The notebook's schedule (dT = 1e-3, 1,400 epochs) is too fast; it stopped at 185 conflicts on 10³.
- Runs solve near T ≈ 0.27 at 10³ and T ≈ 0.17 to 0.2 at 10⁴; the schedule must cool slowly there.
- Sweeps needed grow with N (time likely ~N² overall), so 10⁴ needs roughly dT = 1e-5 and 10⁵ likely slower still.
- Sequential C: about 15 µs per sweep at 10³ and 136 µs at 10⁴.

## Next steps

1. Run `--selftest` on atar's GPU.
2. Run 10⁴ with `--dT 1e-5`, 10 seeds; compare ms/sweep with `seq.c`.
3. Generate 10⁵ with `plantstate.cpp` (`mant = 5`) and run both.
