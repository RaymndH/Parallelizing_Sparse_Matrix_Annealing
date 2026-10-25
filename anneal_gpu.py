#!/usr/bin/env python3
"""Independent-set-parallel simulated annealing for planted q-coloring (torch, GPU or CPU).

All seeds run together as a batch (replicas), so on a GPU 10 seeds cost about the same as 1.
Each sweep picks one of the precomputed splits into independent sets and updates every node of
one set at once; nodes in a set share no edges, so this matches sequential updates exactly.
Aq[r, i, c] (# neighbours of node i with color c in replica r) is updated incrementally.

Examples
  # C++ schedule (linear 1.4 -> 0, dT = 5e-5), 10 seeds, greedy splits
  python anneal_gpu.py graph/N-1e4_c-18.0_q-5 --seeds 10 --dT 5e-5
  # geometric schedule with precomputed networkx splits (one line of N class labels per split)
  python anneal_gpu.py g.txt --splits splits.txt --schedule geom --T0 0.6 --T1 0.15 --sweeps 50000
Graph file: first line is N (bare number or "N = 1000, c = 18, q = 5"), then one edge "a b" per line.
"""
import argparse, random, re, time
import numpy as np
import torch


def read_graph(path):
    with open(path) as f:
        head = f.readline()
        m = re.search(r"N\s*=\s*(\d+)", head)
        N = int(m.group(1)) if m else int(head.split()[0])
        e = np.array(f.read().split(), dtype=np.int64).reshape(-1, 2)
    return N, e


def build_csr(N, e):
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    order = np.argsort(src, kind="stable")
    ptr = np.zeros(N + 1, np.int64)
    np.cumsum(np.bincount(src, minlength=N), out=ptr[1:])
    return ptr, dst[order]


def greedy_split(N, ptr_l, nbr_l, rng):
    """Random-order first-fit coloring: class 0 is a maximal independent set, class 1 a maximal
    independent set of the rest, ... (same structure as iterative MIS peeling, one pass)."""
    col = [-1] * N
    for v in rng.permutation(N).tolist():
        used = {col[u] for u in nbr_l[ptr_l[v]:ptr_l[v + 1]]}
        c = 0
        while c in used:
            c += 1
        col[v] = c
    return np.array(col, np.int64)


def prepare_split(labels, ptr, nbr, Q, device):
    """Per class: node indices, and for every edge out of the class its owner (position in the
    class) and neighbour*Q, so the Aq update needs no data-dependent sizes (no GPU syncs)."""
    deg = np.diff(ptr)
    classes = []
    for c in range(labels.max() + 1):
        idx = np.flatnonzero(labels == c)
        if idx.size == 0:
            continue
        cnt = deg[idx]
        owner = np.repeat(np.arange(idx.size), cnt)
        pos = np.repeat(ptr[idx] - (np.cumsum(cnt) - cnt), cnt) + np.arange(cnt.sum())
        rowsQ = nbr[pos] * Q
        classes.append(tuple(torch.as_tensor(a, device=device) for a in (idx, owner, rowsQ)))
    return classes


def fresh_Aq(q, src_t, nbr_t, base, Q, R, NQ):
    Aq = torch.zeros(R * NQ, device=q.device)
    Aq.index_add_(0, (base + src_t * Q + q[:, nbr_t]).reshape(-1),
                  torch.ones(R * nbr_t.numel(), device=q.device))
    return Aq


def class_step(q, Aq, idx, owner, rowsQ, base, Q, T, off, r, active):
    """Metropolis update of every node of one independent set, in all replicas at once.
    off: color offsets in 1..Q-1, r: uniforms, both shaped (R, n_class). Updates q and Aq in place."""
    old = q[:, idx]                                              # (R, n_class)
    new = (old + off) % Q
    bi = base + idx * Q
    dE = Aq[bi + new] - Aq[bi + old]
    if T > 0:
        acc = (dE <= 0) | (r < torch.exp(-dE / T))
    else:
        acc = dE <= 0
    acc &= active
    q[:, idx] = torch.where(acc, new, old)
    # neighbours of each accepted node: -1 at old color, +1 at new color
    w = acc[:, owner].float().reshape(-1)
    bo = base + rowsQ
    Aq.index_add_(0, (bo + old[:, owner]).reshape(-1), -w)
    Aq.index_add_(0, (bo + new[:, owner]).reshape(-1), w)


def planted_graph(N, c, Q, rng):
    """Same construction as plantstate.cpp: random planted colors, c*N/2 distinct edges
    between differently colored nodes."""
    plant = rng.integers(0, Q, N)
    M = int(c * N / 2); edges = set()
    while len(edges) < M:
        a, b = rng.integers(0, N, 2)
        if plant[a] != plant[b]:
            edges.add((min(a, b), max(a, b)))
    return plant, np.array(sorted(edges), np.int64)


def selftest(device, seed=0):
    """1) one class step on GPU == the same moves done one node at a time (numpy, sequential),
       2) incremental Aq == fresh Aq after many sweeps, 3) a planted 1000-node graph gets solved."""
    print(f"selftest on {device}")
    rng = np.random.default_rng(seed); Q = 5
    gen = torch.Generator(device=device).manual_seed(seed)
    N = 1000
    plant, e = planted_graph(N, 18.0, Q, rng)
    ptr, nbr = build_csr(N, e)
    lab = greedy_split(N, ptr.tolist(), nbr.tolist(), rng)
    split = prepare_split(lab, ptr, nbr, Q, device)
    R = 3; NQ = N * Q
    base = (torch.arange(R, device=device) * NQ)[:, None]
    nbr_t = torch.as_tensor(nbr, device=device)
    src_t = torch.repeat_interleave(torch.arange(N, device=device), torch.as_tensor(np.diff(ptr), device=device))
    active = torch.ones(R, 1, dtype=torch.bool, device=device)

    # 1) parallel step vs sequential, same random draws, several temperatures
    for T in (2.0, 0.5, 0.2, 0.0):
        q = torch.randint(0, Q, (R, N), device=device, generator=gen)
        Aq = fresh_Aq(q, src_t, nbr_t, base, Q, R, NQ)
        qs = q.cpu().numpy().copy()
        for idx, owner, rowsQ in split:
            shape = (R, idx.numel())
            off = torch.randint(1, Q, shape, device=device, generator=gen)
            r = torch.rand(shape, device=device, generator=gen)
            class_step(q, Aq, idx, owner, rowsQ, base, Q, T, off, r, active)
            idx_n, off_n, r_n = idx.cpu().numpy(), off.cpu().numpy(), r.cpu().numpy()
            for rep in range(R):
                for j in rng.permutation(idx_n.size):            # any order within the set
                    v = idx_n[j]; o = qs[rep, v]; n = (o + off_n[rep, j]) % Q
                    nb = qs[rep, nbr[ptr[v]:ptr[v + 1]]]
                    dE = float((nb == n).sum() - (nb == o).sum())
                    if dE <= 0 or (T > 0 and r_n[rep, j] < np.exp(np.float32(-dE) / np.float32(T))):
                        qs[rep, v] = n
        assert np.array_equal(q.cpu().numpy(), qs), f"parallel != sequential at T={T}"
        assert torch.equal(Aq, fresh_Aq(q, src_t, nbr_t, base, Q, R, NQ)), f"Aq drifted at T={T}"
    print("  1) parallel step matches sequential updates: OK")

    # 2+3) full anneal on the planted graph; anneal() asserts Aq == fresh at the end
    e_t = torch.as_tensor(e, device=device)
    splits = [prepare_split(greedy_split(N, ptr.tolist(), nbr.tolist(), rng), ptr, nbr, Q, device) for _ in range(5)]
    temps = np.geomspace(0.6, 0.15, 8000).tolist()
    best_q, best, solved_at, used, el = anneal(N, Q, e_t, ptr, nbr, splits, 8, temps, 100,
                                               device, gen, random.Random(seed), verbose=False)
    n_ok = int((solved_at >= 0).sum())
    print(f"  2) incremental Aq consistent after {used} sweeps: OK")
    print(f"  3) planted N=1000: solved {n_ok}/8 seeds, {1e3*el/used:.2f} ms/sweep for 8 seeds")
    assert n_ok >= 1, "no seed solved the planted graph"
    print("selftest passed")


def anneal(N, Q, e_t, ptr, nbr, splits, R, temps, check_every, device, gen, pyrng, verbose=True):
    NQ = N * Q
    base = (torch.arange(R, device=device) * NQ)[:, None]          # replica offset into flat Aq
    nbr_t = torch.as_tensor(nbr, device=device)
    src_t = torch.repeat_interleave(torch.arange(N, device=device),
                                    torch.as_tensor(np.diff(ptr), device=device))
    e0, e1 = e_t[:, 0], e_t[:, 1]

    q = torch.randint(0, Q, (R, N), device=device, generator=gen)
    Aq = fresh_Aq(q, src_t, nbr_t, base, Q, R, NQ)
    conf = lambda: (q[:, e0] == q[:, e1]).sum(1)

    best = conf(); best_q = q.clone()
    solved_at = torch.full((R,), -1, device=device)
    active = torch.ones(R, 1, dtype=torch.bool, device=device)     # solved replicas are frozen
    t0 = time.perf_counter()

    for s, T in enumerate(temps):
        for idx, owner, rowsQ in splits[pyrng.randrange(len(splits))]:
            shape = (R, idx.numel())
            off = torch.randint(1, Q, shape, device=device, generator=gen)
            r = torch.rand(shape, device=device, generator=gen)
            class_step(q, Aq, idx, owner, rowsQ, base, Q, T, off, r, active)

        if (s + 1) % check_every == 0 or s == len(temps) - 1:
            c = conf()
            better = c < best
            best = torch.where(better, c, best)
            best_q[better] = q[better]
            newly = (c == 0) & (solved_at < 0)
            solved_at[newly] = s + 1
            active = (solved_at < 0)[:, None]
            n_solved = int((solved_at >= 0).sum())                   # one sync per check
            if verbose:
                el = time.perf_counter() - t0
                print(f"\rsweep {s+1}/{len(temps)}  T={T:.4f}  conflicts min/med "
                      f"{int(c.min())}/{int(c.float().median())}  solved {n_solved}/{R}  "
                      f"{(s+1)/el:.0f} sweeps/s   ", end="", flush=True)
            if n_solved == R:
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    if verbose:
        print()

    # incremental Aq must equal a fresh computation
    assert torch.equal(Aq, fresh_Aq(q, src_t, nbr_t, base, Q, R, NQ)), "Aq drifted"
    return best_q, best, solved_at, s + 1, elapsed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("graph", nargs="?")
    p.add_argument("--selftest", action="store_true", help="check correctness on this device and exit")
    p.add_argument("--q", type=int, default=5)
    p.add_argument("--seeds", type=int, default=10, help="replicas run as one batch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--schedule", choices=["linear", "geom"], default="linear")
    p.add_argument("--T0", type=float, default=1.4)
    p.add_argument("--T1", type=float, default=0.0)
    p.add_argument("--dT", type=float, default=5e-5, help="linear step (ignored if --sweeps given)")
    p.add_argument("--sweeps", type=int, default=None)
    p.add_argument("--splits", default=None, help="file of class labels, one line of N ints per split")
    p.add_argument("--n-splits", type=int, default=10, help="greedy splits if --splits not given")
    p.add_argument("--check-every", type=int, default=100)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    p.add_argument("--out", default="coloring.txt")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else torch.device(a.device)
    if a.threads:
        torch.set_num_threads(a.threads)
    if device.type == "cuda":
        if device.index is not None:
            torch.cuda.set_device(device.index)
        print(f"GPU {torch.cuda.current_device()}: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    else:
        print(f"CPU, torch {torch.__version__} (no CUDA device found)" if a.device == "auto" else "CPU")
    if a.selftest:
        selftest(device)
        return
    if a.graph is None:
        p.error("graph file required (or --selftest)")
    gen = torch.Generator(device=device).manual_seed(a.seed)
    pyrng = random.Random(a.seed)
    nprng = np.random.default_rng(a.seed)

    t = time.perf_counter()
    N, e = read_graph(a.graph)
    ptr, nbr = build_csr(N, e)
    if a.splits:
        labels = np.atleast_2d(np.loadtxt(a.splits, dtype=np.int64))
        assert labels.shape[1] == N, "splits file does not match N"
    else:
        ptr_l, nbr_l = ptr.tolist(), nbr.tolist()
        labels = np.stack([greedy_split(N, ptr_l, nbr_l, nprng) for _ in range(a.n_splits)])
    for lab in labels:
        assert not (lab[e[:, 0]] == lab[e[:, 1]]).any(), "a split class is not an independent set"
    splits = [prepare_split(lab, ptr, nbr, a.q, device) for lab in labels]
    e_t = torch.as_tensor(e, device=device)

    if a.sweeps is None:
        a.sweeps = int(round(abs(a.T0 - a.T1) / a.dT)) + 1 if a.schedule == "linear" else 10000
    temps = (np.linspace(a.T0, a.T1, a.sweeps) if a.schedule == "linear"
             else np.geomspace(a.T0, a.T1, a.sweeps)).tolist()
    print(f"N={N} edges={len(e)} q={a.q} device={device} seeds={a.seeds} "
          f"sets/split={[int(l.max()) + 1 for l in labels]} {a.schedule} {a.T0}->{a.T1} "
          f"over {a.sweeps} sweeps  (setup {time.perf_counter()-t:.1f}s)")

    best_q, best, solved_at, used, el = anneal(N, a.q, e_t, ptr, nbr, splits, a.seeds, temps,
                                               a.check_every, device, gen, pyrng)
    best, solved_at = best.cpu().numpy(), solved_at.cpu().numpy()
    for r in range(a.seeds):
        print(f"seed {r}: conflicts={best[r]}" + (f"  solved at sweep {solved_at[r]}" if solved_at[r] >= 0 else ""))
    ok = solved_at >= 0
    print(f"solved {ok.sum()}/{a.seeds}  ran {used} sweeps in {el:.1f}s ({1e3*el/used:.3f} ms/sweep for all seeds)")
    if ok.any():
        qs = best_q[torch.as_tensor(ok, device=device)].cpu().numpy()
        assert not (qs[:, e[:, 0]] == qs[:, e[:, 1]]).any()
        np.savetxt(a.out, qs.T, fmt="%d")
        print(f"saved {ok.sum()} valid colorings to {a.out} (one column per solved seed)")


if __name__ == "__main__":
    main()
