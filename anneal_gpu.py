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


def fresh_G(q, y, N, Q):
    """G[g, i, c]: # replicas of group g (y consecutive replicas) with color c at node i."""
    R = q.shape[0]
    gb = ((torch.arange(R, device=q.device) // y) * (N * Q))[:, None]
    G = torch.zeros((R // y) * N * Q, device=q.device)
    G.index_add_(0, (gb + torch.arange(N, device=q.device)[None, :] * Q + q).reshape(-1),
                 torch.ones(R * N, device=q.device))
    return G


def class_step(q, Aq, idx, owner, rowsQ, base, Q, T, off, r, active, reps=None, G=None, gbase=None, coup=0.0):
    """Metropolis update of every node of one independent set, in all replicas at once.
    off: color offsets in 1..Q-1, r: uniforms, both shaped (R, n_class). Updates q and Aq in place.
    RSA: reps = replicas updated in this call (one per group, so the y replicas of a group are
    visited one after another), G = per-group color counts, coup = gamma/y. A move o -> n adds
    -coup*(#other replicas of the group with n - #other replicas with o) to the energy change."""
    sub = reps is not None
    old = q[reps][:, idx] if sub else q[:, idx]                  # (R, n_class)
    new = (old + off) % Q
    base_s = base[reps] if sub else base
    bi = base_s + idx * Q
    dE = Aq[bi + new] - Aq[bi + old]
    if G is not None:
        gb = (gbase[reps] if sub else gbase) + idx * Q
        dE = dE - coup * (G[gb + new] - G[gb + old] + 1)         # +1: the replica itself holds color o
    if isinstance(T, torch.Tensor):                              # CUDA-graph mode: T lives on the GPU, T >= 1e-9
        acc = (dE <= 0) | (r < torch.exp(-dE / T))
    elif T > 0:
        acc = (dE <= 0) | (r < torch.exp(-dE / T))
    else:
        acc = dE <= 0
    acc &= active[reps] if sub else active
    upd = torch.where(acc, new, old)
    if sub:
        q[reps[:, None], idx[None, :]] = upd
    else:
        q[:, idx] = upd
    # neighbours of each accepted node: -1 at old color, +1 at new color
    w = acc[:, owner].float().reshape(-1)
    bo = base_s + rowsQ
    Aq.index_add_(0, (bo + old[:, owner]).reshape(-1), -w)
    Aq.index_add_(0, (bo + new[:, owner]).reshape(-1), w)
    if G is not None:
        wg = acc.float().reshape(-1)
        G.index_add_(0, (gb + old).reshape(-1), -wg)
        G.index_add_(0, (gb + new).reshape(-1), wg)


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

    # 1b) RSA: staged parallel step (replica k of every group, one class at a time) vs the same
    #     Metropolis moves done one node at a time with the explicit replica-coupling energy
    y, groups, gamma = 3, 2, 1.0
    R2 = y * groups; base2 = (torch.arange(R2, device=device) * NQ)[:, None]
    gbase = ((torch.arange(R2, device=device) // y) * NQ)[:, None]
    active2 = torch.ones(R2, 1, dtype=torch.bool, device=device)
    reps_k = [torch.arange(groups, device=device) * y + k for k in range(y)]
    for T in (2.0, 0.5, 0.0):
        q = torch.randint(0, Q, (R2, N), device=device, generator=gen)
        Aq = fresh_Aq(q, src_t, nbr_t, base2, Q, R2, NQ); G = fresh_G(q, y, N, Q)
        qs = q.cpu().numpy().copy()
        for idx, owner, rowsQ in split:
            idx_n = idx.cpu().numpy()
            for k in range(y):
                shape = (groups, idx.numel())
                off = torch.randint(1, Q, shape, device=device, generator=gen)
                r = torch.rand(shape, device=device, generator=gen)
                class_step(q, Aq, idx, owner, rowsQ, base2, Q, T, off, r, active2, reps_k[k], G, gbase, gamma / y)
                off_n, r_n = off.cpu().numpy(), r.cpu().numpy()
                for g in range(groups):
                    rep = g * y + k
                    for j in rng.permutation(idx_n.size):
                        v = idx_n[j]; o = qs[rep, v]; n = (o + off_n[g, j]) % Q
                        nb = qs[rep, nbr[ptr[v]:ptr[v + 1]]]
                        others = np.array([qs[g * y + m, v] for m in range(y) if m != k])
                        dE = np.float32((nb == n).sum() - (nb == o).sum()) \
                            - np.float32(gamma / y) * np.float32((others == n).sum() - (others == o).sum())
                        if dE <= 0 or (T > 0 and r_n[g, j] < np.exp(np.float32(-dE) / np.float32(T))):
                            qs[rep, v] = n
        assert np.array_equal(q.cpu().numpy(), qs), f"RSA parallel != sequential at T={T}"
        assert torch.equal(Aq, fresh_Aq(q, src_t, nbr_t, base2, Q, R2, NQ)), f"RSA Aq drifted at T={T}"
        assert torch.equal(G, fresh_G(q, y, N, Q)), f"RSA G drifted at T={T}"
    print("  1b) RSA staged step matches sequential updates with replica coupling: OK")

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


def anneal(N, Q, e_t, ptr, nbr, splits, R, temps, check_every, device, gen, pyrng, verbose=True, y=1, gamma=1.0, use_graph=False):
    """R replicas in total. y > 1: replicated SA, R/y independent groups of y coupled replicas
    (energy = sum of conflicts per replica - gamma/(2y) * sum over ordered replica pairs a != b of
    #nodes with equal color). y = 1 is plain SA."""
    assert R % y == 0
    NQ = N * Q
    base = (torch.arange(R, device=device) * NQ)[:, None]          # replica offset into flat Aq
    nbr_t = torch.as_tensor(nbr, device=device)
    src_t = torch.repeat_interleave(torch.arange(N, device=device),
                                    torch.as_tensor(np.diff(ptr), device=device))
    e0, e1 = e_t[:, 0], e_t[:, 1]

    q = torch.randint(0, Q, (R, N), device=device, generator=gen)
    Aq = fresh_Aq(q, src_t, nbr_t, base, Q, R, NQ)
    conf = lambda: (q[:, e0] == q[:, e1]).sum(1)
    if y > 1:
        groups = R // y
        gbase = ((torch.arange(R, device=device) // y) * NQ)[:, None]
        G = fresh_G(q, y, N, Q)
        reps_k = [torch.arange(groups, device=device) * y + k for k in range(y)]

    best = conf(); best_q = q.clone()
    solved_at = torch.full((R,), -1, device=device)
    active = torch.ones(R, 1, dtype=torch.bool, device=device)     # solved replicas are frozen

    def run_split(split, T):
        for idx, owner, rowsQ in split:
            if y == 1:
                shape = (R, idx.numel())
                off = torch.randint(1, Q, shape, device=device, generator=gen)
                r = torch.rand(shape, device=device, generator=gen)
                class_step(q, Aq, idx, owner, rowsQ, base, Q, T, off, r, active)
            else:
                for k in range(y):                                   # replicas of a group one after another
                    shape = (groups, idx.numel())
                    off = torch.randint(1, Q, shape, device=device, generator=gen)
                    r = torch.rand(shape, device=device, generator=gen)
                    class_step(q, Aq, idx, owner, rowsQ, base, Q, T, off, r, active,
                               reps_k[k], G, gbase, gamma / y)

    graphs = None
    if use_graph:
        # capture one whole sweep per split; replay needs static tensors (q, Aq, G, active, T_t) updated in place
        assert device.type == "cuda" and gen is None, "CUDA graphs need a GPU and the default generator"
        tg = time.perf_counter()
        T_t = torch.full((), max(temps[0], 1e-9), device=device)
        saved = [q.clone(), Aq.clone()] + ([G.clone()] if y > 1 else [])
        side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):                                # warmup before capture
            for sp in splits:
                run_split(sp, T_t)
        torch.cuda.current_stream().wait_stream(side)
        graphs = []
        for sp in splits:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                run_split(sp, T_t)
            graphs.append(g)
        for dst, src in zip([q, Aq] + ([G] if y > 1 else []), saved):
            dst.copy_(src)                                           # undo the warmup sweeps
        torch.cuda.synchronize()
        if verbose:
            print(f"captured {len(graphs)} CUDA graphs in {time.perf_counter()-tg:.1f}s")
    t0 = time.perf_counter()

    for s, T in enumerate(temps):
        si = pyrng.randrange(len(splits))
        if graphs is not None:
            T_t.fill_(max(T, 1e-9)); graphs[si].replay()
        else:
            run_split(splits[si], T)

        if (s + 1) % check_every == 0 or s == len(temps) - 1:
            c = conf()
            better = c < best
            best = torch.where(better, c, best)
            best_q[better] = q[better]
            newly = (c == 0) & (solved_at < 0)
            solved_at[newly] = s + 1
            active.copy_((solved_at < 0)[:, None])
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
    if y > 1:
        assert torch.equal(G, fresh_G(q, y, N, Q)), "G drifted"
    return best_q, best, solved_at, s + 1, elapsed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("graph", nargs="?")
    p.add_argument("--selftest", action="store_true", help="check correctness on this device and exit")
    p.add_argument("--q", type=int, default=5)
    p.add_argument("--seeds", type=int, default=10, help="independent runs (groups) in one batch")
    p.add_argument("--y", type=int, default=1, help="replicated SA: replicas per run (1 = plain SA)")
    p.add_argument("--gamma", type=float, default=1.0, help="replicated SA: coupling strength")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--schedule", choices=["linear", "geom"], default="linear")
    p.add_argument("--T0", type=float, default=1.4)
    p.add_argument("--T1", type=float, default=0.0)
    p.add_argument("--dT", type=float, default=5e-5, help="linear step (ignored if --sweeps given)")
    p.add_argument("--sweeps", type=int, default=None)
    p.add_argument("--splits", default=None, help="file of class labels, one line of N ints per split")
    p.add_argument("--n-splits", type=int, default=10, help="greedy splits if --splits not given")
    p.add_argument("--check-every", type=int, default=100)
    p.add_argument("--cuda-graph", action="store_true", help="capture each sweep as a CUDA graph (GPU only)")
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
    if a.cuda_graph:
        torch.manual_seed(a.seed); gen = None          # default generator is the graph-safe one
    else:
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
          f"y={a.y} gamma={a.gamma} over {a.sweeps} sweeps  (setup {time.perf_counter()-t:.1f}s)")

    R = a.seeds * a.y
    best_q, best, solved_at, used, el = anneal(N, a.q, e_t, ptr, nbr, splits, R, temps,
                                               a.check_every, device, gen, pyrng, y=a.y, gamma=a.gamma, use_graph=a.cuda_graph)
    best, solved_at = best.cpu().numpy(), solved_at.cpu().numpy()
    for r in range(a.seeds):
        sl = slice(r * a.y, (r + 1) * a.y)
        n_ok = int((solved_at[sl] >= 0).sum())
        print(f"seed {r}: conflicts={best[sl].tolist() if a.y > 1 else best[r]}" +
              (f"  solved {n_ok}/{a.y} replicas" if a.y > 1 else "") +
              (f"  solved at sweep {solved_at[sl].max()}" if n_ok == a.y else ""))
    ok = solved_at >= 0
    print(f"solved {ok.reshape(a.seeds, a.y).all(1).sum()}/{a.seeds} runs  ran {used} sweeps in {el:.1f}s ({1e3*el/used:.3f} ms/sweep for all replicas)")
    if ok.any():
        qs = best_q[torch.as_tensor(ok, device=device)].cpu().numpy()
        assert not (qs[:, e[:, 0]] == qs[:, e[:, 1]]).any()
        np.savetxt(a.out, qs.T, fmt="%d")
        print(f"saved {ok.sum()} valid colorings to {a.out} (one column per solved seed)")


if __name__ == "__main__":
    main()
