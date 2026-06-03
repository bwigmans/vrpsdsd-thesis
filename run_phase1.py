"""
Phase 1: Find best split candidate per seed using three objectives.
  - exact:    minimize exact cost delta (ExactCostCalculator)
  - mean:     minimize mean sample delta (PairedVehicleRecourse, post_stage1)
  - cvar:     minimize CVaR(20%) sample delta (PairedVehicleRecourse, post_stage1)

Loads Phase 0 solutions from solutions/{INSTANCE}/sampling/.
Saves best candidate per method to solutions/{INSTANCE}/phase1/.
"""
import sys
import json
import os
import time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from io_thesis.instance_reader import read_solomon_instance
from core.instance import Node
from core.route import Route
from core.solution import Solution
from core.recourse import PairedVehicleRecourse
from cost.calculator import ExactCostCalculator
from cost.sample_bank import DemandSampleBank

# ── config ────────────────────────────────────────────────────────────────────
INSTANCE_FILE   = "data/RC101.txt"
VEHICLE_CAPACITY = 70.0
DEMAND_SCALE    = (0.51, 0.70)
NUM_CUSTOMERS   = 25
BANK_PATH       = "data/samples/sample_bank.npz"
ALPHA_GRID      = [round(i * 0.1, 1) for i in range(1, 10)]
CVAR_LEVEL      = 0.2
SEEDS           = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

INSTANCE_NAME = os.path.splitext(os.path.basename(INSTANCE_FILE))[0]
IN_DIR  = f"solutions/{INSTANCE_NAME}/sampling"
OUT_DIR = f"solutions/{INSTANCE_NAME}/phase1"


def phase1_path(seed):
    return os.path.join(OUT_DIR, f"phase1_{seed}.json")


def load_solution(seed, node_by_id, depot, inst):
    path = os.path.join(IN_DIR, f"base_solution_{seed}.json")
    with open(path) as f:
        data = json.load(f)
    routes = []
    for cids in data["routes"]:
        nodes = [depot] + [node_by_id[c] for c in cids] + [depot]
        routes.append(Route(nodes, inst))
    return Solution(routes), data["exact_cost"]


def best_insertion_pos(tr2, partner, calc):
    base = calc.total_expected_cost(tr2)
    best_pos, best_inc = None, float("inf")
    for pos in range(1, len(tr2.nodes)):
        tmp = tr2.nodes.copy()
        tmp.insert(pos, partner)
        t = Route(tmp, tr2.instance)
        if not t.is_feasible():
            continue
        inc = calc.total_expected_cost(t) - base
        if inc < best_inc:
            best_inc, best_pos = inc, pos
    return best_pos


def build_split_routes(sol, cid, r1_idx, r2_idx, alpha, calc):
    trial = sol.copy()
    tr1, tr2 = trial.routes[r1_idx], trial.routes[r2_idx]
    node = next((n for n in tr1.nodes if not n.is_depot and n.id == cid), None)
    if node is None:
        return None, None, None
    node.is_split = True
    node.alpha = alpha
    node.original_id = cid
    if not tr1.is_feasible():
        return None, None, None
    partner = Node(cid, node.x, node.y, node.mean_demand,
                   is_split=True, alpha=round(1.0 - alpha, 10))
    partner.original_id = cid
    pos = best_insertion_pos(tr2, partner, calc)
    if pos is None:
        return None, None, None
    tr2.nodes.insert(pos, partner)
    return trial, tr1, tr2


def exact_delta(sol, tr1, tr2, r1_idx, r2_idx, calc):
    r1_orig = sol.routes[r1_idx]
    r2_orig = sol.routes[r2_idx]
    return (calc.total_expected_cost(tr1) + calc.total_expected_cost(tr2)
            - calc.total_expected_cost(r1_orig) - calc.total_expected_cost(r2_orig))


def _sim_route(rec, route, sample_dict, N, alpha_override=None, orig_id=None):
    custs = [n for n in route.nodes if not n.is_depot]
    ids = [getattr(n, "original_id", n.id) for n in custs]
    costs = np.zeros(N)
    for i in range(N):
        demands = []
        for k, nd in enumerate(custs):
            d = float(sample_dict[ids[k]][i])
            if nd.is_split:
                if alpha_override is not None and getattr(nd, "original_id", nd.id) == orig_id:
                    d *= alpha_override
                else:
                    d *= nd.alpha
            demands.append(d)
        costs[i] = rec.compute_cost(route, demands)
    return costs + route.travel_cost()


def sample_delta_pair(sol, tr1, tr2, r1_idx, r2_idx, samples):
    rec = PairedVehicleRecourse()
    N = len(next(iter(samples.values())))
    orig_r1_costs = _sim_route(rec, sol.routes[r1_idx], samples, N)
    orig_r2_costs = _sim_route(rec, sol.routes[r2_idx], samples, N)
    new_r1_costs  = _sim_route(rec, tr1, samples, N)
    new_r2_costs  = _sim_route(rec, tr2, samples, N)
    delta_per_sample = (new_r1_costs + new_r2_costs) - (orig_r1_costs + orig_r2_costs)
    return delta_per_sample


def oracle_delta_pair(sol, tr1, tr2, r1_idx, r2_idx, samples, original_id):
    """Hindsight oracle: per-sample minimum over alpha in {0, 0.1..0.9, 1}.
    alpha=0: r1 visits node, delivers 0; r2 delivers full (partner still visited).
    alpha=1: r1 delivers full; r2 skips partner entirely (original r2 route, no detour).
    """
    rec = PairedVehicleRecourse()
    N = len(next(iter(samples.values())))
    orig_r1_costs = _sim_route(rec, sol.routes[r1_idx], samples, N)
    orig_r2_costs = _sim_route(rec, sol.routes[r2_idx], samples, N)
    best_per_sample = np.full(N, np.inf)

    # alpha=0: r1 delivers nothing (visit kept), r2 delivers full
    r1_zero = _sim_route(rec, tr1, samples, N, alpha_override=0.0, orig_id=original_id)
    r2_full = _sim_route(rec, tr2, samples, N, alpha_override=1.0, orig_id=original_id)
    best_per_sample = np.minimum(best_per_sample, r1_zero + r2_full)

    # alpha in (0, 1): coordinated split
    for alpha in ALPHA_GRID:
        r1_costs = _sim_route(rec, tr1, samples, N, alpha_override=alpha, orig_id=original_id)
        r2_costs = _sim_route(rec, tr2, samples, N, alpha_override=round(1.0 - alpha, 10), orig_id=original_id)
        best_per_sample = np.minimum(best_per_sample, r1_costs + r2_costs)

    # alpha=1: r1 delivers full, r2 removes partner (original r2, no detour)
    r1_full = _sim_route(rec, tr1, samples, N, alpha_override=1.0, orig_id=original_id)
    best_per_sample = np.minimum(best_per_sample, r1_full + orig_r2_costs)

    return float((best_per_sample - (orig_r1_costs + orig_r2_costs)).mean())


def sample_base_cost(sol, samples):
    rec = PairedVehicleRecourse()
    N = len(next(iter(samples.values())))
    total = np.zeros(N)
    for route in sol.routes:
        total += _sim_route(rec, route, samples, N)
    return float(total.mean())


def cvar(arr, level=CVAR_LEVEL):
    cutoff = int(len(arr) * (1 - level))
    return float(np.mean(np.sort(arr)[cutoff:]))


def search_phase1(sol, calc, samples):
    """
    Exhaustive search over (node, r1, r2). For each triple:
      - best_fixed: min mean delta across ALPHA_GRID (best any fixed alpha can do)
      - oracle: E[min_alpha cost(alpha)] — per-sample best, then averaged
      - premium: oracle - best_fixed (gain only adaptive policy can capture)
    Filter: oracle < 0 AND at least one of (exact, mean) < 0 across alphas.
    """
    best = {
        "exact":  (float("inf"), None),
        "oracle": (float("inf"), None),
    }
    table = []

    for r1_idx, r1 in enumerate(sol.routes):
        customers = [n for n in r1.nodes if not n.is_depot and not n.is_split]
        for node in customers:
            for r2_idx in range(len(sol.routes)):
                if r2_idx == r1_idx:
                    continue
                r2 = sol.routes[r2_idx]
                if not any(n for n in r2.nodes if not n.is_depot):
                    continue

                best_exact = float("inf")
                best_fixed = float("inf")  # min mean across alphas
                best_tr_pair = None        # route pair for oracle (uses best-mean alpha)

                for alpha in ALPHA_GRID:
                    trial, tr1, tr2 = build_split_routes(
                        sol, node.id, r1_idx, r2_idx, alpha, calc
                    )
                    if tr1 is None:
                        continue

                    e = exact_delta(sol, tr1, tr2, r1_idx, r2_idx, calc)
                    deltas = sample_delta_pair(sol, tr1, tr2, r1_idx, r2_idx, samples)
                    m = float(deltas.mean())

                    if e < best_exact:
                        best_exact = e
                    if m < best_fixed:
                        best_fixed = m
                        best_tr_pair = (tr1, tr2)

                if best_tr_pair is None:
                    continue

                tr1, tr2 = best_tr_pair
                o = oracle_delta_pair(sol, tr1, tr2, r1_idx, r2_idx, samples, node.id)
                premium = o - best_fixed

                if best_exact < best["exact"][0]:
                    best["exact"] = (best_exact, dict(customer=node.id, r1=r1_idx,
                                                       r2=r2_idx, score=round(best_exact, 6)))
                if o < best["oracle"][0]:
                    best["oracle"] = (o, dict(customer=node.id, r1=r1_idx,
                                              r2=r2_idx, score=round(o, 6)))

                # Filter: oracle negative AND at least one of exact/best_fixed negative
                if o < -0.001 and (best_exact < -0.001 or best_fixed < -0.001):
                    table.append(dict(
                        customer=node.id, r1=r1_idx, r2=r2_idx,
                        oracle=round(o, 4),
                        best_fixed=round(best_fixed, 4),
                        premium=round(premium, 4),
                        best_exact=round(best_exact, 4),
                    ))

    table.sort(key=lambda x: x["oracle"])
    return {k: v[1] for k, v in best.items()}, table


def main():
    inst = read_solomon_instance(
        INSTANCE_FILE, vehicle_capacity=VEHICLE_CAPACITY,
        demand_scale=DEMAND_SCALE, num_customers=NUM_CUSTOMERS,
    )
    node_by_id = {n.id: n for n in inst.nodes}
    depot = node_by_id[0]
    calc = ExactCostCalculator(PairedVehicleRecourse(), cache=True)

    bank = DemandSampleBank.load_or_create(
        BANK_PATH, inst, op_size=250, eval_size=500, seed=42,
    )
    stage1 = bank.post_stage1  # N=1000 for Phase 1 selection

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Phase 1 — {INSTANCE_NAME} 25c, post_stage1 N=1000\n")

    for seed in SEEDS:
        out_path = phase1_path(seed)
        if os.path.exists(out_path):
            with open(out_path) as f:
                saved = json.load(f)
            print(f"seed={seed}  [loaded]  "
                  + "  ".join(f"{k}=c{saved[k]['customer']}"
                               for k in ["exact", "oracle"] if saved.get(k)))
            continue

        sol, _ = load_solution(seed, node_by_id, depot, inst)
        base_cost = sample_base_cost(sol, stage1)
        print(f"seed={seed}  base={base_cost:.3f}  searching...", flush=True)
        t0 = time.perf_counter()
        calc.invalidate_cache()

        result, table = search_phase1(sol, calc, stage1)

        elapsed = time.perf_counter() - t0

        # Print table — one row per (customer, r1, r2), sorted by oracle
        print(f"  {'cust':>5} {'r1':>3} {'r2':>3}  {'oracle':>8}  {'best_fixed':>10}  {'premium':>8}  {'best_exact':>10}")
        print(f"  {'-'*60}")
        for row in table[:15]:
            marker = ""
            if result["exact"]  and row["customer"] == result["exact"]["customer"]  and row["r1"] == result["exact"]["r1"]:  marker += "E"
            if result["oracle"] and row["customer"] == result["oracle"]["customer"] and row["r1"] == result["oracle"]["r1"]: marker += "O"
            cust_str = f"c{row['customer']}"
            print(f"  {cust_str:>5} {row['r1']:>3} {row['r2']:>3}"
                  f"  {row['oracle']:>+8.4f}  {row['best_fixed']:>+10.4f}  {row['premium']:>+8.4f}  {row['best_exact']:>+10.4f}"
                  f"  {marker}")
        if len(table) > 15:
            print(f"  ... ({len(table)} total)")
        print(f"  ({elapsed:.1f}s)\n", flush=True)

        with open(out_path, "w") as f:
            json.dump({"seed": seed, "sample_base_cost": base_cost,
                       "exact": result["exact"], "oracle": result["oracle"],
                       "table": table}, f, indent=2)


if __name__ == "__main__":
    main()
