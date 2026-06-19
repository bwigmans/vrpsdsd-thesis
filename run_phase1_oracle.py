"""
Phase 1 on oracle Phase 0 topology.
Loads oracle solutions, removes pre-built splits (reinserting customers as
full nodes), then runs the same phase1 analysis as run_phase1.py.
Lets us compare: standard topology + best post-hoc split vs oracle topology + best post-hoc split.
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
from run_phase1 import (
    load_solution, build_split_routes, search_phase1,
    phase1_path,
)

INSTANCE_FILE    = "data/RC101.txt"
VEHICLE_CAPACITY = 70.0
DEMAND_SCALE     = (0.51, 0.70)
NUM_CUSTOMERS    = 25
BANK_PATH        = "data/samples/sample_bank.npz"
SEEDS            = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

INSTANCE_NAME = os.path.splitext(os.path.basename(INSTANCE_FILE))[0]
ORACLE_IN_DIR  = f"solutions/{INSTANCE_NAME}/oracle"
OUT_DIR        = f"solutions/{INSTANCE_NAME}/phase1_oracle"



def min_cost_insertion_pos(route, node, calc):
    """Return best insertion position by cost only — no feasibility check. Always returns a pos."""
    base = calc.total_expected_cost(route)
    best_pos, best_inc = 1, float("inf")
    for pos in range(1, len(route.nodes)):
        tmp = route.nodes[:pos] + [node] + route.nodes[pos:]
        inc = calc.total_expected_cost(Route(tmp, route.instance)) - base
        if inc < best_inc:
            best_inc, best_pos = inc, pos
    return best_pos


def unsplit_solution(data, inst, node_by_id, depot, calc):
    """
    Remove pre-built splits from oracle solution and reinsert each split customer
    as a full node. Strategy:
      1. Try inserting into the routes that originally contained the split nodes
         (prefer the route with the split node closer to depot).
      2. If neither is feasible, greedy insert into any route.
    Returns a clean Solution with no splits.
    """
    if "split_node_map" not in data:
        raise RuntimeError("Oracle solution JSON missing 'split_node_map' — delete cached JSONs and rerun Phase 0")
    split_node_map = {int(k): int(v) for k, v in data["split_node_map"].items()}

    # Find which routes contain which customers' splits, and which is closer to depot
    cust_routes = {}  # cid -> (ri_closer, ri_farther)
    for cid in set(split_node_map.values()):
        ri_list = [ri for ri, cids in enumerate(data["routes"])
                   if any(c in split_node_map and split_node_map[c] == cid for c in cids)]
        if len(ri_list) != 2:
            raise RuntimeError(
                f"Split customer c{cid} found in {len(ri_list)} routes (expected 2) — "
                "oracle solution data is corrupted"
            )
        # r_closer = the one where split node appears earlier (lower position)
        def pos_in_route(ri):
            for pos, c in enumerate(data["routes"][ri]):
                if c in split_node_map and split_node_map[c] == cid:
                    return pos
            return 999
        ri_list.sort(key=pos_in_route)
        cust_routes[cid] = (ri_list[0], ri_list[1])

    # Strip all split nodes, reinsert each split customer as a full node at min-cost position.
    # No feasibility check — post-ALNS solutions are valid, Phase 1/2 handle recourse.
    routes_nodes = []
    for cids in data["routes"]:
        nodes = [depot] + [node_by_id[c] for c in cids if c > 0] + [depot]
        routes_nodes.append(nodes)

    for cid, (ri_pref, ri_alt) in cust_routes.items():
        full_node = Node(cid, node_by_id[cid].x, node_by_id[cid].y,
                         node_by_id[cid].mean_demand,
                         demand_distribution=node_by_id[cid].demand_distribution)
        pos = min_cost_insertion_pos(Route(routes_nodes[ri_pref], inst), full_node, calc)
        routes_nodes[ri_pref] = routes_nodes[ri_pref][:pos] + [full_node] + routes_nodes[ri_pref][pos:]

    routes = [Route(nodes, inst) for nodes in routes_nodes]
    return Solution(routes)


def phase1_oracle_path(seed):
    return os.path.join(OUT_DIR, f"phase1_oracle_{seed}.json")


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
    stage1 = bank.post_stage1

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Phase 1 (oracle topology) — {INSTANCE_NAME} 25c, post_stage1 N=1000\n")

    for seed in SEEDS:
        oracle_path = os.path.join(ORACLE_IN_DIR, f"base_solution_{seed}.json")
        if not os.path.exists(oracle_path):
            print(f"seed={seed}  [oracle solution not found, skipping]")
            continue

        out_path = phase1_oracle_path(seed)
        if os.path.exists(out_path):
            with open(out_path) as f:
                saved = json.load(f)
            print(f"seed={seed}  [loaded]  "
                  + "  ".join(f"{k}=c{saved[k]['customer']}"
                               for k in ["exact", "oracle"] if saved.get(k)))
            continue

        with open(oracle_path) as f:
            data = json.load(f)

        sol = unsplit_solution(data, inst, node_by_id, depot, calc)

        from run_phase1 import sample_base_cost
        base_cost = sample_base_cost(sol, stage1)

        print(f"seed={seed}  base={base_cost:.3f}  routes={len(sol.routes)}  searching...", flush=True)
        t0 = time.perf_counter()
        calc.invalidate_cache()

        result, table = search_phase1(sol, calc, stage1)
        elapsed = time.perf_counter() - t0

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
            json.dump({"seed": seed, "base_cost": base_cost,
                       "exact": result["exact"], "oracle": result["oracle"],
                       "table": table}, f, indent=2)


if __name__ == "__main__":
    main()
