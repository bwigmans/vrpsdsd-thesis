"""
Phase 0: Run ALNS (Both+EC, 1000 iter) for 10 seeds, save base solutions to JSON.
On subsequent runs, existing seeds are skipped.
"""
import sys
import json
import os
import time

sys.stdout.reconfigure(encoding="utf-8")

from algorithms.alns import ALNSSolver
from io_thesis.instance_reader import read_solomon_instance
from core.recourse import PairedVehicleRecourse
from cost.calculator import ExactCostCalculator
from utils import Configuration

INSTANCE_FILE = "data/RC101.txt"
VEHICLE_CAPACITY = 70.0
DEMAND_SCALE = (0.51, 0.70)
NUM_CUSTOMERS = 25
DEMAND_DIST = "poisson"  # Node default — scipy.stats.poisson
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
ALNS_ITERATIONS = 1000
COST_METHOD = "sampling"   # "exact" or "sampling"
OPERATOR_NUM_SAMPLES = 250
EVAL_NUM_SAMPLES = 500
BANK_PATH = "data/samples/sample_bank.npz"

INSTANCE_NAME = os.path.splitext(os.path.basename(INSTANCE_FILE))[0]
OUT_DIR = f"solutions/{INSTANCE_NAME}/{COST_METHOD}"


def make_cfg(seed):
    cfg = Configuration(
        vehicle_capacity=VEHICLE_CAPACITY,
        alns_iterations=ALNS_ITERATIONS,
        alns_segment_length=50,
        seed=seed,
        verbose=True,
        alpha_policy='lei',
        alpha_reoptimize=False,
        use_ec_operators=True,
        cost_method=COST_METHOD,
    )
    if COST_METHOD == "sampling":
        cfg.operator_num_samples = OPERATOR_NUM_SAMPLES
        cfg.evaluation_num_samples = EVAL_NUM_SAMPLES
        cfg.sample_bank_path = BANK_PATH
    return cfg


def solution_path(seed):
    return os.path.join(OUT_DIR, f"base_solution_{seed}.json")


def save_solution(seed, sol, exact_cost):
    routes = []
    for r in sol.routes:
        cids = [n.id for n in r.nodes if not n.is_depot]
        routes.append(cids)
    data = {"seed": seed, "exact_cost": exact_cost, "routes": routes,
            "cost_method": COST_METHOD, "demand_dist": DEMAND_DIST}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(solution_path(seed), "w") as f:
        json.dump(data, f, indent=2)


def main():
    inst = read_solomon_instance(
        INSTANCE_FILE, vehicle_capacity=VEHICLE_CAPACITY,
        demand_scale=DEMAND_SCALE, num_customers=NUM_CUSTOMERS,
    )
    calc = ExactCostCalculator(PairedVehicleRecourse())

    print(f"Phase 0 — RC101 25c, {ALNS_ITERATIONS} iter, Both+EC\n")

    for seed in SEEDS:
        path = solution_path(seed)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            print(f"seed={seed}  [loaded]  exact={data['exact_cost']:.3f}  routes={len(data['routes'])}")
            continue

        t0 = time.perf_counter()
        print(f"seed={seed}  solving...", flush=True)
        solver = ALNSSolver(inst, make_cfg(seed))
        sol = solver.solve()
        elapsed = time.perf_counter() - t0

        exact_cost = sol.get_total_cost(calc)
        save_solution(seed, sol, exact_cost)
        print(f"seed={seed}  exact={exact_cost:.3f}  routes={len(sol.routes)}  t={elapsed:.1f}s  [saved]")


if __name__ == "__main__":
    main()
