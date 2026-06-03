"""
Phase 0 variant: ALNS with AdaptivePairedVehicleRecourse (equalize_slack) as
the sampling evaluation policy. Tests whether adaptive-aware topology search
finds solutions better suited for real-time alpha decisions.
"""
import sys
import json
import os
import time

sys.stdout.reconfigure(encoding="utf-8")

from algorithms.alns import ALNSSolver
from io_thesis.instance_reader import read_solomon_instance
from core.recourse import PairedVehicleRecourse, AdaptivePairedVehicleRecourse
from algorithms.alpha_policies import make_recourse_alpha_policy
from cost.calculator import ExactCostCalculator
from utils import Configuration

INSTANCE_FILE    = "data/RC101.txt"
VEHICLE_CAPACITY = 70.0
DEMAND_SCALE     = (0.51, 0.70)
NUM_CUSTOMERS    = 25
SEEDS            = [42]
ALNS_ITERATIONS  = 1000
BANK_PATH        = "data/samples/sample_bank.npz"

INSTANCE_NAME = os.path.splitext(os.path.basename(INSTANCE_FILE))[0]
OUT_DIR = f"solutions/{INSTANCE_NAME}/adaptive"


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
        cost_method='sampling',
    )
    cfg.operator_num_samples = 250
    cfg.evaluation_num_samples = 500
    cfg.sample_bank_path = BANK_PATH
    return cfg


def solution_path(seed):
    return os.path.join(OUT_DIR, f"base_solution_{seed}.json")


def main():
    inst = read_solomon_instance(
        INSTANCE_FILE, vehicle_capacity=VEHICLE_CAPACITY,
        demand_scale=DEMAND_SCALE, num_customers=NUM_CUSTOMERS,
    )
    exact_calc = ExactCostCalculator(PairedVehicleRecourse())
    adaptive_rec = AdaptivePairedVehicleRecourse(
        make_recourse_alpha_policy('equalize_slack')
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Phase 0 adaptive — RC101 25c, {ALNS_ITERATIONS} iter, equalize_slack recourse\n")

    for seed in SEEDS:
        path = solution_path(seed)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            print(f"seed={seed}  [loaded]  exact={data['exact_cost']:.3f}")
            continue

        t0 = time.perf_counter()
        print(f"seed={seed}  solving...", flush=True)
        solver = ALNSSolver(inst, make_cfg(seed), recourse_policy=adaptive_rec)
        sol = solver.solve()
        elapsed = time.perf_counter() - t0

        exact_cost = sol.get_total_cost(exact_calc)
        routes = [[n.id for n in r.nodes if not n.is_depot] for r in sol.routes]
        with open(path, "w") as f:
            json.dump({"seed": seed, "exact_cost": exact_cost,
                       "routes": routes, "cost_method": "adaptive_equalize_slack"}, f, indent=2)
        print(f"seed={seed}  exact={exact_cost:.3f}  routes={len(sol.routes)}  t={elapsed:.1f}s  [saved]")


if __name__ == "__main__":
    main()
