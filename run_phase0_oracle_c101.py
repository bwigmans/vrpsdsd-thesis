"""Phase 0 oracle ALNS for C101, seed 42."""
import sys, json, os, time
sys.stdout.reconfigure(encoding="utf-8")

from algorithms.alns import ALNSSolver
from io_thesis.instance_reader import read_solomon_instance
from core.recourse import PairedVehicleRecourse
from cost.calculator import ExactCostCalculator
from utils import Configuration

INSTANCE_FILE    = "data/C101.txt"
VEHICLE_CAPACITY = 70.0
DEMAND_SCALE     = (0.51, 0.70)
NUM_CUSTOMERS    = 25
SEEDS            = [42]
ALNS_ITERATIONS  = 1000
BANK_PATH        = "data/samples/sample_bank_c101.npz"

INSTANCE_NAME = "C101"
OUT_DIR = f"solutions/{INSTANCE_NAME}/oracle"

def make_cfg(seed):
    cfg = Configuration(
        vehicle_capacity=VEHICLE_CAPACITY,
        alns_iterations=ALNS_ITERATIONS,
        alns_segment_length=50,
        seed=seed, verbose=True,
        alpha_policy='lei', alpha_reoptimize=False,
        use_ec_operators=True, cost_method='sampling',
        use_oracle_splits=True,
    )
    cfg.operator_num_samples = 250
    cfg.evaluation_num_samples = 500
    cfg.sample_bank_path = BANK_PATH
    return cfg

def main():
    inst = read_solomon_instance(INSTANCE_FILE, vehicle_capacity=VEHICLE_CAPACITY,
                                  demand_scale=DEMAND_SCALE, num_customers=NUM_CUSTOMERS)
    exact_calc = ExactCostCalculator(PairedVehicleRecourse())
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Phase 0 oracle — C101 25c, {ALNS_ITERATIONS} iter\n")

    for seed in SEEDS:
        path = os.path.join(OUT_DIR, f"base_solution_{seed}.json")
        if os.path.exists(path):
            with open(path) as f: d = json.load(f)
            print(f"seed={seed}  [loaded]  exact={d['exact_cost']:.3f}")
            continue
        t0 = time.perf_counter()
        print(f"seed={seed}  solving...", flush=True)
        solver = ALNSSolver(inst, make_cfg(seed))
        sol = solver.solve()
        elapsed = time.perf_counter() - t0
        exact_cost = sol.get_total_cost(exact_calc)
        routes = [[n.id for n in r.nodes if not n.is_depot] for r in sol.routes]
        split_customers = sorted(set(
            getattr(n,'original_id',n.id)
            for r in sol.routes for n in r.nodes
            if not n.is_depot and n.is_split
        ))
        paired_indices = {}
        for r1, r2 in sol.paired_routes.items():
            i1, i2 = sol.routes.index(r1), sol.routes.index(r2)
            if str(i1) not in paired_indices:
                paired_indices[str(i1)] = i2
        split_node_map = {str(n.id): getattr(n,'original_id',n.id)
                          for r in sol.routes for n in r.nodes if n.is_split}
        with open(path, "w") as f:
            json.dump({"seed": seed, "exact_cost": exact_cost, "routes": routes,
                       "split_customers": split_customers,
                       "paired_indices": paired_indices,
                       "split_node_map": split_node_map}, f, indent=2)
        print(f"seed={seed}  exact={exact_cost:.3f}  routes={len(sol.routes)}"
              f"  splits={split_customers}  t={elapsed:.1f}s  [saved]")

        std_path = f"solutions/{INSTANCE_NAME}/sampling/base_solution_{seed}.json"
        if os.path.exists(std_path):
            with open(std_path) as f: std = json.load(f)
            std_routes = frozenset(frozenset(r) for r in std["routes"])
            ora_routes = frozenset(frozenset(r) for r in routes)
            print(f"  topology vs standard: {'SAME' if std_routes == ora_routes else 'DIFFERENT'}")

if __name__ == "__main__":
    main()
