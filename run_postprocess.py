"""
Post-processing split search on Both+EC solutions (10 seeds, dynamic ALNS).
Phase 1: exact delta for candidate selection (grid alpha)
Phase 2: oracle / lei / equalize_slack / marginal_cost for delivery evaluation
"""
import sys
import time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from algorithms.alpha_policies import make_recourse_alpha_policy
from algorithms.alns import ALNSSolver
from io_thesis.instance_reader import read_solomon_instance
from core.instance import Node
from core.route import Route
from core.solution import Solution
from core.recourse import PairedVehicleRecourse, AdaptivePairedVehicleRecourse
from cost.calculator import ExactCostCalculator
from cost.sample_bank import DemandSampleBank
from utils import Configuration

# ── config ────────────────────────────────────────────────────────────────────
INSTANCE_FILE = "data/RC101.txt"
VEHICLE_CAPACITY = 70.0
DEMAND_SCALE = (0.51, 0.70)
NUM_CUSTOMERS = 25
BANK_PATH = "data/samples/sample_bank.npz"
ALPHA_GRID = [round(i * 0.1, 1) for i in range(1, 10)]
P2_METHODS = ["oracle"]
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
ALNS_ITERATIONS = 1000


# ── helpers ───────────────────────────────────────────────────────────────────

def make_split_routes(cust: Node, r1: Route, r2: Route, alpha: float):
    calc = ExactCostCalculator(PairedVehicleRecourse())
    new_r1_nodes = []
    for n in r1.nodes:
        if not n.is_depot and n.id == cust.id:
            split_n = Node(n.id, n.x, n.y, n.mean_demand, is_split=True,
                           alpha=alpha, demand_distribution=n.demand_distribution)
            split_n.original_id = cust.id
            new_r1_nodes.append(split_n)
        else:
            new_r1_nodes.append(n)
    new_r1 = Route(new_r1_nodes, r1.instance)
    if not new_r1.is_feasible():
        return None, None

    partner = Node(-999, cust.x, cust.y, cust.mean_demand, is_split=True,
                   alpha=round(1.0 - alpha, 10),
                   demand_distribution=cust.demand_distribution)
    partner.original_id = cust.id

    base_r2 = calc.total_expected_cost(r2)
    best_pos, best_inc = -1, float("inf")
    for pos in range(1, len(r2.nodes)):
        tmp = Route(r2.nodes[:pos] + [partner] + r2.nodes[pos:], r2.instance)
        if not tmp.is_feasible():
            continue
        inc = calc.total_expected_cost(tmp) - base_r2
        if inc < best_inc:
            best_inc, best_pos = inc, pos

    if best_pos == -1:
        return None, None

    new_r2 = Route(r2.nodes[:best_pos] + [partner] + r2.nodes[best_pos:], r2.instance)
    return new_r1, new_r2


def exact_delta(r1, r2, new_r1, new_r2) -> float:
    calc = ExactCostCalculator(PairedVehicleRecourse())
    return float(calc.total_expected_cost(new_r1) + calc.total_expected_cost(new_r2)
                 - calc.total_expected_cost(r1) - calc.total_expected_cost(r2))


def sample_delta(r1, r2, new_r1, new_r2, samples, p2_method) -> float:
    if p2_method == "oracle":
        return exact_delta(r1, r2, new_r1, new_r2)

    base_rec = PairedVehicleRecourse()
    if p2_method == "lei":
        split_rec = base_rec
        scale_splits = True
    else:
        split_rec = AdaptivePairedVehicleRecourse(make_recourse_alpha_policy(p2_method))
        scale_splits = False

    n = len(next(iter(samples.values())))

    def eval_route(route, paired, rec, scale):
        costs = []
        custs = [nd for nd in route.nodes if not nd.is_depot]
        for i in range(n):
            demands = []
            for nd in custs:
                cid = getattr(nd, "original_id", nd.id)
                d = float(samples[cid][i])
                if nd.is_split and scale:
                    d *= nd.alpha
                demands.append(d)
            costs.append(rec.compute_cost(route, demands, paired_route=paired))
        return np.array(costs)

    orig = eval_route(r1, None, base_rec, True) + eval_route(r2, None, base_rec, True)
    new  = (eval_route(new_r1, new_r2, split_rec, scale_splits) +
            eval_route(new_r2, new_r1, split_rec, scale_splits))
    travel_delta = (new_r1.travel_cost() + new_r2.travel_cost()
                    - r1.travel_cost() - r2.travel_cost())
    return float((new - orig).mean()) + travel_delta


# ── Phase 1 search ────────────────────────────────────────────────────────────

def phase1_search(base_sol: Solution, delta_fn):
    """Grid search over all customers x r2 x alphas using the provided delta_fn."""
    best_score = float("inf")
    best = None

    all_customers = [n for r in base_sol.routes for n in r.nodes
                     if not n.is_depot and not n.is_split]

    for cust in all_customers:
        r1 = next((r for r in base_sol.routes
                   if any(n.id == cust.id for n in r.nodes if not n.is_depot)), None)
        if r1 is None:
            continue
        for r2 in base_sol.routes:
            if r2 is r1:
                continue
            if not any(n for n in r2.nodes if not n.is_depot):
                continue
            for alpha in ALPHA_GRID:
                new_r1, new_r2 = make_split_routes(cust, r1, r2, alpha)
                if new_r1 is None:
                    continue
                score = delta_fn(r1, r2, new_r1, new_r2)
                if score < best_score:
                    best_score = score
                    best = (cust, r1, r2, alpha, score)

    return best


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    inst = read_solomon_instance(
        INSTANCE_FILE, vehicle_capacity=VEHICLE_CAPACITY,
        demand_scale=DEMAND_SCALE, num_customers=NUM_CUSTOMERS,
    )
    calc = ExactCostCalculator(PairedVehicleRecourse())

    all_results = []

    for seed in SEEDS:
        cfg = Configuration(
            vehicle_capacity=VEHICLE_CAPACITY,
            alns_iterations=ALNS_ITERATIONS,
            alns_segment_length=50,
            seed=seed,
            verbose=False,
            alpha_policy='lei',
            alpha_reoptimize=False,
            use_ec_operators=True,
            cost_method='exact',
        )
        t_alns = time.perf_counter()
        solver = ALNSSolver(inst, cfg)
        base_sol = solver.solve()
        print(f"\nseed={seed}  solved in {time.perf_counter()-t_alns:.1f}s", flush=True)
        base_exact = base_sol.get_total_cost(calc)
        print(f"seed={seed}  exact={base_exact:.3f}", flush=True)

        for p2 in P2_METHODS:
            is_exact = (p2 == "oracle")
            base = base_exact if is_exact else None

            if is_exact:
                delta_fn = exact_delta
            else:
                delta_fn = lambda r1, r2, nr1, nr2, _p2=p2: sample_delta(r1, r2, nr1, nr2, stage2, _p2)

            t0 = time.perf_counter()
            best = phase1_search(base_sol, delta_fn)

            if best is None:
                print(f"  [{p2}] no improving split found", flush=True)
                all_results.append({"seed": seed, "p2": p2, "customer": None,
                                    "delta": 0.0, "cost_final": base,
                                    "base": base, "is_exact": is_exact})
                continue

            cust, r1, r2, alpha, score = best
            r1_idx = base_sol.routes.index(r1)
            r2_idx = base_sol.routes.index(r2)
            label = "cost_exact" if is_exact else "cost_s2"
            cost_final = base + score
            print(f"  [{p2}] c{cust.id} r{r1_idx}→r{r2_idx} α={alpha:.1f}"
                  f"  Δ={score:+.4f}  {label}={cost_final:.3f}"
                  f"  ({time.perf_counter()-t0:.1f}s)", flush=True)
            all_results.append({"seed": seed, "p2": p2, "customer": cust.id,
                                 "r1": r1_idx, "r2": r2_idx, "alpha": alpha,
                                 "delta": score, "cost_final": cost_final,
                                 "base": base, "is_exact": is_exact})

    # Summary table
    print(f"\n{'='*80}", flush=True)
    print("SUMMARY — delta per seed per P2 method (oracle=exact, others=sample)", flush=True)
    print(f"{'seed':>6}", end="")
    for p2 in P2_METHODS:
        print(f"  {p2:>18}", end="")
    print(flush=True)
    print("-" * 80, flush=True)

    for seed in SEEDS:
        print(f"{seed:>6}", end="")
        for p2 in P2_METHODS:
            row = next((r for r in all_results if r["seed"] == seed and r["p2"] == p2), None)
            val = f"{row['delta']:+.4f}" if row and row["customer"] else "   none"
            print(f"  {val:>18}", end="")
        print(flush=True)


if __name__ == "__main__":
    main()
