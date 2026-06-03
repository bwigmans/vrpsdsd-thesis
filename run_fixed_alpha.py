"""
Run ALNS with fixed alpha values (0.1..0.9) on Poisson demand (C101 25c).
Compares topology quality across fixed alpha policies + Lei baseline.
Final evaluation uses large-N true model sampling.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from algorithms.alns import ALNSSolver
from io_thesis.instance_reader import read_solomon_instance
from utils import Configuration
from cost.calculator import ExactCostCalculator
from cost.sampling import SamplingCostCalculator
from cost.sampling_strategy import MonteCarloStrategy
from core.recourse import PairedVehicleRecourse
from test_sampling_poisson import FirstFailureOnlyRecourse

SEED = 42
ITERATIONS = 1000
INSTANCE_FILE = "data/RC101.txt"
VEHICLE_CAPACITY = 70.0
DEMAND_SCALE = (0.51, 0.70)
NUM_CUSTOMERS = 25
DIST = "poisson"  # "poisson" or "nb"
N_EVAL_REF = 1000
N_EVAL_REPS = 1

FIXED_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]
SPLIT_ONLY = True

ALPHA_POLICIES = ["lei", "equalize_slack", "equalize_std_slack", "marginal_cost"]


def make_cfg(alpha_grid=None, alpha_policy="lei"):
    return Configuration(
        vehicle_capacity=VEHICLE_CAPACITY,
        alns_iterations=ITERATIONS,
        alns_segment_length=50,
        seed=SEED,
        verbose=True,
        log_frequency=100,
        alpha_policy=alpha_policy,
        alpha_grid=alpha_grid,
        use_ec_operators=False,
        cost_method="exact",
    )


def run(label, config, instance):
    print(f"  Running: {label}", flush=True)
    t0 = time.perf_counter()
    solver = ALNSSolver(instance, config, recourse_policy=FirstFailureOnlyRecourse())
    if SPLIT_ONLY:
        from algorithms.operators import SplitInsertion
        split_idx = next(i for i, op in enumerate(solver.insertion_operators) if isinstance(op, SplitInsertion))
        solver.insertion_operators = [solver.insertion_operators[split_idx]]
        solver.insertion_weights = [1.0]
        solver.insertion_scores = [0.0]
        solver.insertion_counts = [0]
        solver.insertion_breakdown = [{'new_best': 0, 'improved': 0, 'accepted': 0, 'rejected': 0}]
    solution = solver.solve()
    elapsed = time.perf_counter() - t0

    exact_calc = ExactCostCalculator(PairedVehicleRecourse())
    exact_cost = solution.get_total_cost(exact_calc)
    travel = sum(r.travel_cost() for r in solution.routes)

    # Large-N true model evaluation
    rep_costs = []
    for rep in range(N_EVAL_REPS):
        strategy = MonteCarloStrategy(PairedVehicleRecourse(), num_samples=N_EVAL_REF, seed=SEED + 9000 + rep)
        calc = SamplingCostCalculator(PairedVehicleRecourse(), strategy)
        rep_costs.append(solution.get_total_cost(calc) - travel)
    true_recourse = np.mean(rep_costs)

    route_details = []
    for r in solution.routes:
        cids = [n.id for n in r.nodes if not n.is_depot]
        route_details.append({"customers": cids, "travel": r.travel_cost(),
                               "recourse": exact_calc.compute_recourse_cost(r)})

    # Split stats
    split_op = next((op for op in solver.insertion_operators
                     if type(op).__name__ == "SplitInsertion"), None)
    split_stats = split_op.stats if split_op else {}

    print(f"    done in {elapsed:.1f}s  routes={len(solution.routes)}  "
          f"exact={exact_cost:.2f}  true_total={travel + true_recourse:.2f}", flush=True)

    return {
        "label": label,
        "elapsed": elapsed,
        "routes": len(solution.routes),
        "exact_cost": exact_cost,
        "travel": travel,
        "true_recourse": true_recourse,
        "true_total": travel + true_recourse,
        "route_details": route_details,
        "split_stats": split_stats,
        "solver": solver,
        "solution": solution,
    }


def main():
    instance = read_solomon_instance(
        INSTANCE_FILE, vehicle_capacity=VEHICLE_CAPACITY,
        demand_scale=DEMAND_SCALE, num_customers=NUM_CUSTOMERS,
    )

    print(f"C101 25c Poisson — fixed alpha grid + Lei baseline", flush=True)
    print(f"iterations={ITERATIONS}, seed={SEED}", flush=True)
    print(flush=True)

    results = []
    # for a in FIXED_ALPHAS:
    #     results.append(run(f"fixed α={a}", make_cfg(alpha_grid=[a]), instance))

    for policy in ALPHA_POLICIES:
        results.append(run(f"policy={policy}", make_cfg(alpha_policy=policy), instance))

    plot_topology_comparison(results, instance, save_path="results/topology_comparison.png")

    # Summary table
    print(f"\n{'='*80}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'Variant':<18}  {'Time':>6}  {'Routes':>6}  {'ExactCost':>10}  "
          f"{'Travel':>8}  {'Recourse(true)':>14}  {'Total(true)':>12}", flush=True)
    print("-" * 80, flush=True)
    for r in results:
        print(f"{r['label']:<18}  {r['elapsed']:>5.1f}s  {r['routes']:>6}  "
              f"{r['exact_cost']:>10.3f}  {r['travel']:>8.3f}  "
              f"{r['true_recourse']:>14.3f}  {r['true_total']:>12.3f}", flush=True)

    # Topology comparison — which customers are split, and route groups
    def topology(result):
        """Extract frozenset of frozensets of original customer IDs per route."""
        groups = []
        for rd in result["route_details"]:
            groups.append(frozenset(rd["customers"]))
        return frozenset(groups)

    def split_customers(result, solution):
        """Return set of original_ids that are split in the best solution."""
        split_ids = set()
        for route in solution.routes:
            for n in route.nodes:
                if not n.is_depot and n.is_split:
                    split_ids.add(getattr(n, "original_id", n.id))
        return split_ids

    print(f"\n{'='*80}", flush=True)
    print("Topology comparison", flush=True)
    ref_topo = topology(results[0])
    ref_label = results[0]["label"]
    for r in results:
        topo = topology(r)
        same = topo == ref_topo
        print(f"  {r['label']:<18}  same_as_{ref_label}: {'YES' if same else 'NO '}", flush=True)

    print(f"\n  Per-variant: split customers in best solution", flush=True)
    for r in results:
        sol = r["solution"]
        splits = set()
        for route in sol.routes:
            for n in route.nodes:
                if not n.is_depot and getattr(n, "is_split", False):
                    splits.add(getattr(n, "original_id", n.id))
        print(f"  {r['label']:<18}  split customers: {sorted(splits)}", flush=True)

    # Route details per variant
    print(f"\n{'='*80}", flush=True)
    for r in results:
        print(f"\n  Routes — {r['label']}", flush=True)
        print(f"  {'#':<3}  {'Customers':<28}  {'Travel':>8}  {'Recourse':>10}", flush=True)
        for i, rd in enumerate(r["route_details"], 1):
            print(f"  {i:<3}  {str(rd['customers']):<28}  {rd['travel']:>8.3f}  "
                  f"{rd['recourse']:>10.3f}", flush=True)

    # Split insertion diagnostics
    print(f"\n{'='*80}", flush=True)
    print("SplitInsertion diagnostics", flush=True)
    print(f"{'Variant':<18}  {'Splits':>6}  {'AvgAlpha':>9}  {'AvgPos_r1':>10}  "
          f"{'Avg_r1_len':>10}  {'Avg_r2_len':>10}", flush=True)
    print("-" * 70, flush=True)
    for r in results:
        s = r["split_stats"]
        if not s or s.get("splits_attempted", 0) == 0:
            print(f"{r['label']:<18}  {'0':>6}", flush=True)
            continue
        n = s["splits_attempted"]
        avg_alpha = np.mean(s["alpha_chosen"])
        avg_pos = np.mean(s["position_r1"])
        avg_r1 = np.mean(s["r1_len"])
        avg_r2 = np.mean(s["r2_len"])
        print(f"{r['label']:<18}  {n:>6}  {avg_alpha:>9.3f}  {avg_pos:>10.3f}  "
              f"{avg_r1:>10.1f}  {avg_r2:>10.1f}", flush=True)

    # Operator breakdown for each variant
    print(f"\n{'='*80}", flush=True)
    print("Operator improvement rates", flush=True)
    for r in results:
        solver = r["solver"]
        print(f"\n  {r['label']}", flush=True)
        print(f"  {'Operator':<28}  {'Sel':>5}  {'NewBest':>7}  {'Impr':>6}  {'Acc':>6}  {'Rej':>6}  {'Impr%':>7}", flush=True)
        for i, op in enumerate(solver.insertion_operators):
            bd = solver.insertion_breakdown[i]
            sel = solver.insertion_counts[i]
            impr_pct = (bd['new_best'] + bd['improved']) / sel * 100 if sel > 0 else 0
            print(f"  {type(op).__name__:<28}  {sel:>5}  {bd['new_best']:>7}  "
                  f"{bd['improved']:>6}  {bd['accepted']:>6}  {bd['rejected']:>6}  {impr_pct:>6.1f}%", flush=True)


def plot_topology_comparison(results: list, instance, save_path: str = None):
    """
    2×N grid comparing route topologies across policies.
    - Customers colored by route assignment (consistent palette)
    - Split customers shown as stars with α annotated
    - Recourse cost annotated at each route centroid
    """
    n = len(results)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    axes = np.array(axes).flatten()

    palette = plt.cm.tab10.colors  # 10 distinct colors
    depot = next(nd for nd in instance.nodes if nd.is_depot)
    cust_map = {nd.id: nd for nd in instance.nodes if not nd.is_depot}

    exact_calc = __import__('cost.calculator', fromlist=['ExactCostCalculator']).ExactCostCalculator(
        __import__('core.recourse', fromlist=['PairedVehicleRecourse']).PairedVehicleRecourse()
    )

    for ax, result in zip(axes, results):
        solution = result["solution"]
        label = result["label"]
        exact_cost = result["exact_cost"]
        true_total = result["true_total"]

        ax.scatter(depot.x, depot.y, c='black', s=200, marker='s', zorder=6)
        ax.annotate("D", (depot.x, depot.y), fontsize=8, ha='center', va='center',
                    color='white', fontweight='bold', zorder=7)

        # Draw all customer dots faintly first
        for nd in cust_map.values():
            ax.scatter(nd.x, nd.y, c='lightgray', s=60, zorder=2)

        for r_idx, route in enumerate(solution.routes):
            color = palette[r_idx % len(palette)]
            customers = [nd for nd in route.nodes if not nd.is_depot]
            if not customers:
                continue

            # Draw route line
            xs = [nd.x for nd in route.nodes]
            ys = [nd.y for nd in route.nodes]
            ax.plot(xs, ys, color=color, linewidth=1.8, alpha=0.7, zorder=3)

            # Draw customer nodes
            for nd in customers:
                if nd.is_split:
                    ax.scatter(nd.x, nd.y, c=[color], s=180, marker='*',
                               edgecolors='black', linewidths=0.6, zorder=5)
                    ax.annotate(f"{getattr(nd, 'original_id', nd.id)}\nα={nd.alpha:.2f}",
                                (nd.x, nd.y), fontsize=6, ha='left', va='bottom',
                                xytext=(3, 3), textcoords='offset points', color=color)
                else:
                    ax.scatter(nd.x, nd.y, c=[color], s=70, edgecolors='black',
                               linewidths=0.4, zorder=4)
                    ax.annotate(str(nd.id), (nd.x, nd.y), fontsize=6,
                                ha='center', va='bottom', xytext=(0, 3),
                                textcoords='offset points')

            # Annotate recourse cost at route centroid
            rec = exact_calc.compute_recourse_cost(route)
            cx = np.mean([nd.x for nd in customers])
            cy = np.mean([nd.y for nd in customers])
            ax.annotate(f"rec={rec:.0f}", (cx, cy), fontsize=6.5,
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', fc=color, alpha=0.3, ec='none'),
                        zorder=8)

        ax.set_title(f"{label}\nexact={exact_cost:.1f}  true={true_total:.1f}",
                     fontsize=9, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Hide unused subplots
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout(pad=1.5)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved topology plot to {save_path}", flush=True)
    plt.show()


if __name__ == "__main__":
    main()



