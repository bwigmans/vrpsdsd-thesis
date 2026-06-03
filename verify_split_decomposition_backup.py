"""
Verify across 10 seeds: do post-processing splits save recourse while keeping travel fixed?
For each seed: solve with Both+EC (no splits), then run exhaustive split search.
Report travel vs recourse delta for every improving split.
"""
import time
from io_thesis.instance_reader import read_solomon_instance
from core.recourse import PairedVehicleRecourse
from cost.calculator import ExactCostCalculator
from core.route import Route
from core.instance import Node
from core.solution import Solution
from algorithms.alns import ALNSSolver
from algorithms.operators import (
    GreedyInsertion, RegretInsertion,
    DemandFailureSortingInsertion, SplitInsertion, InsertionOperator,
)
from typing import List
from utils import Configuration

ALPHA_GRID = [round(i * 0.1, 1) for i in range(1, 10)]
SEEDS = [42]

calc = ExactCostCalculator(PairedVehicleRecourse())
inst = read_solomon_instance(
    'data/RC101.txt', vehicle_capacity=70.0,
    demand_scale=(0.51, 0.70), num_customers=25,
)


class GreedyInsertionEC(InsertionOperator):
    def __init__(self, operator_calculator):
        self.operator_calculator = operator_calculator

    def insert(self, solution, nodes):
        for node in nodes:
            if solution.customer_present(getattr(node, "original_id", node.id)):
                continue
            best_route, best_pos, best_increase = None, None, float("inf")
            for route in solution.routes:
                base = self.operator_calculator.total_expected_cost(route)
                for pos in range(1, len(route.nodes)):
                    tmp = route.nodes.copy()
                    tmp.insert(pos, node)
                    t = Route(tmp, route.instance)
                    if not t.is_feasible():
                        continue
                    inc = self.operator_calculator.total_expected_cost(t) - base
                    if inc < best_increase:
                        best_increase, best_route, best_pos = inc, route, pos
            if best_route is not None:
                best_route.nodes.insert(best_pos, node)
            else:
                depot = solution.routes[0].nodes[0]
                solution.routes.append(Route([depot, node, depot], solution.routes[0].instance))
        return solution


class RegretInsertionEC(InsertionOperator):
    def __init__(self, operator_calculator):
        self.operator_calculator = operator_calculator

    def insert(self, solution, nodes):
        def route_cost(r): return self.operator_calculator.total_expected_cost(r)

        def best_in_route(route, node):
            base = route_cost(route)
            best_pos, best_inc = None, float("inf")
            for pos in range(1, len(route.nodes)):
                tmp = route.nodes.copy(); tmp.insert(pos, node)
                t = Route(tmp, route.instance)
                if not t.is_feasible(): continue
                inc = route_cost(t) - base
                if inc < best_inc: best_inc, best_pos = inc, pos
            return best_inc, best_pos

        uninserted = [n for n in nodes if not solution.customer_present(getattr(n, "original_id", n.id))]
        while uninserted:
            infeasible, regret_scores = [], []
            for node in uninserted:
                costs = []
                for route in solution.routes:
                    inc, pos = best_in_route(route, node)
                    if pos is not None: costs.append((inc, route, pos))
                if not costs: infeasible.append(node); continue
                costs.sort(key=lambda x: x[0])
                best_inc, best_route, best_pos = costs[0]
                z = len(costs) - 1
                regret = 0.0 if z == 0 else sum(c[0] - best_inc for c in costs[1:]) / z
                regret_scores.append((regret, node, best_route, best_pos))
            if infeasible and not regret_scores:
                node = infeasible[0]; uninserted.remove(node)
                depot = solution.routes[0].nodes[0]
                solution.routes.append(Route([depot, node, depot], solution.routes[0].instance))
                continue
            if not regret_scores: break
            regret_scores.sort(key=lambda x: x[0], reverse=True)
            _, best_node, best_route, best_pos = regret_scores[0]
            best_route.nodes.insert(best_pos, best_node)
            uninserted.remove(best_node)
        return solution


class ALNSSolverBothEC(ALNSSolver):
    def __init__(self, instance, config):
        super().__init__(instance, config)
        self.insertion_operators = [
            GreedyInsertion(),
            GreedyInsertionEC(self.operator_calculator),
            RegretInsertion(),
            RegretInsertionEC(self.operator_calculator),
            DemandFailureSortingInsertion(self.operator_calculator),
            SplitInsertion(self.operator_calculator, config.alpha_policy, config.alpha_grid),
        ]
        self.insertion_weights = [1.0] * len(self.insertion_operators)
        self.insertion_scores = [0.0] * len(self.insertion_operators)
        self.insertion_counts = [0] * len(self.insertion_operators)
        self.insertion_breakdown = [{'new_best': 0, 'improved': 0, 'accepted': 0, 'rejected': 0} for _ in self.insertion_operators]


def try_split(solution, node, r1, r2, alpha):
    trial = solution.copy()
    r1_idx = solution.routes.index(r1)
    r2_idx = solution.routes.index(r2)
    trial_r1 = trial.routes[r1_idx]
    trial_r2 = trial.routes[r2_idx]

    trial_node = next((n for n in trial_r1.nodes if not n.is_depot and n.id == node.id), None)
    if trial_node is None:
        return None, None, None
    trial_node.is_split = True
    trial_node.alpha = alpha
    trial_node.original_id = node.id
    if not trial_r1.is_feasible():
        return None, None, None

    partner = Node(id=node.id, x=node.x, y=node.y, mean_demand=node.mean_demand,
                   is_split=True, alpha=round(1.0 - alpha, 10))
    partner.original_id = node.id

    best_pos, best_increase = None, float('inf')
    base_r2 = calc.total_expected_cost(trial_r2)
    for pos in range(1, len(trial_r2.nodes)):
        tmp = trial_r2.nodes.copy(); tmp.insert(pos, partner)
        t = Route(tmp, trial_r2.instance)
        if not t.is_feasible(): continue
        inc = calc.total_expected_cost(t) - base_r2
        if inc < best_increase: best_increase, best_pos = inc, pos

    if best_pos is None:
        return None, None, None

    trial_r2.nodes.insert(best_pos, partner)
    trial.paired_routes[trial_r1] = trial_r2
    trial.paired_routes[trial_r2] = trial_r1

    old_cost = solution.get_total_cost(calc)
    new_cost = trial.get_total_cost(calc)
    return trial, new_cost - old_cost, new_cost


def post_process_splits(solution):
    baseline = solution.get_total_cost(calc)
    travel_base = sum(r.travel_cost() for r in solution.routes)
    recourse_base = baseline - travel_base

    results = []
    unpaired = [r for r in solution.routes if r not in solution.paired_routes]
    combo_count = 0
    total_combos = sum(len([n for n in r.nodes if not n.is_depot and not n.is_split]) for r in unpaired) * (len(unpaired) - 1) * len(ALPHA_GRID)
    print(f"    Total combos to try: {total_combos}", flush=True)
    for r1 in unpaired:
        customers = [n for n in r1.nodes if not n.is_depot and not n.is_split]
        for node in customers:
            for r2 in unpaired:
                if r2 is r1: continue
                for alpha in ALPHA_GRID:
                    combo_count += 1
                    if combo_count % 50 == 0:
                        print(f"    combo {combo_count}/{total_combos}...", flush=True)
                    trial, delta, new_cost = try_split(solution, node, r1, r2, alpha)
                    if delta is not None and delta < -0.001:
                        travel_new = sum(r.travel_cost() for r in trial.routes)
                        recourse_new = new_cost - travel_new
                        results.append({
                            'customer': node.id,
                            'r1': solution.routes.index(r1),
                            'r2': solution.routes.index(r2),
                            'alpha': alpha,
                            'delta': delta,
                            'delta_travel': travel_new - travel_base,
                            'delta_recourse': recourse_new - recourse_base,
                        })

    results.sort(key=lambda x: x['delta'])
    return baseline, travel_base, recourse_base, results


print("=" * 70)
print("Split decomposition: travel vs recourse delta across 10 seeds")
print("=" * 70)

all_improving = []
seed_summary = []

for seed in SEEDS:
    cfg = Configuration(
        vehicle_capacity=70.0, cost_method='exact',
        alns_iterations=1000, alns_segment_length=50,
        verbose=False, seed=seed,
        alpha_policy='lei', alpha_reoptimize=False,
    )
    t0 = time.time()
    print(f"  Building solver...", flush=True)
    solver = ALNSSolverBothEC(inst, cfg)
    print(f"  Solving...", flush=True)
    sol = solver.solve()
    elapsed = time.time() - t0
    print(f"  Solved in {elapsed:.1f}s, running split search...", flush=True)

    baseline, travel_base, recourse_base, improvements = post_process_splits(sol)
    print(f"  Split search done.", flush=True)
    best_delta = improvements[0]['delta'] if improvements else 0.0

    print(f"\nseed={seed}  baseline={baseline:.3f} (travel={travel_base:.3f} recourse={recourse_base:.3f})  t={elapsed:.1f}s")
    if improvements:
        print(f"  {len(improvements)} improving splits:")
        for r in improvements[:5]:
            print(f"    c{r['customer']} r{r['r1']}+r{r['r2']} a={r['alpha']}  "
                  f"dtotal={r['delta']:+.4f}  dtravel={r['delta_travel']:+.4f}  drecourse={r['delta_recourse']:+.4f}")
        all_improving.extend(improvements)
    else:
        print(f"  no improving splits")

    seed_summary.append((seed, baseline, travel_base, recourse_base, len(improvements), best_delta))

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'seed':>6} {'baseline':>10} {'travel':>10} {'recourse':>10} {'n_splits':>9} {'best_delta':>11}")
print("-" * 60)
for seed, base, tr, rec, n, bd in seed_summary:
    print(f"{seed:>6} {base:>10.3f} {tr:>10.3f} {rec:>10.3f} {n:>9} {bd:>+11.4f}")

if all_improving:
    print(f"\nAcross all {len(all_improving)} improving splits:")
    travel_deltas = [r['delta_travel'] for r in all_improving]
    recourse_deltas = [r['delta_recourse'] for r in all_improving]
    total_deltas = [r['delta'] for r in all_improving]
    print(f"  dtravel:   min={min(travel_deltas):+.4f}  max={max(travel_deltas):+.4f}  avg={sum(travel_deltas)/len(travel_deltas):+.4f}")
    print(f"  drecourse: min={min(recourse_deltas):+.4f}  max={max(recourse_deltas):+.4f}  avg={sum(recourse_deltas)/len(recourse_deltas):+.4f}")
    print(f"  dtotal:    min={min(total_deltas):+.4f}  max={max(total_deltas):+.4f}  avg={sum(total_deltas)/len(total_deltas):+.4f}")
    zero_travel = sum(1 for d in travel_deltas if abs(d) < 0.001)
    print(f"  Splits with zero travel change: {zero_travel}/{len(all_improving)}")
