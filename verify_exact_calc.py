"""
Verify ExactCostCalculator matches FirstFailureOnlyRecourse Monte Carlo
for both Poisson and NB demand distributions, with and without split nodes.
"""
import sys
import numpy as np
from scipy.stats import poisson, nbinom

sys.stdout.reconfigure(encoding="utf-8")

from io_thesis.instance_reader import read_solomon_instance
from core.instance import Node, ProblemInstance
from core.route import Route
from core.recourse import PairedVehicleRecourse
from cost.calculator import ExactCostCalculator
from cost.sampling_strategy import MonteCarloStrategy
from cost.sampling import SamplingCostCalculator
from test_sampling_poisson import FirstFailureOnlyRecourse

N_SAMPLES = 500
SEED = 42
TOL = 0.5  # acceptable absolute difference (MC noise at 50k)


def nb_factory(dispersion=5.0):
    """Return a callable(mu) -> frozen NB distribution."""
    def make(mu):
        n = dispersion
        p = n / (n + mu)
        return nbinom(n, p)
    return make


def make_instance(nodes, Q=70.0):
    return ProblemInstance(nodes, vehicle_capacity=Q)


def exact_cost(route):
    calc = ExactCostCalculator(PairedVehicleRecourse())
    return calc.total_expected_cost(route)


def mc_cost(route, seed=SEED):
    rec = FirstFailureOnlyRecourse()
    rng = np.random.default_rng(seed)
    customers = [n for n in route.nodes if not n.is_depot]
    total = 0.0
    for _ in range(N_SAMPLES):
        demands = []
        for nd in customers:
            dist = route._node_dist(nd)
            d = float(dist.rvs(random_state=rng))
            demands.append(d)
        total += rec.compute_cost(route, demands)
    recourse_mc = total / N_SAMPLES
    return route.travel_cost() + recourse_mc


def check(label, route):
    e = exact_cost(route)
    m = mc_cost(route)
    diff = abs(e - m)
    status = "OK " if diff < TOL else "FAIL"
    print(f"  [{status}] {label:<40}  exact={e:.4f}  mc={m:.4f}  diff={diff:.4f}")


def main():
    inst = read_solomon_instance(
        "data/RC101.txt", vehicle_capacity=70.0,
        demand_scale=(0.51, 0.70), num_customers=25,
    )
    node_by_id = {n.id: n for n in inst.nodes}
    depot = node_by_id[0]

    nb = nb_factory(dispersion=5.0)

    print(f"N_SAMPLES={N_SAMPLES:,}   tol={TOL}\n")

    # ── Poisson, no splits ────────────────────────────────────────────────
    print("=== Poisson, no splits ===")
    for cids in [[5, 3, 4], [25, 23, 21], [9, 10, 12]]:
        nodes = [depot] + [node_by_id[c] for c in cids] + [depot]
        route = Route(nodes, inst)
        check(f"r={cids}", route)

    # ── Poisson, with splits ──────────────────────────────────────────────
    print("\n=== Poisson, with splits ===")
    for alpha in [0.3, 0.5, 0.7, 0.9]:
        base = node_by_id[22]
        split_n = Node(base.id, base.x, base.y, base.mean_demand,
                       is_split=True, alpha=alpha,
                       demand_distribution=poisson)
        split_n.original_id = base.id
        nodes = [depot, node_by_id[24], split_n, node_by_id[21], depot]
        route = Route(nodes, inst)
        check(f"split c22 alpha={alpha}  r=[24,22*,21]", route)

    # ── NB, no splits ─────────────────────────────────────────────────────
    print("\n=== NB, no splits ===")
    nb_inst_nodes = []
    for n in inst.nodes:
        nb_node = Node(n.id, n.x, n.y, n.mean_demand,
                       is_depot=n.is_depot, demand_distribution=nb)
        nb_inst_nodes.append(nb_node)
    nb_inst = ProblemInstance(nb_inst_nodes, vehicle_capacity=70.0)
    nb_by_id = {n.id: n for n in nb_inst_nodes}
    nb_depot = nb_by_id[0]

    for cids in [[5, 3, 4], [25, 23, 21], [9, 10, 12]]:
        nodes = [nb_depot] + [nb_by_id[c] for c in cids] + [nb_depot]
        route = Route(nodes, nb_inst)
        check(f"r={cids}", route)

    # ── NB, with splits ───────────────────────────────────────────────────
    print("\n=== NB, with splits ===")
    for alpha in [0.3, 0.5, 0.7, 0.9]:
        base = nb_by_id[22]
        split_n = Node(base.id, base.x, base.y, base.mean_demand,
                       is_split=True, alpha=alpha,
                       demand_distribution=nb)
        split_n.original_id = base.id
        nodes = [nb_depot, nb_by_id[24], split_n, nb_by_id[21], nb_depot]
        route = Route(nodes, nb_inst)
        check(f"split c22 alpha={alpha}  r=[24,22*,21]", route)


if __name__ == "__main__":
    main()
