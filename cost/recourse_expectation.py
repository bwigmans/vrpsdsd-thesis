from typing import Dict, Optional

import numpy as np

from core.instance import Node
from core.recourse import PairedVehicleRecourse, RecoursePolicy
from core.route import Route
from core.solution import Solution
from cost.sampling_strategy import MonteCarloStrategy


def compute_vertex_recourse(
    route: Route,
    method: str = "exact",
    recourse_policy: Optional[RecoursePolicy] = None,
    sampling_strategy: Optional[MonteCarloStrategy] = None,
) -> Dict[Node, float]:
    """Compute expected recourse cost contributions per vertex."""
    recourse_policy = recourse_policy or PairedVehicleRecourse()

    if method == "exact":
        return _exact_vertex_recourse(route)

    if method == "sampling":
        if sampling_strategy is None:
            raise ValueError("sampling_strategy is required for method='sampling'")
        return _sampled_vertex_recourse(route, recourse_policy, sampling_strategy)

    raise ValueError(f"Unsupported method: {method}")


def compute_all_vertex_recourse_costs(
    solution: Solution,
    method: str = "sampling",
    **kwargs,
) -> Dict[Node, float]:
    """Compute expected recourse costs for all nodes in a solution."""
    vertex_costs: Dict[Node, float] = {}
    for route in solution.routes:
        vertex_costs.update(compute_vertex_recourse(route, method=method, **kwargs))
    return vertex_costs


def _exact_vertex_recourse(route: Route) -> Dict[Node, float]:
    customers = [node for node in route.nodes if not node.is_depot]
    if not customers:
        return {}

    probs = route.failure_probabilities()
    costs: Dict[Node, float] = {}
    depot = route.nodes[0]

    for idx, node in enumerate(route.nodes[1:-1], start=1):
        next_node = route.nodes[idx + 1] if idx + 1 < len(route.nodes) else depot
        s_i = 2 * route.instance.get_distance(node, depot)
        s_bar = (
            route.instance.get_distance(node, depot)
            + route.instance.get_distance(depot, next_node)
            - route.instance.get_distance(node, next_node)
        )
        prob_second = route.second_type_failure_probability(idx)
        total_failure = probs[idx - 1] if idx - 1 < len(probs) else 0.0
        prob_first = total_failure - prob_second
        costs[node] = (prob_first * s_i) + (prob_second * s_bar)

    return costs


def _sampled_vertex_recourse(
    route: Route,
    recourse_policy: RecoursePolicy,
    sampling_strategy: MonteCarloStrategy,
) -> Dict[Node, float]:
    customers = [node for node in route.nodes if not node.is_depot]
    if not customers:
        return {}

    acc = {node: 0.0 for node in customers}
    rng = np.random.default_rng(sampling_strategy.seed)

    for _ in range(sampling_strategy.num_samples):
        demands = sampling_strategy.generate_demands(route, rng)
        costs = recourse_policy.compute_vertex_costs(route, demands)
        for node, cost in costs.items():
            acc[node] += cost

    return {node: acc[node] / sampling_strategy.num_samples for node in customers}
