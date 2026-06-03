from typing import Optional, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy import stats as scipy_stats

from core.route import Route
from core.instance import Node


def _route_load(route: Route) -> float:
    return sum(
        n.mean_demand * n.alpha if n.is_split else n.mean_demand
        for n in route.nodes if not n.is_depot
    )


def _route_variance(route: Route) -> float:
    total_var = 0.0
    for n in route.nodes:
        if n.is_depot:
            continue
        try:
            v = float(route.instance.get_demand_distribution(n).var())
        except Exception:
            v = n.mean_demand  # Poisson fallback: Var = mean
        total_var += (n.alpha ** 2) * v if n.is_split else v
    return total_var


def _route_failure_cost(route: Route) -> float:
    """2 * d(depot, last customer) as proxy for recourse cost on failure."""
    depot = route.nodes[0]
    customers = [n for n in route.nodes if not n.is_depot]
    if not customers:
        return 1.0
    return 2.0 * route.instance.get_distance(depot, customers[-1])


def _survival(route_load: float, route_var: float, threshold: float) -> float:
    """P(D_route > threshold) via normal approximation."""
    std = max(route_var ** 0.5, 1e-9)
    return float(scipy_stats.norm.sf(threshold, loc=route_load, scale=std))


def compute_alpha(
    policy: str,
    r1: Route,
    r2: Route,
    node: Optional[Node] = None,
) -> Tuple[float, float]:
    """Compute planning-time split fractions (alpha1, alpha2=1-alpha1) for a route pair."""
    Q = r1.instance.vehicle_capacity
    n1 = _route_load(r1)
    n2 = _route_load(r2)
    mu = node.mean_demand if node is not None else 0.0

    if policy == "lei":
        total = n1 + n2
        alpha1 = n2 / total if total > 0 else 0.5

    elif policy == "equalize_slack":
        # Set remaining slack equal: Q-n1-α·μ = Q-n2-(1-α)·μ → α = 0.5 + (n2-n1)/(2μ)
        if mu > 0:
            alpha1 = 0.5 + (n2 - n1) / (2.0 * mu)
        else:
            total = n1 + n2
            alpha1 = n2 / total if total > 0 else 0.5

    elif policy == "equalize_std_slack":
        # Equalize standardized slack: (Q-n1-α·μ)/std1 = (Q-n2-(1-α)·μ)/std2
        # → α = [(Q-n1)·std2 - (Q-n2)·std1 + μ·std1] / [μ·(std1+std2)]
        std1 = max(_route_variance(r1) ** 0.5, 1e-9)
        std2 = max(_route_variance(r2) ** 0.5, 1e-9)
        if mu > 0:
            alpha1 = ((Q - n1) * std2 - (Q - n2) * std1 + mu * std1) / (mu * (std1 + std2))
        else:
            alpha1 = 0.5

    elif policy == "marginal_cost":
        # Equalize marginal failure cost: c1·P(D_r1 > Q-α·μ) = c2·P(D_r2 > Q-(1-α)·μ)
        # Solved via bisection.
        c1 = max(_route_failure_cost(r1), 1e-9)
        c2 = max(_route_failure_cost(r2), 1e-9)
        var1 = _route_variance(r1)
        var2 = _route_variance(r2)
        mu_val = max(mu, 1e-9)

        def objective(a: float) -> float:
            return (c1 * _survival(n1, var1, Q - a * mu_val)
                    - c2 * _survival(n2, var2, Q - (1.0 - a) * mu_val))

        try:
            f0, f1 = objective(1e-6), objective(1.0 - 1e-6)
            if f0 * f1 < 0:
                alpha1 = brentq(objective, 1e-6, 1.0 - 1e-6, maxiter=20)
            else:
                # Corner: push load to cheaper-failure vehicle
                alpha1 = 1.0 - 1e-6 if c1 < c2 else 1e-6
        except Exception:
            alpha1 = 0.5

    elif policy == "adaptive":
        alpha1 = 0.5

    else:
        raise ValueError(f"Unknown alpha_policy: {policy}")

    return float(np.clip(alpha1, 0.01, 0.99)), float(np.clip(1.0 - alpha1, 0.01, 0.99))


def _q2_exp_remaining(node, paired_route) -> float:
    """Expected remaining capacity on r2 when it reaches the split vertex."""
    if paired_route is None:
        return None
    original_id = getattr(node, "original_id", node.id)
    Q = paired_route.instance.vehicle_capacity
    load_before = 0.0
    for n in paired_route.nodes:
        if n.is_depot:
            continue
        if n.is_split and getattr(n, "original_id", n.id) == original_id:
            break
        load_before += n.mean_demand * n.alpha if n.is_split else n.mean_demand
    return Q - load_before


def make_recourse_alpha_policy(policy: str):
    """
    Returns an alpha_policy(q1_rem, xi_v, node, paired_route) -> float
    for use with AdaptivePairedVehicleRecourse.
    Uses realized q1_rem and xi_v instead of expected values.
    """
    def _lei(_q1, _xi, node, _pr):
        return float(node.alpha)

    def _equalize_slack(q1_rem, xi_v, node, paired_route):
        if xi_v <= 0:
            return node.alpha
        q2_rem = _q2_exp_remaining(node, paired_route)
        if q2_rem is None:
            return node.alpha
        alpha = (q1_rem - q2_rem + xi_v) / (2.0 * xi_v)
        return float(np.clip(alpha, 0.01, 0.99))

    def _marginal_cost(q1_rem, xi_v, node, paired_route):
        if paired_route is None or xi_v <= 0:
            return node.alpha
        q2_rem = _q2_exp_remaining(node, paired_route)
        if q2_rem is None:
            return node.alpha
        c1 = _route_failure_cost_from_node(node, paired_route, side="r1")
        c2 = _route_failure_cost_from_node(node, paired_route, side="r2")
        # future_mean: proxy for future load mean after split vertex
        future_mean = max(node.mean_demand, 1e-9)
        var1 = var2 = future_mean
        xi = max(xi_v, 1e-9)

        def objective(a):
            return (c1 * _survival(future_mean, var1, q1_rem - a * xi)
                    - c2 * _survival(future_mean, var2, q2_rem - (1.0 - a) * xi))
        try:
            f0, f1 = objective(1e-6), objective(1.0 - 1e-6)
            if f0 * f1 < 0:
                alpha = brentq(objective, 1e-6, 1.0 - 1e-6, maxiter=15)
            else:
                alpha = 1.0 - 1e-6 if c1 < c2 else 1e-6
        except Exception:
            alpha = node.alpha
        return float(np.clip(alpha, 0.01, 0.99))

    policies = {"lei": _lei, "equalize_slack": _equalize_slack, "marginal_cost": _marginal_cost}
    if policy not in policies:
        raise ValueError(f"Unknown recourse alpha policy: {policy}")
    return policies[policy]


def _route_failure_cost_from_node(node, paired_route, side: str) -> float:
    """Failure cost proxy: 2 * d(depot, next node after split vertex)."""
    depot = paired_route.nodes[0]
    original_id = getattr(node, "original_id", node.id)
    nodes = [n for n in paired_route.nodes if not n.is_depot]
    for i, n in enumerate(nodes):
        if n.is_split and getattr(n, "original_id", n.id) == original_id:
            next_n = nodes[i + 1] if i + 1 < len(nodes) else depot
            return max(2.0 * paired_route.instance.get_distance(n, depot), 1e-9)
    return 1.0
