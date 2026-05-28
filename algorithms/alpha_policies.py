from typing import Tuple

from core.route import Route


def compute_alpha(policy: str, r1: Route, r2: Route) -> Tuple[float, float]:
    """Compute split fractions for two routes using the selected policy."""
    if policy == "lei":
        n1 = sum(n.mean_demand * n.alpha if n.is_split else n.mean_demand for n in r1.nodes if not n.is_depot)
        n2 = sum(n.mean_demand * n.alpha if n.is_split else n.mean_demand for n in r2.nodes if not n.is_depot)
        total = n1 + n2
        alpha1 = n2 / total if total > 0 else 0.5
    elif policy == "adaptive":
        alpha1 = 0.5
    else:
        raise ValueError(f"Unknown alpha_policy: {policy}")
    return alpha1, 1.0 - alpha1
