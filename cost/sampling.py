from typing import Dict, Optional, Tuple
import numpy as np

from cost.calculator import CostCalculator
from core.route import Route
from core.recourse import RecoursePolicy
from cost.sampling_strategy import SamplingStrategy


def _route_signature(route: Route) -> Tuple:
    """Hashable fingerprint of a route's node sequence and split proportions."""
    return tuple(
        (n.id, getattr(n, "original_id", n.id), round(getattr(n, "alpha", 1.0), 6), n.is_split, n.is_depot)
        for n in route.nodes
    )


class SamplingCostCalculator(CostCalculator):
    def __init__(self, recourse_policy: RecoursePolicy, sampling_strategy: SamplingStrategy):
        self.recourse_policy = recourse_policy
        self.sampling_strategy = sampling_strategy
        self._cache: Dict[Tuple, float] = {}

    def compute_recourse_cost(
        self,
        route: Route,
        samples: Optional[np.ndarray] = None,
        paired_route=None,
    ) -> float:
        """Approximate expected recourse cost via Monte Carlo sampling, with per-route caching."""
        if samples is not None:
            # Bypass cache when explicit samples are provided
            sample_costs = self.sampling_strategy.sample(
                route, self.sampling_strategy.num_samples, samples=samples,
                paired_route=paired_route,
            )
            return float(np.mean(sample_costs))

        sig = _route_signature(route)
        if sig in self._cache:
            return self._cache[sig]

        sample_costs = self.sampling_strategy.sample(
            route, self.sampling_strategy.num_samples, paired_route=paired_route,
        )
        cost = float(np.mean(sample_costs))
        self._cache[sig] = cost
        return cost

    def evaluate_solution(self, solution) -> float:
        """Coordinated solution evaluation. Paired routes use compute_split_pair_costs
        (alpha_r1 + alpha_r2 = 1 guaranteed). Falls back to get_total_cost otherwise."""
        from core.recourse import AdaptivePairedVehicleRecourse
        if isinstance(self.recourse_policy, AdaptivePairedVehicleRecourse):
            samples = self.sampling_strategy._precomputed
            if samples is not None:
                return solution.get_total_cost_adaptive(self.recourse_policy, samples)
        return solution.get_total_cost(self)

    def invalidate_cache(self) -> None:
        """Clear the route cost cache (call when starting a new major phase)."""
        self._cache.clear()
