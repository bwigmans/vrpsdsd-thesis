from abc import ABC, abstractmethod
from typing import List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.recourse import RecoursePolicy
from core.route import Route
import numpy as np


class CostCalculator(ABC):
    """Abstract base class for cost calculation strategies."""

    @abstractmethod
    def compute_recourse_cost(
        self, route: Route, samples: Optional[np.ndarray] = None,
        paired_route: Optional[Route] = None,
    ) -> float:
        """Compute expected recourse cost E[ψ(r_k)] for a single route."""
        pass

    def total_expected_cost(
        self, route: Route, samples: Optional[np.ndarray] = None,
        paired_route: Optional[Route] = None,
    ) -> float:
        """Return φ(r_k) + E[ψ(r_k)] (Equation 1)."""
        return route.travel_cost() + self.compute_recourse_cost(
            route, samples=samples, paired_route=paired_route
        )

    def evaluate_solution(self, solution) -> float:
        """Evaluate full solution cost. Override in subclasses for coordinated evaluation."""
        return solution.get_total_cost(self)


class ExactCostCalculator(CostCalculator):
    """
    Exact cost calculator using Poisson probability formulas.
    Implements Proposition 5 (discrete case) from the paper.
    """

    def __init__(self, recourse_policy: RecoursePolicy, cache: bool = False):
        self.recourse_policy = recourse_policy
        self._cache: dict = {} if cache else None

    def invalidate_cache(self) -> None:
        if self._cache is not None:
            self._cache.clear()

    def compute_recourse_cost(
        self, route: Route, samples: Optional[np.ndarray] = None,
        paired_route: Optional[Route] = None,
    ) -> float:
        """
        Compute exact expected recourse cost using Poisson distributions.
        Formula: Σ_i (κ_i * s_i + λ_i * s̄_i) where:
          κ_i = probability of first‑type failure (overflow)
          λ_i = probability of second‑type failure (exact fill)
          s_i, s̄_i = recourse costs from Proposition 2.
        """
        if samples is not None:
            raise ValueError("ExactCostCalculator does not accept precomputed samples.")
        if len(route.nodes) <= 2:
            return 0.0

        if self._cache is not None:
            from cost.sampling import _route_signature
            sig = _route_signature(route)
            if sig in self._cache:
                return self._cache[sig]

        total = 0.0
        # Iterate over customer positions (1 .. n-1, since last is depot)
        for i in range(1, len(route.nodes) - 1):
            node = route.nodes[i]
            next_node = route.nodes[i + 1] if i + 1 < len(route.nodes) else route.nodes[0]
            depot = route.nodes[0]

            # Recourse costs from Proposition 2
            s_i = 2 * route.instance.get_distance(node, depot)
            s_bar = (route.instance.get_distance(node, depot) +
                     route.instance.get_distance(depot, next_node) -
                     route.instance.get_distance(node, next_node))

            # Probabilities
            prob_second = route.second_type_failure_probability(i)
            # Total failure probability at this vertex (from Proposition 4)
            total_failure = route.failure_probabilities()[i - 1]
            prob_first = total_failure - prob_second

            total += prob_first * s_i + prob_second * s_bar

        if self._cache is not None:
            self._cache[sig] = total
        return total
