from abc import ABC, abstractmethod
from typing import List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.recourse import RecoursePolicy
from core.route import Route
from core.instance import Node, ProblemInstance


class CostCalculator(ABC):
    """Abstract base class for cost calculation strategies."""

    @abstractmethod
    def compute_recourse_cost(self, route: Route) -> float:
        """Compute expected recourse cost E[ψ(r_k)] for a single route."""
        pass

    def total_expected_cost(self, route: Route) -> float:
        """Return φ(r_k) + E[ψ(r_k)] (Equation 1)."""
        return route.travel_cost() + self.compute_recourse_cost(route)


class ExactCostCalculator(CostCalculator):
    """
    Exact cost calculator using Poisson probability formulas.
    Implements Proposition 5 (discrete case) from the paper.
    """

    def __init__(self, recourse_policy: RecoursePolicy):
        self.recourse_policy = recourse_policy

    def compute_recourse_cost(self, route: Route) -> float:
        """
        Compute exact expected recourse cost using Poisson distributions.
        Formula: Σ_i (κ_i * s_i + λ_i * s̄_i) where:
          κ_i = probability of first‑type failure (overflow)
          λ_i = probability of second‑type failure (exact fill)
          s_i, s̄_i = recourse costs from Proposition 2.
        """
        if len(route.nodes) <= 2:  # only depot or depot+single customer?
            return 0.0

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

        return total
    
if __name__ == "__main__":
    # Test ExactCostCalculator with a simple route
   
   
    # from core.recourse import RecoursePolicy  # placeholder, not used
    

    # Create a dummy recourse policy (not used by calculator)
    class DummyRecoursePolicy(RecoursePolicy):
        def compute_cost(self, route, demand_realization):
            raise NotImplementedError

    # Create nodes
    depot = Node(0, 0.0, 0.0, 0.0, is_depot=True)
    cust1 = Node(1, 3.0, 4.0, 2.5)          # unsplit, lambda=2.5
    cust2 = Node(2, 6.0, 8.0, 1.2)          # unsplit, lambda=1.2
    cust_split = Node(3, 5.0, 5.0, 8.0, is_split=True, alpha=0.6)  # split, 60% = 4.8

    # Instance with capacity 10
    instance = ProblemInstance([depot, cust1, cust2, cust_split], vehicle_capacity=10.0)

    # Build route: depot -> cust1 -> cust2 -> depot
    route1 = Route([depot, cust1, cust2, depot], instance)

    # Build route with split node: depot -> cust_split -> depot
    route2 = Route([depot, cust_split, depot], instance)

    # Cost calculator
    calc = ExactCostCalculator(DummyRecoursePolicy())
    print("=== Testing ExactCostCalculator ===")
    print(f"Route 1 (unsplit):")
    print(f"  Travel cost: {route1.travel_cost():.2f}")
    print(f"  Expected recourse cost: {calc.compute_recourse_cost(route1):.6f}")
    print(f"  Total expected cost: {calc.total_expected_cost(route1):.2f}")

    print(f"\nRoute 2 (split node, alpha=0.6):")
    print(f"  Planned demand: {route2.expected_load():.2f}")
    print(f"  Travel cost: {route2.travel_cost():.2f}")
    print(f"  Failure prob: {route2.failure_probabilities()[0]:.6f}")
    print(f"  Second-type prob: {route2.second_type_failure_probability(1):.6f}")
    print(f"  Expected recourse cost: {calc.compute_recourse_cost(route2):.6f}")
    print(f"  Total expected cost: {calc.total_expected_cost(route2):.2f}")