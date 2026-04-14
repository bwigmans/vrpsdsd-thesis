from typing import List
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cost.calculator import CostCalculator
from core.route import Route, Node, ProblemInstance
from core.recourse import RecoursePolicy, PairedVehicleRecourse
from cost.sampling_strategy import SamplingStrategy, MonteCarloStrategy
from cost.calculator import ExactCostCalculator

class SamplingCostCalculator(CostCalculator):
    def __init__(self, recourse_policy: RecoursePolicy,
                 sampling_strategy: 'SamplingStrategy'):
        """Initialize with recourse policy and sampling strategy."""
        self.recourse_policy = recourse_policy
        self.sampling_strategy = sampling_strategy

    def compute_recourse_cost(self, route: Route) -> float:
        """Approximate expected recourse cost via Monte Carlo sampling."""
        # Delegate to the sampling strategy which returns a list of sample costs
        sample_costs = self.sampling_strategy.sample(route, self.sampling_strategy.num_samples)
        # Return the average (expected value)
        return float(np.mean(sample_costs))

    def _sample_demands(self, route: Route, num_samples: int) -> List[List[float]]:
        """
        Generate demand realizations for Monte Carlo simulation.
        Returns a list of lists, each inner list contains realized total demands
        for customers in the order they appear on the route (excluding depot).
        """
        customers = [n for n in route.nodes if not n.is_depot]
        samples = []
        rng = np.random.default_rng(self.sampling_strategy.seed)
        for _ in range(num_samples):
            realization = []
            for node in customers:
                lam = route.instance.get_expected_demand(node)
                demand = rng.poisson(lam)
                realization.append(float(demand))
            samples.append(realization)
        return samples
if __name__ == "__main__":
    depot = Node(0, 0, 0, 0, is_depot=True)
    cust1 = Node(1, 3, 4, 4.0)  
    cust2 = Node(2, 6, 8, 4.0)
    instance = ProblemInstance([depot, cust1, cust2], vehicle_capacity=10.0)
    route = Route([depot, cust1, cust2, depot], instance)

    policy = PairedVehicleRecourse()
    exact_calc = ExactCostCalculator(policy)
    exact_cost = exact_calc.compute_recourse_cost(route)
    print(f"Exact expected recourse cost: {exact_cost:.4f}")

    # Sampling with large N and fixed seed
    strategy = MonteCarloStrategy(policy, num_samples=1000, seed=42, parallel=False)
    sample_costs = strategy.sample(route, num_samples=1000)
    sample_mean = np.mean(sample_costs)
    sample_std = np.std(sample_costs)
    print(f"Sampling mean: {sample_mean:.4f} ± {sample_std/np.sqrt(1000):.4f}")
    print(f"Zero cost samples: {np.sum(np.array(sample_costs)==0)} / 1000")

    depot = Node(0, 0, 0, 0, is_depot=True)
    cust1 = Node(1, 3, 4, 5.5)   # λ=5.5 (55% of Q=10)
    cust2 = Node(2, 6, 8, 5.5)   # λ=5.5
    instance = ProblemInstance([depot, cust1, cust2], vehicle_capacity=10.0)
    route = Route([depot, cust1, cust2, depot], instance)

    policy = PairedVehicleRecourse()
    exact_calc = ExactCostCalculator(policy)
    exact_cost = exact_calc.compute_recourse_cost(route)
    print(f"Exact expected recourse cost: {exact_cost:.4f}")

    strategy = MonteCarloStrategy(policy, num_samples=1000, seed=42, parallel=False)
    sample_costs = strategy.sample(route, num_samples=1000)
    sample_mean = np.mean(sample_costs)
    sample_std = np.std(sample_costs)
    print(f"Sampling mean: {sample_mean:.4f} ± {sample_std/np.sqrt(1000):.4f}")
    print(f"Zero cost samples: {np.sum(np.array(sample_costs)==0)} / 1000")

     # Create depot
    depot = Node(0, 0, 0, 0, is_depot=True)

    # Create 10 customers at random positions (fixed for reproducibility)
    np.random.seed(42)
    nodes = [depot]
    for i in range(1, 100):
        x = np.random.uniform(1, 10)
        y = np.random.uniform(1, 10)
        # Mean demand = 1.5 (well below Q=10)
        nodes.append(Node(i, x, y, 1.7))

    instance = ProblemInstance(nodes, vehicle_capacity=100.0)

    # Build a route: depot -> all customers in order -> depot
    route_nodes = [depot] + nodes[1:] + [depot]
    route = Route(route_nodes, instance)

    policy = PairedVehicleRecourse()
    exact_calc = ExactCostCalculator(policy)
    exact_cost = exact_calc.compute_recourse_cost(route)
    print(f"Exact expected recourse cost: {exact_cost:.4f}")

    # Sampling with 10,000 samples for stable comparison
    strategy = MonteCarloStrategy(policy, num_samples=10000, seed=42, parallel=False)
    sample_costs = strategy.sample(route, num_samples=10000)
    sample_mean = np.mean(sample_costs)
    sample_std = np.std(sample_costs)
    print(f"Sampling mean: {sample_mean:.4f} ± {sample_std/np.sqrt(10000):.4f}")
    print(f"Zero cost samples: {np.sum(np.array(sample_costs)==0)} / 10000")