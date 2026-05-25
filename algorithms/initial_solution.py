import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networkx import nodes
import numpy as np
from core.instance import ProblemInstance, Node 
from core.solution import Solution, Route
from typing import List, Optional
from cost.calculator import CostCalculator, ExactCostCalculator, MonteCarloCostCalculator
from cost.sampling import SamplingCostCalculator
from cost.sampling_strategy import MonteCarloStrategy
from core.recourse import RecoursePolicy, PairedVehicleRecourse 
from io_thesis.instance_reader import read_solomon_instance
from io_thesis.vizualtion import SolutionVisualizer
class InitialSolutionBuilder:
    def __init__(self, instance: ProblemInstance):
        """Initialize solution builder."""
        self.instance = instance
        self.depot = instance.nodes[0]
    
    def build(self) -> Solution:
        """Construct initial solution using greedy heuristic from paper."""
        sorted_nodes = self._sort_nodes_by_expected_demand()
        routes = []
        solution = Solution(routes)
        for node in sorted_nodes:
            if not self._cheapest_insertion(solution, node):
                if not self._try_split_insertion(solution, node):
                    new_route = Route([self.depot, node, self.depot], self.instance)
                    if new_route.is_feasible():
                        solution.routes.append(new_route)
        solution.routes = [r for r in solution.routes if len(r.nodes) > 2]
        return solution

    def _sort_nodes_by_expected_demand(self) -> List[Node]:
        """Sort nodes by increasing expected demand."""
        return sorted([n for n in self.instance.nodes if not n.is_depot],
                      key=lambda n: n.mean_demand)

    def _cheapest_insertion(self, solution: Solution, node: Node) -> bool:
        """Insert node at position with minimal cost increase."""
        best_route_idx = -1
        best_pos = -1
        best_cost_increase = float('inf')
        for r_idx, route in enumerate(solution.routes):
            for pos in range(1, len(route.nodes)):  # Try all insertion positions
                new_nodes = route.nodes.copy()
                new_nodes.insert(pos, node)
                temp_route = Route(new_nodes, self.instance)
                if temp_route.is_feasible():
                    cost_increase = temp_route.travel_cost() - route.travel_cost()
                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_route_idx = r_idx
                        best_pos = pos
        if best_route_idx >= 0:
            route = solution.routes[best_route_idx]
            route.nodes.insert(best_pos, node)
            return True
        return False

    def _try_split_insertion(self, solution: Solution, node: Node) -> bool:
        """Allow split delivery if no single route can accommodate demand."""
        if len(solution.routes) < 2:
            return False
        route_loads = [r.expected_load() for r in solution.routes]
        for i in range(len(solution.routes)):
            for j in range(i+1, len(solution.routes)):
                r1 = solution.routes[i]
                r2 = solution.routes[j]
                total_load = route_loads[i] + route_loads[j]
                if total_load == 0:
                    alpha1 = 0.5
                else:
                    alpha1 = route_loads[j] / total_load
                alpha2 = 1.0 - alpha1
                if alpha1 <= 0 or alpha1 >= 1:
                    continue
                node1 = Node(node.id, node.x, node.y, node.mean_demand,
                             is_depot=False, is_split=True, alpha=alpha1)
                node2 = Node(node.id, node.x, node.y, node.mean_demand,
                             is_depot=False, is_split=True, alpha=alpha2)
                if (self._can_insert_split(r1, node1) and
                    self._can_insert_split(r2, node2)):
                    pos1 = self._best_insertion_position(r1, node1)
                    r1.nodes.insert(pos1, node1)
                    pos2 = self._best_insertion_position(r2, node2)
                    r2.nodes.insert(pos2, node2)
                    return True
        return False

    def _can_insert_split(self, route: Route, node: Node) -> bool:
        return self._best_insertion_position(route, node) is not None

    def _best_insertion_position(self, route: Route, node: Node) -> Optional[int]:
        best_pos = None
        best_increase = float('inf')
        for pos in range(1, len(route.nodes)):
            new_nodes = route.nodes.copy()
            new_nodes.insert(pos, node)
            temp_route = Route(new_nodes, self.instance)
            if temp_route.is_feasible():
                increase = temp_route.travel_cost() - route.travel_cost()
                if increase < best_increase:
                    best_increase = increase
                    best_pos = pos
        return best_pos
    
if __name__ == "__main__":
  
    instance = read_solomon_instance('data/C102.txt', vehicle_capacity=70.0)
   
    
    builder = InitialSolutionBuilder(instance)
    solution = builder.build()
    viz = SolutionVisualizer(solution=solution, instance=instance)
    viz.plot_split_node_routes(split_node_id=21)  # Example: visualize routes containing node 1
    exact_calc = ExactCostCalculator(PairedVehicleRecourse())
    print(f"Exact total expected cost: {solution.get_total_cost(exact_calc):.2f}")
    print(f"Total travel cost: {solution.total_travel_cost():.2f}")

    sampling_calc = MonteCarloCostCalculator(PairedVehicleRecourse(), num_samples=1000, seed=42)
    print(f"Sampling total expected cost: {solution.get_total_cost(sampling_calc):.2f}")

    for i, route in enumerate(solution.routes):
        print(f"Route {i+1}: {[n.id for n in route.nodes]}")
        print(f"  Travel cost: {route.travel_cost():.2f}")
        print(f"  Expected load: {route.expected_load():.2f}")
        print(f"  Feasible: {route.is_feasible()}")