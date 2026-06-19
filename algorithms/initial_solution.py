import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from core.instance import ProblemInstance, Node 
from core.solution import Solution, Route
from typing import List, Optional
from cost.calculator import CostCalculator, ExactCostCalculator
from cost.sampling import SamplingCostCalculator
from cost.sampling_strategy import MonteCarloStrategy
from core.recourse import RecoursePolicy, PairedVehicleRecourse 
from io_thesis.instance_reader import read_solomon_instance
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
        """
        Lei et al. (2012) Section 4.2 construction split insertion — paper-faithful.

        Finds the FIRST unpaired route r1 where the largest feasible fraction fits,
        then looks for any second unpaired route r2 for the remainder.
        If no r2 exists → return False so build() opens a new unsplit route (paper §4.2).

        The previous implementation (commented out below) searched ALL r1 candidates
        and also created a new paired route when no r2 was found.  That produced
        fewer, denser routes (~12) because it was more aggressive with splits.
        """
        if len(solution.routes) < 2:
            return False

        unpaired_routes = [
            r for r in solution.routes
            if r not in solution.paired_routes and r not in solution.paired_routes.values()
        ]
        if not unpaired_routes:
            return False

        # Step 1: find the FIRST unpaired route r1 where any fraction fits.
        for r1 in unpaired_routes:
            best_pos = None
            best_frac = None
            # Largest feasible fraction (paper: initial value 0.1, increment 0.1 → largest wins)
            for frac in [i / 10 for i in range(9, 0, -1)]:
                node_part = Node(
                    node.id, node.x, node.y, node.mean_demand,
                    is_depot=False, is_split=True, alpha=frac,
                )
                pos = self._best_insertion_position(r1, node_part)
                if pos is not None:
                    best_pos = pos
                    best_frac = frac
                    break

            if best_pos is None:
                continue  # this route can't take any fraction; try next

            # Step 2: search for ANY second unpaired route r2 for the remainder.
            remainder = 1.0 - best_frac
            for r2 in unpaired_routes:
                if r2 is r1:
                    continue
                node_rest = Node(
                    node.id, node.x, node.y, node.mean_demand,
                    is_depot=False, is_split=True, alpha=remainder,
                )
                pos2 = self._best_insertion_position(r2, node_rest)
                if pos2 is None:
                    continue
                # Both routes found — do the paired split insertion.
                node_part = Node(
                    node.id, node.x, node.y, node.mean_demand,
                    is_depot=False, is_split=True, alpha=best_frac,
                )
                r1.nodes.insert(best_pos, node_part)
                r2.nodes.insert(pos2, node_rest)
                solution.paired_routes[r1] = r2
                solution.paired_routes[r2] = r1
                return True

            # r1 found but no r2 — paper §4.2: "a new route is created and the current
            # vertex is inserted into the new route without split."  Return False so
            # build() opens a fresh unsplit route; do NOT modify r1.
            return False

        return False

    # ---------------------------------------------------------------------------
    # Previous _try_split_insertion (commented out — produced ~12 routes by
    # searching all r1 candidates and creating a paired route when no r2 existed;
    # outperformed paper's 16-route construction on cost but diverged from §4.2).
    # ---------------------------------------------------------------------------
    # def _try_split_insertion_aggressive(self, solution, node):
    #     if len(solution.routes) < 2:
    #         return False
    #     unpaired_routes = [r for r in solution.routes
    #                        if r not in solution.paired_routes
    #                        and r not in solution.paired_routes.values()]
    #     if not unpaired_routes:
    #         return False
    #     for route in unpaired_routes:
    #         best_pos = None; best_frac = None
    #         for frac in [i/10 for i in range(9, 0, -1)]:
    #             node_part = Node(node.id, node.x, node.y, node.mean_demand,
    #                              is_depot=False, is_split=True, alpha=frac)
    #             pos = self._best_insertion_position(route, node_part)
    #             if pos is not None:
    #                 best_pos = pos; best_frac = frac; break
    #         if best_pos is None or best_frac is None:
    #             continue
    #         remainder = 1.0 - best_frac
    #         for route2 in unpaired_routes:
    #             if route2 is route:
    #                 continue
    #             node_rest = Node(node.id, node.x, node.y, node.mean_demand,
    #                              is_depot=False, is_split=True, alpha=remainder)
    #             pos2 = self._best_insertion_position(route2, node_rest)
    #             if pos2 is None:
    #                 continue
    #             node_part = Node(node.id, node.x, node.y, node.mean_demand,
    #                              is_depot=False, is_split=True, alpha=best_frac)
    #             route.nodes.insert(best_pos, node_part)
    #             route2.nodes.insert(pos2, node_rest)
    #             solution.paired_routes[route] = route2
    #             solution.paired_routes[route2] = route
    #             return True
    #         # No existing r2 — create new paired route for remainder (deviates from paper)
    #         node_part = Node(node.id, node.x, node.y, node.mean_demand,
    #                          is_depot=False, is_split=True, alpha=best_frac)
    #         node_rest = Node(node.id, node.x, node.y, node.mean_demand,
    #                          is_depot=False, is_split=True, alpha=remainder)
    #         new_route = Route([self.depot, node_rest, self.depot], self.instance)
    #         if new_route.is_feasible():
    #             route.nodes.insert(best_pos, node_part)
    #             solution.routes.append(new_route)
    #             solution.paired_routes[route] = new_route
    #             solution.paired_routes[new_route] = route
    #             return True
    #     return False

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


    # NOTE: Previous heuristic (Lei-style alpha using route loads) kept for reference.
    #
    # def _try_split_insertion(self, solution: Solution, node: Node) -> bool:
    #     """Allow split delivery if no single route can accommodate demand."""
    #     if len(solution.routes) < 2:
    #         return False
    #     route_loads = [r.expected_load() for r in solution.routes]
    #     for i in range(len(solution.routes)):
    #         for j in range(i+1, len(solution.routes)):
    #             r1 = solution.routes[i]
    #             r2 = solution.routes[j]
    #             total_load = route_loads[i] + route_loads[j]
    #             if total_load == 0:
    #                 alpha1 = 0.5
    #             else:
    #                 alpha1 = route_loads[j] / total_load
    #             alpha2 = 1.0 - alpha1
    #             if alpha1 <= 0 or alpha1 >= 1:
    #                 continue
    #             node1 = Node(node.id, node.x, node.y, node.mean_demand,
    #                          is_depot=False, is_split=True, alpha=alpha1)
    #             node2 = Node(node.id, node.x, node.y, node.mean_demand,
    #                          is_depot=False, is_split=True, alpha=alpha2)
    #             if (self._can_insert_split(r1, node1) and
    #                 self._can_insert_split(r2, node2)):
    #                 pos1 = self._best_insertion_position(r1, node1)
    #                 r1.nodes.insert(pos1, node1)
    #                 pos2 = self._best_insertion_position(r2, node2)
    #                 r2.nodes.insert(pos2, node2)
    #                 return True
    #     return False
    
if __name__ == "__main__":
    from io_thesis.vizualtion import SolutionVisualizer
  
    instance = read_solomon_instance('data/C102.txt', vehicle_capacity=70.0)
   
    
    builder = InitialSolutionBuilder(instance)
    solution = builder.build()
    viz = SolutionVisualizer(solution=solution, instance=instance)
    viz.plot_split_node_routes(split_node_id=21)  # Example: visualize routes containing node 1
    exact_calc = ExactCostCalculator(PairedVehicleRecourse())
    print(f"Exact total expected cost: {solution.get_total_cost(exact_calc):.2f}")
    print(f"Total travel cost: {solution.total_travel_cost():.2f}")

    for i, route in enumerate(solution.routes):
        print(f"Route {i+1}: {[n.id for n in route.nodes]}")
        print(f"  Travel cost: {route.travel_cost():.2f}")
        print(f"  Expected load: {route.expected_load():.2f}")
        print(f"  Feasible: {route.is_feasible()}")