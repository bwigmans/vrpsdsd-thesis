
from copy import deepcopy
from typing import Dict, List

from core.route import Route
from cost.calculator import CostCalculator, ExactCostCalculator
from core.recourse import RecoursePolicy, PairedVehicleRecourse
# Placeholder for the complete solution class that will manage multiple routes and compute overall costs.
class Solution:
    def __init__(self, routes: List[Route]):
        """Complete solution with multiple routes."""
        self.routes = routes
        self.paired_routes: Dict[Route, Route] = {}
        self._split_id_counter = -1

    
    def total_travel_cost(self) -> float:
        """Sum of all route travel costs."""
        return sum(route.travel_cost() for route in self.routes)
        
    
    def total_recourse_cost(self, cost_calculator: 'CostCalculator') -> float:
        """Compute total recourse cost using specified calculator."""
        return sum(cost_calculator.compute_recourse_cost(route) for route in self.routes)
        
    
    def is_feasible(self) -> bool:
        """Check if solution respects vehicle capacity constraints."""
        return all(route.is_feasible() for route in self.routes)
    
    def get_total_cost(self, cost_calculator: 'CostCalculator') -> float:
        """Compute total cost (travel + recourse)."""
        return self.total_travel_cost() + self.total_recourse_cost(cost_calculator)
    
    def copy(self) -> 'Solution':
        """Create a deep copy of the solution."""
        route_map = {}
        copied_routes = []
        for route in self.routes:
            copied_nodes = deepcopy(route.nodes)
            new_route = Route(copied_nodes, route.instance)
            copied_routes.append(new_route)
            route_map[route] = new_route
        copied_solution = Solution(copied_routes)
        copied_solution._split_id_counter = self._split_id_counter
        for old_route, paired in self.paired_routes.items():
            if old_route in route_map and paired in route_map:
                copied_solution.paired_routes[route_map[old_route]] = route_map[paired]
        return copied_solution
