
from typing import List

from core.route import Route

# Placeholder for the complete solution class that will manage multiple routes and compute overall costs.
class Solution:
    def __init__(self, routes: List[Route]):
        """Complete solution with multiple routes."""
        self.routes = routes

    
    def total_travel_cost(self) -> float:
        """Sum of all route travel costs."""
        return sum(route.travel_cost() for route in self.routes)
        
    
    def total_recourse_cost(self, cost_calculator: 'CostCalculator') -> float:
        """Compute total recourse cost using specified calculator."""
        pass
    
    def is_feasible(self) -> bool:
        """Check if solution respects vehicle capacity constraints."""
        return all(route.is_feasible() for route in self.routes)
    
    def get_total_cost(self, cost_calculator: 'CostCalculator') -> float:
        """Compute total cost (travel + recourse)."""
        pass
    
    def copy(self) -> 'Solution':
        """Create a deep copy of the solution."""
        copied_routes = [Route(route.nodes.copy(), route.instance) for route in self.routes]
        return Solution(copied_routes)
