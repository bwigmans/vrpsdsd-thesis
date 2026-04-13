

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from core.route import Route


class RecoursePolicy(ABC):
    @abstractmethod
    def compute_cost(self, route: Route, demand_realization: List[int]) -> float:
        """Compute recourse cost for given demand realization."""
        pass

class PairedVehicleRecourse(RecoursePolicy):
    def __init__(self, paired_routes: Dict[Route, Route]):
        """Initialize with route pairings for split deliveries."""
        self.paired_routes = paired_routes
    
    def compute_cost(self, route: Route, demand_realization: List[int]) -> float:
        """Implement paired vehicle recourse policy from paper."""
        pass



    
    def _handle_type1_failure(self, position: int, remaining_load: float) -> float:
        """Handle demand > remaining load."""
        pass
        
    
    def _handle_type2_failure(self, position: int) -> float:
        """Handle demand = remaining load."""
        pass
    
    def _get_paired_route(self, route: Route) -> Optional[Route]:
        """Get route paired with given route."""
        return self.paired_routes.get(route, None)