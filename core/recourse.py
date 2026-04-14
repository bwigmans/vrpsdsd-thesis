from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from core.route import Route


class RecoursePolicy(ABC):
    @abstractmethod
    def compute_cost(self, route: Route, demand_realization: List[float]) -> float:
        """Compute recourse cost for given demand realization."""
        pass


class PairedVehicleRecourse(RecoursePolicy):
    def __init__(self, paired_routes: Dict[Route, Route] = None):
        """Initialize with route pairings for split deliveries."""
        self.paired_routes = paired_routes or {}

    def compute_cost(self, route: Route, demand_realization: List[float]) -> float:
        """
        Simulate the route with realized demands and compute extra recourse cost.
        Uses non‑cooperative paired vehicle policy (cooperative not implemented as it
        yields negligible gains per the paper).
        """
        Q = route.instance.vehicle_capacity
        remaining = Q
        total_recourse = 0.0
        nodes = route.nodes
        customers = [n for n in nodes if not n.is_depot]

        if len(demand_realization) != len(customers):
            raise ValueError("demand_realization length must equal number of customers")

        for i, (node, demand_total) in enumerate(zip(customers, demand_realization)):
            # Determine next node (depot if last customer)
            if i + 1 < len(customers):
                next_node = customers[i + 1]
            else:
                next_node = nodes[0]  # depot

            # Planned fraction for split vertices
            demand = demand_total * node.alpha if node.is_split else demand_total

            if demand > remaining + 1e-9:  # Type 1 failure
                cost, remaining = self._handle_type1_failure(
                    route, node, next_node, remaining, demand
                )
                total_recourse += cost
            elif abs(demand - remaining) < 1e-9:  # Type 2 failure (exact)
                cost, remaining = self._handle_type2_failure(
                    route, node, next_node
                )
                total_recourse += cost
            else:  # No failure
                remaining -= demand

        return total_recourse

    def _handle_type1_failure(
        self, route: Route, node, next_node, remaining: float, demand: float
    ) -> Tuple[float, float]:
        """
        Handle Type 1 failure (demand > remaining load).
        Returns (recourse_cost, new_remaining_load).
        Proposition 2 case 1: s_i = 2 * distance(node, depot).
        After reload, vehicle serves leftover demand.
        """
        depot = route.nodes[0]
        s_i = 2 * route.instance.get_distance(node, depot)
        leftover = demand - remaining
        new_remaining = route.instance.vehicle_capacity - leftover
        return s_i, new_remaining

    def _handle_type2_failure(
        self, route: Route, node, next_node
    ) -> Tuple[float, float]:
        """
        Handle Type 2 failure (demand = remaining load).
        Returns (recourse_cost, new_remaining_load).
        Proposition 2 case 2: s̄_i = c(node, depot) + c(depot, next) - c(node, next).
        After exact fill, vehicle reloads to full capacity.
        """
        depot = route.nodes[0]
        s_bar = (
            route.instance.get_distance(node, depot)
            + route.instance.get_distance(depot, next_node)
            - route.instance.get_distance(node, next_node)
        )
        return s_bar, route.instance.vehicle_capacity

    def _get_paired_route(self, route: Route) -> Optional[Route]:
        """Get route paired with given route (for cooperative policy)."""
        return self.paired_routes.get(route, None)