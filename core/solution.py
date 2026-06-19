
from copy import deepcopy
from typing import Dict, List

import numpy as np

from core.route import Route
from core.instance import Node
from cost.calculator import CostCalculator
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
        return sum(
            cost_calculator.compute_recourse_cost(
                route, paired_route=self.paired_routes.get(route)
            )
            for route in self.routes
        )

    def is_feasible(self) -> bool:
        """Check if solution respects vehicle capacity constraints."""
        return all(route.is_feasible() for route in self.routes)

    def get_total_cost(self, cost_calculator: 'CostCalculator') -> float:
        """Compute total cost (travel + recourse)."""
        return self.total_travel_cost() + self.total_recourse_cost(cost_calculator)

    def get_total_cost_adaptive(self, rec, samples: dict) -> float:
        """
        Coordinated evaluation for AdaptivePairedVehicleRecourse.
        Paired routes are evaluated together via compute_split_pair_costs so
        r1 decides alpha and r2 receives the complement.
        Unpaired routes fall back to independent compute_cost per sample.
        """
        recourse = 0.0
        visited = set()
        N = len(next(iter(samples.values())))

        for route in self.routes:
            partner = self.paired_routes.get(route)
            if partner is not None:
                pair_key = (min(id(route), id(partner)), max(id(route), id(partner)))
                if pair_key in visited:
                    continue
                visited.add(pair_key)

                split_nodes = [n for n in route.nodes if n.is_split] or \
                              [n for n in partner.nodes if n.is_split]
                original_id = getattr(split_nodes[0], 'original_id', split_nodes[0].id) if split_nodes else -1

                r1_custs = [n for n in route.nodes if not n.is_depot]
                r2_custs = [n for n in partner.nodes if not n.is_depot]
                demands_r1 = [[float(samples[getattr(n, 'original_id', n.id)][i]) for n in r1_custs] for i in range(N)]
                demands_r2 = [[float(samples[getattr(n, 'original_id', n.id)][i]) for n in r2_custs] for i in range(N)]

                c1, c2 = rec.compute_split_pair_costs(route, partner, demands_r1, demands_r2, original_id)
                recourse += float(np.mean(c1)) + float(np.mean(c2))
            else:
                custs = [n for n in route.nodes if not n.is_depot]
                costs = []
                for i in range(N):
                    demands = [float(samples[getattr(n, 'original_id', n.id)][i]) for n in custs]
                    costs.append(rec._compute_cost_single(route, demands))
                recourse += float(np.mean(costs))

        return self.total_travel_cost() + recourse
    
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

    def customer_present(self, customer_id: int) -> bool:
        """Return True if any node with this customer id is already in the solution."""
        for route in self.routes:
            for n in route.nodes:
                if n.is_depot:
                    continue
                if getattr(n, "original_id", n.id) == customer_id:
                    return True
        return False

    def remove_node_from_routes(self, node: Node) -> Node:
        """Remove a node from the solution, handling split nodes correctly."""
        if node.is_split:
            original_id = getattr(node, "original_id", node.id)

            for route in self.routes:
                to_remove = [
                    n for n in route.nodes
                    if n.is_split and getattr(n, "original_id", n.id) == original_id
                ]
                for n in to_remove:
                    route.nodes.remove(n)

            # Also remove any non-split counterpart with the same customer id
            for route in self.routes:
                to_remove = [
                    n for n in route.nodes
                    if not n.is_split and not n.is_depot and n.id == original_id
                ]
                for n in to_remove:
                    route.nodes.remove(n)

            routes_to_unpair = [
                route for route in self.routes
                if not any(n.is_split for n in route.nodes)
            ]
            for route in routes_to_unpair:
                paired = self.paired_routes.pop(route, None)
                if paired is not None:
                    self.paired_routes.pop(paired, None)

            self.routes = [r for r in self.routes if any(not n.is_depot for n in r.nodes)]

            return Node(
                original_id,
                node.x,
                node.y,
                node.mean_demand,
                is_depot=False,
                is_split=False,
                alpha=1.0,
                demand_distribution=node.demand_distribution,
            )

        for route in self.routes:
            if node in route.nodes:
                route.nodes.remove(node)
                if not any(not n.is_depot for n in route.nodes):
                    paired = self.paired_routes.pop(route, None)
                    if paired is not None:
                        self.paired_routes.pop(paired, None)
                    self.routes.remove(route)
                break

        # Remove any orphaned split halves for the same customer
        for route in self.routes:
            to_remove = [
                n for n in route.nodes
                if n.is_split and getattr(n, "original_id", n.id) == node.id
            ]
            for n in to_remove:
                route.nodes.remove(n)
        routes_to_unpair = [
            r for r in self.routes
            if not any(n.is_split for n in r.nodes)
        ]
        for r in routes_to_unpair:
            paired = self.paired_routes.pop(r, None)
            if paired is not None:
                self.paired_routes.pop(paired, None)
        self.routes = [r for r in self.routes if any(not n.is_depot for n in r.nodes)]

        return node
