from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
from core.route import Route


class RecoursePolicy(ABC):
    @abstractmethod
    def compute_cost(
        self, route: Route, demand_realization: List[float], paired_route=None
    ) -> float:
        """Compute recourse cost for given demand realization."""
        pass

class PairedVehicleRecourse(RecoursePolicy):
    def __init__(self, paired_routes: Dict[Route, Route] = None):
        """Initialize with route pairings for split deliveries."""
        self.paired_routes = paired_routes or {}

    def compute_cost(
        self, route: Route, demand_realization: List[float], paired_route: Optional[Route] = None
    ) -> float:
        """
        Simulate the route with realized demands and compute extra recourse cost.
        Uses non‑cooperative paired vehicle policy (cooperative not implemented as it
        yields negligible gains per the paper).
        paired_route is stored for use by adaptive alpha policies.
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

            demand = demand_total

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


_ORACLE_ALPHA_GRID = [round(i * 0.1, 1) for i in range(0, 11)]



class AdaptivePairedVehicleRecourse(PairedVehicleRecourse):
    """
    Paired vehicle recourse with adaptive alpha at split vertices.

    Modes:
      alpha_policy: delivery-time policy (lei, equalize_slack, marginal_cost)
      oracle_mode='oracle_true': full hindsight — tries all alphas per realization, picks min cost
      oracle_mode='oracle_avg': r1 (closest to depot) samples future demands, picks best alpha
                                in expectation; r2 falls back to node.alpha
    """

    def __init__(self, alpha_policy=None, paired_routes: Dict[Route, Route] = None,
                 oracle_mode: Optional[str] = None, oracle_samples: int = 50,
                 rng=None):
        super().__init__(paired_routes)
        if alpha_policy is None and oracle_mode is None:
            raise ValueError(
                "AdaptivePairedVehicleRecourse requires either alpha_policy or oracle_mode — "
                "both are None, which would silently fall back to node.alpha for all splits"
            )
        self.alpha_policy = alpha_policy
        self.oracle_mode = oracle_mode  # None | 'oracle_true' | 'oracle_avg'
        self.oracle_samples = oracle_samples
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def compute_cost(
        self, route: Route, demand_realization: List[float], paired_route=None
    ) -> float:
        """Per-route approximation used during ALNS search. For coordinated evaluation
        (guaranteed alpha_r1 + alpha_r2 = 1) use compute_split_pair_costs()."""
        if self.oracle_mode is not None:
            raise RuntimeError(
                f"{self.oracle_mode} requires coordinated evaluation — use compute_split_pair_costs()"
            )
        return self._compute_cost_single(route, demand_realization, paired_route)

    def _compute_cost_single(
        self, route: Route, demand_realization: List[float], paired_route=None
    ) -> float:
        """Single-route simulation — approximation for unpaired routes or search."""
        Q = route.instance.vehicle_capacity
        remaining = Q
        total_recourse = 0.0
        nodes = route.nodes
        customers = [n for n in nodes if not n.is_depot]

        if len(demand_realization) != len(customers):
            raise ValueError("demand_realization length must equal number of customers")

        for i, (node, demand_total) in enumerate(zip(customers, demand_realization)):
            next_node = customers[i + 1] if i + 1 < len(customers) else nodes[0]

            if node.is_split:
                if self.alpha_policy is not None:
                    alpha = self.alpha_policy(remaining, demand_total, node, paired_route)
                    alpha = float(np.clip(alpha, 0.0, 1.0))
                else:
                    alpha = float(np.clip(node.alpha, 0.0, 1.0))
                demand = demand_total * alpha
            else:
                demand = demand_total

            if demand > remaining + 1e-9:
                cost, remaining = self._handle_type1_failure(
                    route, node, next_node, remaining, demand
                )
                total_recourse += cost
            elif abs(demand - remaining) < 1e-9:
                cost, remaining = self._handle_type2_failure(
                    route, node, next_node
                )
                total_recourse += cost
            else:
                remaining -= demand

        return total_recourse

    # ── oracle helpers ─────────────────────────────────────────────────────────

    def _sim_from(self, route: Route, customers: list, start: int,
                  remaining: float, demands: List[float]) -> float:
        """Simulate customers[start:] with unscaled demands, return recourse cost."""
        total = 0.0
        for k, (node, demand) in enumerate(zip(customers[start:], demands)):
            next_node = customers[start + k + 1] if start + k + 1 < len(customers) else route.nodes[0]
            if demand > remaining + 1e-9:
                cost, remaining = self._handle_type1_failure(route, node, next_node, remaining, demand)
                total += cost
            elif abs(demand - remaining) < 1e-9:
                cost, remaining = self._handle_type2_failure(route, node, next_node)
                total += cost
            else:
                remaining -= demand
        return total

    def _sim_with_alpha(self, route: Route, customers: list, demands: List[float],
                        alpha: float, original_id: int) -> float:
        """Simulate route applying alpha to the split node for original_id, return recourse cost."""
        Q = route.instance.vehicle_capacity
        remaining = Q
        total = 0.0
        for k, (node, demand_total) in enumerate(zip(customers, demands)):
            next_node = customers[k + 1] if k + 1 < len(customers) else route.nodes[0]
            if node.is_split and getattr(node, 'original_id', node.id) == original_id:
                demand = demand_total * alpha
            else:
                demand = demand_total
            if demand > remaining + 1e-9:
                cost, remaining = self._handle_type1_failure(route, node, next_node, remaining, demand)
                total += cost
            elif abs(demand - remaining) < 1e-9:
                cost, remaining = self._handle_type2_failure(route, node, next_node)
                total += cost
            else:
                remaining -= demand
        return total

    def _best_alpha_by_sampling(self, route: Route, customers: list,
                                 split_idx: int, remaining: float,
                                 demand_total: float,
                                 r2: Route = None, r2_customers: list = None,
                                 original_id: int = None) -> float:
        """Sample oracle_samples future demand vectors; return alpha minimising E[cost_r1 + cost_r2]."""
        node = customers[split_idx]
        next_node = customers[split_idx + 1] if split_idx + 1 < len(customers) else route.nodes[0]
        future_customers = customers[split_idx + 1:]
        alpha_totals = {a: 0.0 for a in _ORACLE_ALPHA_GRID}

        for _ in range(self.oracle_samples):
            # Sample unscaled demands — _sim_from and _sim_with_alpha handle split scaling internally
            future_demands_r1 = [
                float(route.instance.get_demand_distribution(nd).rvs(random_state=self.rng))
                for nd in future_customers
            ]
            future_demands_r2 = None
            if r2 is not None and r2_customers is not None:
                future_demands_r2 = [
                    float(r2.instance.get_demand_distribution(nd).rvs(random_state=self.rng))
                    for nd in r2_customers
                ]

            for alpha in _ORACLE_ALPHA_GRID:
                d = demand_total * alpha
                if d > remaining + 1e-9:
                    c, rem = self._handle_type1_failure(route, node, next_node, remaining, d)
                elif abs(d - remaining) < 1e-9:
                    c, rem = self._handle_type2_failure(route, node, next_node)
                else:
                    c, rem = 0.0, remaining - d
                c += self._sim_from(route, customers, split_idx + 1, rem, future_demands_r1)

                # Add r2's expected cost with complement alpha
                if future_demands_r2 is not None:
                    c += self._sim_with_alpha(r2, r2_customers, future_demands_r2,
                                              round(1.0 - alpha, 10), original_id)

                alpha_totals[alpha] += c

        return min(alpha_totals, key=lambda a: alpha_totals[a])

    def _dist_to_split(self, route: Route, original_id: int) -> float:
        """Cumulative travel distance from depot to the split node for original_id."""
        dist = 0.0
        prev = route.nodes[0]
        for n in route.nodes[1:]:
            dist += route.instance.get_distance(prev, n)
            if n.is_split and getattr(n, 'original_id', n.id) == original_id:
                return dist
            if not n.is_depot:
                prev = n
        return float('inf')

    def _simulate_route(
        self, route: Route, demands: List[float], paired_route=None,
        complement_alpha: Optional[float] = None, original_id: Optional[int] = None,
    ):
        """
        Simulate one route for one sample.
        Returns (recourse_cost, alpha_used). If complement_alpha is set, the split
        partner node delivers (1 - complement_alpha) instead of the policy.
        """
        Q = route.instance.vehicle_capacity
        remaining = Q
        total_recourse = 0.0
        customers = [n for n in route.nodes if not n.is_depot]
        alpha_used = None

        for i, (node, demand_total) in enumerate(zip(customers, demands)):
            next_node = customers[i + 1] if i + 1 < len(customers) else route.nodes[0]

            if node.is_split:
                if complement_alpha is not None and getattr(node, "original_id", node.id) == original_id:
                    alpha = float(np.clip(1.0 - complement_alpha, 0.0, 1.0))
                else:
                    if self.alpha_policy is None:
                        alpha = float(np.clip(node.alpha, 0.0, 1.0))
                    else:
                        alpha = self.alpha_policy(remaining, demand_total, node, paired_route)
                        alpha = float(np.clip(alpha, 0.0, 1.0))
                    alpha_used = alpha
                demand = demand_total * alpha
            else:
                demand = demand_total

            if demand > remaining + 1e-9:
                cost, remaining = self._handle_type1_failure(route, node, next_node, remaining, demand)
                total_recourse += cost
            elif abs(demand - remaining) < 1e-9:
                cost, remaining = self._handle_type2_failure(route, node, next_node)
                total_recourse += cost
            else:
                remaining -= demand

        return total_recourse, alpha_used

    def compute_split_pair_costs(
        self,
        r1: Route, r2: Route,
        demands_r1: List[List[float]],
        demands_r2: List[List[float]],
        original_id: int,
    ):
        """
        Coordinated evaluation: r1 decides alpha, r2 uses 1 - alpha_r1.
        Guarantees alpha_r1 + alpha_r2 = 1 per sample.

        oracle_avg: r1 (closest to depot) samples future demands to pick best alpha,
                    r2 waits and delivers the complement.
        alpha_policy: r1 uses policy(q1_rem, xi_v), r2 uses complement.
        Returns (costs_r1, costs_r2) as numpy arrays of length N.
        """
        N = len(demands_r1)
        costs_r1 = np.zeros(N)
        costs_r2 = np.zeros(N)

        # For all implementable methods: r1 = vehicle that arrives at split node first.
        # oracle_true doesn't need this (hindsight), but apply consistently anyway.
        if self._dist_to_split(r1, original_id) > self._dist_to_split(r2, original_id):
            r1, r2 = r2, r1
            demands_r1, demands_r2 = demands_r2, demands_r1

        r1_customers = [n for n in r1.nodes if not n.is_depot]
        r2_customers = [n for n in r2.nodes if not n.is_depot]

        if self.oracle_mode == 'oracle_true':
            for i in range(N):
                best_cost = float('inf')
                best_alpha = 0.0
                for alpha in _ORACLE_ALPHA_GRID:
                    c1 = self._sim_with_alpha(r1, r1_customers, demands_r1[i], alpha, original_id)
                    c2 = self._sim_with_alpha(r2, r2_customers, demands_r2[i], round(1.0 - alpha, 10), original_id)
                    if c1 + c2 < best_cost:
                        best_cost = c1 + c2
                        best_alpha = alpha
                costs_r1[i] = self._sim_with_alpha(r1, r1_customers, demands_r1[i], best_alpha, original_id)
                costs_r2[i] = self._sim_with_alpha(r2, r2_customers, demands_r2[i], round(1.0 - best_alpha, 10), original_id)

        elif self.oracle_mode == 'oracle_avg':
            for i in range(N):
                cost_r1, alpha_r1 = self._simulate_oracle_avg(
                    r1, r1_customers, demands_r1[i],
                    r2=r2, r2_customers=r2_customers, original_id=original_id,
                )
                cost_r2, _ = self._simulate_route(
                    r2, demands_r2[i], paired_route=r1,
                    complement_alpha=alpha_r1, original_id=original_id,
                )
                costs_r1[i] = cost_r1
                costs_r2[i] = cost_r2
        else:
            for i in range(N):
                cost_r1, alpha_r1 = self._simulate_route(r1, demands_r1[i], paired_route=r2)
                cost_r2, _ = self._simulate_route(
                    r2, demands_r2[i], paired_route=r1,
                    complement_alpha=alpha_r1, original_id=original_id,
                )
                costs_r1[i] = cost_r1
                costs_r2[i] = cost_r2

        return costs_r1, costs_r2

    def _simulate_oracle_avg(self, route: Route, customers: list,
                              demands: List[float],
                              r2: Route = None, r2_customers: list = None,
                              original_id: int = None):
        """
        Simulate route for oracle_avg: at each split node, sample future demands
        for both r1 and r2, pick alpha minimising E[cost_r1 + cost_r2].
        Returns (recourse_cost, alpha_used_at_split).
        """
        Q = route.instance.vehicle_capacity
        remaining = Q
        total_recourse = 0.0
        alpha_used = None

        for i, (node, demand_total) in enumerate(zip(customers, demands)):
            next_node = customers[i + 1] if i + 1 < len(customers) else route.nodes[0]

            if node.is_split:
                alpha = self._best_alpha_by_sampling(
                    route, customers, i, remaining, demand_total,
                    r2=r2, r2_customers=r2_customers, original_id=original_id,
                )
                alpha_used = alpha
                demand = demand_total * alpha
            else:
                demand = demand_total

            if demand > remaining + 1e-9:
                cost, remaining = self._handle_type1_failure(route, node, next_node, remaining, demand)
                total_recourse += cost
            elif abs(demand - remaining) < 1e-9:
                cost, remaining = self._handle_type2_failure(route, node, next_node)
                total_recourse += cost
            else:
                remaining -= demand

        return total_recourse, alpha_used