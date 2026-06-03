from typing import List
import numpy as np
from core.instance import Node, ProblemInstance


class Route:
    def __init__(self, nodes: List[Node], instance: ProblemInstance):
        """Route with sequence of nodes to visit."""
        self.nodes = nodes
        self.instance = instance

    def _planned_demand(self, node: Node) -> float:
        """
        Return the planned demand for this node on this route.
        For unsplit vertices: full expected demand.
        For split vertices: α * expected demand (only the fraction assigned to this route).
        """
        if node.is_split:
            return node.mean_demand * node.alpha
        return node.mean_demand

    def travel_cost(self) -> float:
        """Compute total travel distance."""
        cost = 0.0
        for i in range(len(self.nodes) - 1):
            cost += self.instance.get_distance(self.nodes[i], self.nodes[i + 1])
        return cost

    def expected_load(self) -> float:
        """Compute expected total planned demand on route (respecting split fractions)."""
        load = 0.0
        for node in self.nodes:
            if not node.is_depot:
                load += self._planned_demand(node)
        return load

    def _node_dist(self, node):
        """Return the demand distribution scaled by alpha for split nodes."""
        if not node.is_split:
            return self.instance.get_demand_distribution(node)
        effective_mean = node.mean_demand * node.alpha
        raw = node.demand_distribution
        if hasattr(raw, "dist") and hasattr(raw, "args"):
            raw = raw.dist
        try:
            return raw(mu=effective_mean)
        except TypeError:
            return raw(effective_mean)

    def _cum_pmf(self, up_to_position: int) -> np.ndarray:
        """
        Compute the PMF of cumulative demand for customers at positions 1..(up_to_position-1).
        Returns array p where p[k] = P(X == k) for k = 0, 1, ..., Q.
        Uses numerical convolution — correct for any demand distribution family.
        """
        Q = int(self.instance.vehicle_capacity)
        cum_pmf = np.zeros(Q + 1)
        cum_pmf[0] = 1.0
        for j in range(1, up_to_position):
            node = self.nodes[j]
            if node.is_depot:
                continue
            dist = self._node_dist(node)
            k_vals = np.arange(Q + 1)
            node_pmf = np.array(dist.pmf(k_vals), dtype=float)
            cum_pmf = np.convolve(cum_pmf, node_pmf)[: Q + 1]
        return cum_pmf

    def second_type_failure_probability(self, position: int) -> float:
        """
        Probability that demand at vertex `position` exactly fills the remaining capacity.
        Formula: sum_{l=1}^{Q} P(ξ_i = l) * P(X_{i-1} = Q - l)
        Position is index in self.nodes (must be >= 1, i.e., a customer vertex).
        """
        if position <= 0 or position >= len(self.nodes):
            raise ValueError("Position must be a customer vertex (index >= 1 and < len(nodes))")

        Q = int(self.instance.vehicle_capacity)
        cum_pmf = self._cum_pmf(position)
        dist = self._node_dist(self.nodes[position])

        prob = 0.0
        for l in range(1, Q + 1):
            p_demand = float(dist.pmf(l))
            p_cum = cum_pmf[Q - l] if Q - l >= 0 else 0.0
            prob += p_demand * p_cum
        return prob

    def failure_probabilities(self) -> List[float]:
        """
        Compute total failure probability at each vertex position (excluding depot).
        Based on Proposition 4: P_i = P(X_{i-1} <= Q-1) - P(X_i <= Q-1)
        Returns list aligned with self.nodes[1:] (first element corresponds to first customer).
        Uses numerical convolution — correct for any demand distribution family.
        """
        if len(self.nodes) <= 1:
            return []

        Q = int(self.instance.vehicle_capacity)
        probs = []
        cum_pmf = np.zeros(Q + 1)
        cum_pmf[0] = 1.0  # P(X=0) = 1 before any customer

        for i in range(1, len(self.nodes)):
            node = self.nodes[i]
            if node.is_depot:
                probs.append(0.0)
                continue
            cdf_before = float(np.sum(cum_pmf[:Q]))  # P(X_{i-1} <= Q-1)

            dist = self._node_dist(node)
            k_vals = np.arange(Q + 1)
            node_pmf = np.array(dist.pmf(k_vals), dtype=float)
            cum_pmf = np.convolve(cum_pmf, node_pmf)[: Q + 1]

            cdf_after = float(np.sum(cum_pmf[:Q]))  # P(X_i <= Q-1)
            probs.append(max(0.0, cdf_before - cdf_after))

        return probs

    def split_positions(self) -> List[int]:
        """Identify positions where split deliveries occur."""
        positions = []
        for i, node in enumerate(self.nodes):
            if node.is_split:
                positions.append(i)
        return positions

    def get_segment_load(self, start: int, end: int) -> float:
        """
        Compute expected planned demand for segment of route from start to end-1.
        Uses _planned_demand to respect split fractions.
        """
        load = 0.0
        for i in range(start, end):
            if not self.nodes[i].is_depot:
                load += self._planned_demand(self.nodes[i])
        return load

    def is_feasible(self) -> bool:
        """
        Check if route respects capacity constraints according to Assumption 3:
        P(total planned demand <= 2Q) > 0.9
        Also implicitly assumes each node's demand <= Q (Assumption 2) – user data must satisfy that.
        """
        if len(self.nodes) <= 1:
            return True

        Q = int(self.instance.vehicle_capacity)
        # Need PMF up to 2Q for this check
        cum_pmf = np.zeros(2 * Q + 1)
        cum_pmf[0] = 1.0
        for j in range(1, len(self.nodes)):
            node = self.nodes[j]
            if node.is_depot:
                continue
            dist = self._node_dist(node)
            k_vals = np.arange(2 * Q + 1)
            node_pmf = np.array(dist.pmf(k_vals), dtype=float)
            cum_pmf = np.convolve(cum_pmf, node_pmf)[: 2 * Q + 1]
        return float(np.sum(cum_pmf[: 2 * Q + 1])) > 0.9
    

if __name__ == "__main__":
    # Quick self-test for the Route class
    # Define minimal Node and ProblemInstance stubs for standalone testing
    
    # Create nodes
    depot = Node(0, 0.0, 0.0, 0.0, is_depot=True)
    cust1 = Node(1, 3.0, 4.0, 2.5)
    cust2 = Node(2, 6.0, 8.0, 1.2)
    cust3 = Node(3, 1.0, 1.0, 3.0)

    nodes = [depot, cust1, cust2, cust3]
    instance = ProblemInstance(nodes, vehicle_capacity=10.0)

    # Build a route: depot -> cust1 -> cust2 -> depot
    route_nodes = [depot, cust1, cust2, depot]
    route = Route(route_nodes, instance)  # Note: Route expects core.Node, but duck typing works

    print("=== Route Test ===")
    print(f"Nodes: {[n.id for n in route.nodes]}")
    print(f"Travel cost: {route.travel_cost():.2f}")
    print(f"Expected load: {route.expected_load():.2f}")
    print(f"Failure probabilities per customer: {route.failure_probabilities()}")
    print(f"Split positions: {route.split_positions()}")
    print(f"Segment load (1 to 3): {route.get_segment_load(1, 3):.2f}")  # cust1+cust2
    print(f"Is feasible? {route.is_feasible()}")

    # Test second-type failure probability at position 1 (cust1)
    prob_second = route.second_type_failure_probability(1)
    print(f"Second-type failure probability at cust1: {prob_second:.6f}")

    # Test with split delivery
    print("\n=== Split Delivery Test ===")
    cust_split = Node(4, 5.0, 5.0, 8.0, is_split=True, alpha=0.6)
    route_split_nodes = [depot, cust_split, depot]
    route_split = Route(route_split_nodes, instance)
    print(f"Split node planned demand: {route_split._planned_demand(cust_split):.2f} (60% of 8.0)")
    print(f"Expected load: {route_split.expected_load():.2f}")
    print(f"Failure probability at split node: {route_split.failure_probabilities()[0]:.6f}")
    print(f"Second-type failure probability: {route_split.second_type_failure_probability(1):.6f}")