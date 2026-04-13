from typing import List
from scipy.stats import poisson
from instance import Node, ProblemInstance


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
        lam = self.instance.get_expected_demand(node)
        if node.is_split:
            # Assumes Node has attribute split_alpha (float in (0,1))
            return lam * node.alpha
        return lam

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

    def second_type_failure_probability(self, position: int) -> float:
        """
        Probability that demand at vertex `position` exactly fills the remaining capacity.
        Formula (discrete case): sum_{l=1}^{Q} P(ξ_i = l) * P(X_{i-1} = Q - l)
        where Q = vehicle capacity (assumed integer).
        Position is index in self.nodes (must be >= 1, i.e., a customer vertex).
        """
        if position <= 0 or position >= len(self.nodes):
            raise ValueError("Position must be a customer vertex (index >= 1 and < len(nodes))")

        Q = int(self.instance.vehicle_capacity)  # assume integer capacity
        node = self.nodes[position]

        # Cumulative planned demand before this vertex (X_{i-1})
        cum_before = 0.0
        for j in range(1, position):
            cum_before += self._planned_demand(self.nodes[j])

        # Demand distribution for this vertex (full or split)
        lam_vertex = self._planned_demand(node)

        prob = 0.0
        # l runs from 1 to Q (demand values that cause second-type failure)
        for l in range(1, Q + 1):
            # P(ξ_i = l) for Poisson with mean lam_vertex
            p_demand = poisson.pmf(l, lam_vertex)
            # P(X_{i-1} = Q - l)
            p_cum = poisson.pmf(Q - l, cum_before)
            prob += p_demand * p_cum
        return prob

    def failure_probabilities(self) -> List[float]:
        """
        Compute total failure probability at each vertex position (excluding depot).
        Based on Proposition 4: P_i = P(X_{i-1} <= Q-1) - P(X_i <= Q-1)
        Returns list aligned with self.nodes[1:] (first element corresponds to first customer).
        """
        if len(self.nodes) <= 1:
            return []

        Q = int(self.instance.vehicle_capacity)
        probs = []
        cum_lambda = 0.0  # cumulative planned demand up to previous vertex

        for i in range(1, len(self.nodes)):
            node = self.nodes[i]
            # Add planned demand of this vertex to cumulative
            cum_after = cum_lambda + self._planned_demand(node) #cum first

            # P(X_{i-1} <= Q-1) - P(X_i <= Q-1)
            prob = poisson.cdf(Q - 1, cum_lambda) - poisson.cdf(Q - 1, cum_after)
            probs.append(max(0.0, prob))  # guard against tiny negative due to float errors

            cum_lambda = cum_after

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
            return True  # empty or depot-only route

        total_planned = self.expected_load()
        Q = self.instance.vehicle_capacity
        # Use integer Q for CDF
        prob = poisson.cdf(2 * int(Q), total_planned)
        return prob > 0.9
    

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