from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import random
import numpy as np
from core.instance import Node
from core.solution import Solution
from core.route import Route
from cost.calculator import CostCalculator
from algorithms.alpha_policies import compute_alpha

class RemovalOperator(ABC):
    """Base class for ALNS removal operators."""

    @abstractmethod
    def remove(self, solution: Solution, k: int, rng: random.Random = None) -> List[Node]:
        """Remove k nodes from the solution and return the removed nodes."""
        pass


class InsertionOperator(ABC):
    """Base class for ALNS insertion operators."""

    @abstractmethod
    def insert(self, solution: Solution, nodes: List[Node]) -> Solution:
        """Insert nodes into the solution and return the updated solution."""
        pass


class RandomRemoval(RemovalOperator):
    """Remove k random customer nodes for diversification."""

    def remove(self, solution: Solution, k: int, rng: random.Random = None, lock_splits: bool = False) -> List[Node]:
        """Randomly remove k customer nodes from the solution."""
        _rng = rng or random
        solution_nodes = [
            node for route in solution.routes
            for node in route.nodes if not node.is_depot and not (lock_splits and node.is_split)
        ]
        seen = set()
        unique_nodes = []
        for node in solution_nodes:
            key = getattr(node, "original_id", node.id)
            if key in seen:
                continue
            seen.add(key)
            unique_nodes.append(node)

        if k > len(unique_nodes):
            k = len(unique_nodes)
        selected = _rng.sample(unique_nodes, k)
        removed = []
        for node in selected:
            removed.append(solution.remove_node_from_routes(node))
        return removed



class SimilarityRemoval(RemovalOperator):
    """Remove k nodes based on relatedness/similarity measure."""

    def remove(self, solution: Solution, k: int, rng: random.Random = None, lock_splits: bool = False) -> List[Node]:
        """Remove k nodes with highest similarity to a seed node."""
        _rng = rng or random
        solution_nodes = [
            node for route in solution.routes
            for node in route.nodes if not node.is_depot and not (lock_splits and node.is_split)
        ]
        seen = set()
        unique_nodes = []
        for node in solution_nodes:
            key = getattr(node, "original_id", node.id)
            if key in seen:
                continue
            seen.add(key)
            unique_nodes.append(node)
        if not unique_nodes:
            return []
        if k > len(unique_nodes):
            k = len(unique_nodes)

        removed: List[Node] = []
        removed_keys = set()

        original_route_of: Dict[Node, int] = {}
        for idx, route in enumerate(solution.routes):
            for node in route.nodes:
                if not node.is_depot:
                    key = getattr(node, "original_id", node.id)
                    if key not in removed_keys and node not in original_route_of:
                        original_route_of[node] = idx

        def node_key(target: Node) -> int:
            return getattr(target, "original_id", target.id)

        def remove_node(target: Node) -> None:
            removed_node = solution.remove_node_from_routes(target)
            removed.append(removed_node)
            removed_keys.add(node_key(removed_node))

        seed = _rng.choice(unique_nodes)
        seed_route_idx = original_route_of.get(seed)
        remove_node(seed)

        while len(removed) < k:
            remaining = [
                node for node in unique_nodes
                if node_key(node) not in removed_keys
            ]
            if not remaining:
                break

            seed = _rng.choice(removed)
            seed_route_idx = None
            for rep, route_idx in original_route_of.items():
                if node_key(rep) == node_key(seed):
                    seed_route_idx = route_idx
                    break

            distances = {}
            dist_max = 0.0
            for node in remaining:
                dist = ((node.x - seed.x) ** 2 + (node.y - seed.y) ** 2) ** 0.5
                distances[node] = dist
                if dist > dist_max:
                    dist_max = dist
            if dist_max == 0.0:
                next_node = _rng.choice(remaining)
                remove_node(next_node)
                continue

            relatedness = []
            for node in remaining:
                dist = distances[node]
                same_route = seed_route_idx is not None and seed_route_idx == original_route_of.get(node)
                t_ij = 0 if same_route else 1
                denom = (dist / dist_max) + t_ij
                related = 0.0 if denom <= 0.0 else 1 / denom
                relatedness.append((related, node))

            relatedness.sort(key=lambda item: item[0], reverse=True)
            index = int((_rng.random() ** 2) * len(relatedness))
            _, next_node = relatedness[index]
            remove_node(next_node)

        return removed





class DeterministicWorstRemoval(RemovalOperator):
    """Remove nodes with highest deterministic cost contribution."""

    def remove(self, solution: Solution, k: int, rng: random.Random = None, lock_splits: bool = False) -> List[Node]:
        """Remove k nodes that yield the largest travel cost savings."""
        solution_nodes = [
            node for route in solution.routes
            for node in route.nodes if not node.is_depot and not (lock_splits and node.is_split)
        ]
        routes = solution.routes
        seen = set()
        for node in solution_nodes:
            seen.add(getattr(node, "original_id", node.id))
        if k > len(seen):
            k = len(seen)
        removed: List[Node] = []
        removed_keys = set()

        def node_key(target: Node) -> int:
            return getattr(target, "original_id", target.id)

        def remove_node(target: Node) -> None:
            removed_node = solution.remove_node_from_routes(target)
            removed.append(removed_node)
            removed_keys.add(node_key(removed_node))

        if len(routes) >= k:
            best_per_route = []
            for route in routes:
                cost_without_removal = route.travel_cost()
                best = None
                for node in route.nodes:
                    if node.is_depot:
                        continue
                    temp_nodes = route.nodes.copy()
                    temp_nodes.remove(node)
                    temp_route = Route(temp_nodes, route.instance)
                    cost_with_removal = temp_route.travel_cost()
                    cost_saving = cost_without_removal - cost_with_removal
                    if best is None or cost_saving > best[0]:
                        best = (cost_saving, node)
                if best is not None:
                    best_per_route.append(best)

            best_per_route.sort(key=lambda item: item[0], reverse=True)
            for _, node in best_per_route[:k]:
                if node_key(node) in removed_keys:
                    continue
                remove_node(node)
        else:
            while len(removed) < k:
                best = None
                for route in solution.routes:
                    cost_without_removal = route.travel_cost()
                    for node in route.nodes:
                        if node.is_depot:
                            continue
                        temp_nodes = route.nodes.copy()
                        temp_nodes.remove(node)
                        temp_route = Route(temp_nodes, route.instance)
                        cost_with_removal = temp_route.travel_cost()
                        cost_saving = cost_without_removal - cost_with_removal
                        if best is None or cost_saving > best[0]:
                            best = (cost_saving, node)

                if best is None:
                    break
                _, node = best
                if node_key(node) in removed_keys:
                    break
                remove_node(node)

        return removed


class RecourseWorstRemoval(RemovalOperator):
    """Remove nodes with highest expected recourse cost contribution."""

    def remove(
        self,
        solution: Solution,
        k: int,
        precomputed_costs: Dict[Node, float] = None,
        _rng: random.Random = None,
        lock_splits: bool = False,
    ) -> List[Node]:
        """Remove k nodes using precomputed expected recourse costs."""
        solution_nodes = [
            node for route in solution.routes
            for node in route.nodes if not node.is_depot and not (lock_splits and node.is_split)
        ]
        seen = set()
        for node in solution_nodes:
            seen.add(getattr(node, "original_id", node.id))
        if k > len(seen):
            k = len(seen)

        removed: List[Node] = []
        removed_keys = set()

        def node_key(target: Node) -> int:
            return getattr(target, "original_id", target.id)

        def remove_node(target: Node) -> None:
            removed_node = solution.remove_node_from_routes(target)
            removed.append(removed_node)
            removed_keys.add(node_key(removed_node))

        route_of: Dict[Node, Route] = {}
        for route in solution.routes:
            for node in route.nodes:
                if not node.is_depot:
                    route_of[node] = route

        per_route: Dict[Route, List[tuple]] = {}
        best_by_key: Dict[int, tuple] = {}
        for node, cost in precomputed_costs.items():
            key = node_key(node)
            existing = best_by_key.get(key)
            if existing is None or cost > existing[0]:
                best_by_key[key] = (cost, node)

        for _, node in best_by_key.values():
            route = route_of.get(node)
            if route is None:
                continue
            customers = [n for n in route.nodes if not n.is_depot]
            if len(customers) <= 1:
                continue
            per_route.setdefault(route, []).append((cost, node))

        for route in per_route:
            per_route[route].sort(key=lambda item: item[0], reverse=True)

        routes = list(per_route.keys())
        if not routes:
            return []

        if len(routes) >= k:
            best_per_route = []
            for route in routes:
                candidates = per_route.get(route, [])
                if not candidates:
                    continue
                best_per_route.append(candidates[0])

            best_per_route.sort(key=lambda item: item[0], reverse=True)
            for _, node in best_per_route[:k]:
                if node_key(node) not in removed_keys:
                    remove_node(node)
        else:
            all_candidates = []
            for candidates in per_route.values():
                all_candidates.extend(candidates)
            all_candidates.sort(key=lambda item: item[0], reverse=True)
            for _, node in all_candidates:
                if len(removed) >= k:
                    break
                if node_key(node) not in removed_keys:
                    remove_node(node)

        return removed


class GreedyInsertion(InsertionOperator):
    """Insert nodes at the least cost-increasing positions."""

    def insert(self, solution: Solution, nodes: List[Node]) -> Solution:
        """Greedily insert nodes to minimize travel or total cost increase."""
        for node in nodes:
            if solution.customer_present(getattr(node, "original_id", node.id)):
                continue
            best_route = None
            best_pos = None
            best_increase = float("inf")

            for route in solution.routes:
                base_cost = route.travel_cost()
                for pos in range(1, len(route.nodes)):
                    temp_nodes = route.nodes.copy()
                    temp_nodes.insert(pos, node)
                    temp_route = Route(temp_nodes, route.instance)
                    if not temp_route.is_feasible():
                        continue
                    increase = temp_route.travel_cost() - base_cost
                    if increase < best_increase:
                        best_increase = increase
                        best_route = route
                        best_pos = pos

            if best_route is not None and best_pos is not None:
                best_route.nodes.insert(best_pos, node)
                continue

            if not solution.routes:
                raise ValueError("Cannot insert into empty solution without a depot route.")

            depot = solution.routes[0].nodes[0]
            new_route = Route([depot, node, depot], solution.routes[0].instance)
            if not new_route.is_feasible():
                raise ValueError("No feasible insertion found for greedy insertion.")
            solution.routes.append(new_route)

        return solution


class SplitInsertion(InsertionOperator):
    """Insert nodes while allowing split deliveries across routes."""

    def __init__(
        self,
        operator_calculator: 'CostCalculator' = None,
        alpha_policy: str = "lei",
        alpha_grid: Optional[List[float]] = None,
    ):
        self.operator_calculator = operator_calculator
        self.alpha_policy = alpha_policy
        self.alpha_grid = alpha_grid

    def insert(
        self,
        solution: Solution,
        nodes: List[Node],
        samples: Optional[dict] = None,
    ) -> Solution:
        """Insert nodes, splitting demand if no single feasible position exists."""

        def route_cost(route: Route) -> float:
            if self.operator_calculator is None:
                return route.travel_cost()
            route_samples = None
            if isinstance(samples, dict):
                candidate = samples.get(route)
                if candidate:
                    expected_len = sum(1 for n in route.nodes if not n.is_depot)
                    if all(len(sample) == expected_len for sample in candidate):
                        route_samples = candidate
            elif samples:
                expected_len = sum(1 for n in route.nodes if not n.is_depot)
                if all(len(sample) == expected_len for sample in samples):
                    route_samples = samples
            return self.operator_calculator.total_expected_cost(route, samples=route_samples)

        def paired_set() -> set:
            return set(solution.paired_routes.keys()) | set(solution.paired_routes.values())

        def is_unpaired(route: Route, paired_routes_set: set) -> bool:
            return route not in paired_routes_set

        def best_insertion(route: Route, node_to_insert: Node):
            base_cost = route_cost(route)
            best_pos = None
            best_increase = float("inf")
            for pos in range(1, len(route.nodes)):
                temp_nodes = route.nodes.copy()
                temp_nodes.insert(pos, node_to_insert)
                temp_route = Route(temp_nodes, route.instance)
                if not temp_route.is_feasible():
                    continue
                increase = route_cost(temp_route) - base_cost
                if increase < best_increase:
                    best_increase = increase
                    best_pos = pos
            return best_increase, best_pos

        def next_split_id() -> int:
            solution._split_id_counter -= 1
            return solution._split_id_counter

        for node in nodes:
            if solution.customer_present(getattr(node, "original_id", node.id)):
                continue
            paired_routes_set = paired_set()
            unpaired_routes = [route for route in solution.routes if is_unpaired(route, paired_routes_set)]

            alpha1 = None
            alpha2 = None
            node1 = None
            node2 = None
            if len(unpaired_routes) >= 2:
                alpha1, alpha2 = compute_alpha(self.alpha_policy, unpaired_routes[0], unpaired_routes[1])
                node1 = Node(
                    next_split_id(),
                    node.x,
                    node.y,
                    node.mean_demand,
                    is_depot=False,
                    is_split=True,
                    alpha=alpha1,
                )
                node1.original_id = node.id

                node2 = Node(
                    next_split_id(),
                    node.x,
                    node.y,
                    node.mean_demand,
                    is_depot=False,
                    is_split=True,
                    alpha=alpha2,
                )
                node2.original_id = node.id

            best_pair = None
            best_pair_cost = float("inf")
            best_positions = None
            best_alphas = None

            if node1 is not None and node2 is not None:
                for i in range(len(unpaired_routes)):
                    for j in range(i + 1, len(unpaired_routes)):
                        r1 = unpaired_routes[i]
                        r2 = unpaired_routes[j]

                        # Skip pairs where one route has no customers — Eq. 12 is undefined
                        r1_customers = [n for n in r1.nodes if not n.is_depot]
                        r2_customers = [n for n in r2.nodes if not n.is_depot]
                        if not r1_customers or not r2_customers:
                            continue

                        if self.alpha_grid:
                            alphas_to_try = [(a, 1.0 - a) for a in self.alpha_grid]
                        else:
                            alphas_to_try = [compute_alpha(self.alpha_policy, r1, r2)]

                        for alpha1, alpha2 in alphas_to_try:
                            node1.alpha = alpha1
                            node2.alpha = alpha2

                            inc1, pos1 = best_insertion(r1, node1)
                            inc2, pos2 = best_insertion(r2, node2)

                            if pos1 is None or pos2 is None:
                                continue

                            pair_cost = inc1 + inc2
                            if pair_cost < best_pair_cost:
                                best_pair_cost = pair_cost
                                best_pair = (r1, r2)
                                best_positions = (pos1, pos2)
                                best_alphas = (alpha1, alpha2)

                if best_pair is not None:
                    r1, r2 = best_pair
                    pos1, pos2 = best_positions
                    node1.alpha, node2.alpha = best_alphas
                    r1.nodes.insert(pos1, node1)
                    r2.nodes.insert(pos2, node2)
                    solution.paired_routes[r1] = r2
                    solution.paired_routes[r2] = r1
                    paired_routes_set.add(r1)
                    paired_routes_set.add(r2)
                    continue

            if not solution.routes:
                raise ValueError("Cannot insert into empty solution without a depot route.")

            best_route = None
            best_pos = None
            best_increase = float("inf")
            for route in solution.routes:
                inc, pos = best_insertion(route, node)
                if pos is None:
                    continue
                if inc < best_increase:
                    best_increase = inc
                    best_route = route
                    best_pos = pos

            if best_route is not None:
                best_route.nodes.insert(best_pos, node)
                continue

            depot = solution.routes[0].nodes[0]
            new_route = Route([depot, node, depot], solution.routes[0].instance)
            if not new_route.is_feasible():
                raise ValueError("No feasible insertion found for split insertion.")
            solution.routes.append(new_route)

        return solution


class RegretInsertion(InsertionOperator):
    """Insert nodes using regret-k lookahead criterion."""

    def insert(
        self,
        solution: Solution,
        nodes: List[Node],
    ) -> Solution:
        """Insert nodes by maximizing regret value across candidate positions."""

        def route_cost(route: Route) -> float:
            return route.travel_cost()

        def best_insertion_in_route(route: Route, node: Node):
            """Returns (best_increase, best_pos) for inserting node into route."""
            base_cost = route_cost(route)
            best_pos = None
            best_increase = float("inf")
            for pos in range(1, len(route.nodes)):
                temp_nodes = route.nodes.copy()
                temp_nodes.insert(pos, node)
                temp_route = Route(temp_nodes, route.instance)
                if not temp_route.is_feasible():
                    continue
                increase = route_cost(temp_route) - base_cost
                if increase < best_increase:
                    best_increase = increase
                    best_pos = pos
            return best_increase, best_pos

        uninserted = [n for n in nodes if not solution.customer_present(getattr(n, "original_id", n.id))]

        while uninserted:
            infeasible_nodes = []
            regret_scores = []

            for node in uninserted:
                route_costs = []
                for route in solution.routes:
                    inc, pos = best_insertion_in_route(route, node)
                    if pos is not None:
                        route_costs.append((inc, route, pos))

                if not route_costs:
                    infeasible_nodes.append(node)
                    continue

                route_costs.sort(key=lambda x: x[0])
                best_inc, best_route, best_pos = route_costs[0]

                z = len(route_costs) - 1
                if z == 0:
                    regret = 0.0
                else:
                    regret = sum(rc[0] - best_inc for rc in route_costs[1:]) / z

                regret_scores.append((regret, node, best_route, best_pos))

            if infeasible_nodes and not regret_scores:
                node = infeasible_nodes[0]
                uninserted.remove(node)
                if not solution.routes:
                    raise ValueError("Cannot insert into empty solution without a depot route.")
                depot = solution.routes[0].nodes[0]
                new_route = Route([depot, node, depot], solution.routes[0].instance)
                solution.routes.append(new_route)
                continue

            if not regret_scores:
                break

            regret_scores.sort(key=lambda x: x[0], reverse=True)
            _, best_node, best_route, best_pos = regret_scores[0]

            best_route.nodes.insert(best_pos, best_node)
            uninserted.remove(best_node)

        return solution

class GreedyInsertionEC(InsertionOperator):
    """Greedy insertion using expected cost (recourse-aware) instead of travel cost."""

    def __init__(self, operator_calculator: 'CostCalculator'):
        self.operator_calculator = operator_calculator

    def insert(self, solution: Solution, nodes: List[Node]) -> Solution:
        for node in nodes:
            if solution.customer_present(getattr(node, "original_id", node.id)):
                continue
            best_route, best_pos, best_increase = None, None, float("inf")
            for route in solution.routes:
                base = self.operator_calculator.total_expected_cost(route)
                for pos in range(1, len(route.nodes)):
                    tmp = route.nodes.copy()
                    tmp.insert(pos, node)
                    t = Route(tmp, route.instance)
                    if not t.is_feasible():
                        continue
                    inc = self.operator_calculator.total_expected_cost(t) - base
                    if inc < best_increase:
                        best_increase, best_route, best_pos = inc, route, pos
            if best_route is not None:
                best_route.nodes.insert(best_pos, node)
            else:
                depot = solution.routes[0].nodes[0]
                solution.routes.append(Route([depot, node, depot], solution.routes[0].instance))
        return solution


class RegretInsertionEC(InsertionOperator):
    """Regret insertion using expected cost (recourse-aware) instead of travel cost."""

    def __init__(self, operator_calculator: 'CostCalculator'):
        self.operator_calculator = operator_calculator

    def insert(self, solution: Solution, nodes: List[Node]) -> Solution:
        def route_cost(r):
            return self.operator_calculator.total_expected_cost(r)

        def best_in_route(route, node):
            base = route_cost(route)
            best_pos, best_inc = None, float("inf")
            for pos in range(1, len(route.nodes)):
                tmp = route.nodes.copy()
                tmp.insert(pos, node)
                t = Route(tmp, route.instance)
                if not t.is_feasible():
                    continue
                inc = route_cost(t) - base
                if inc < best_inc:
                    best_inc, best_pos = inc, pos
            return best_inc, best_pos

        uninserted = [n for n in nodes if not solution.customer_present(getattr(n, "original_id", n.id))]
        while uninserted:
            infeasible, regret_scores = [], []
            for node in uninserted:
                costs = []
                for route in solution.routes:
                    inc, pos = best_in_route(route, node)
                    if pos is not None:
                        costs.append((inc, route, pos))
                if not costs:
                    infeasible.append(node)
                    continue
                costs.sort(key=lambda x: x[0])
                best_inc, best_route, best_pos = costs[0]
                z = len(costs) - 1
                regret = 0.0 if z == 0 else sum(c[0] - best_inc for c in costs[1:]) / z
                regret_scores.append((regret, node, best_route, best_pos))
            if infeasible and not regret_scores:
                node = infeasible[0]
                uninserted.remove(node)
                depot = solution.routes[0].nodes[0]
                solution.routes.append(Route([depot, node, depot], solution.routes[0].instance))
                continue
            if not regret_scores:
                break
            regret_scores.sort(key=lambda x: x[0], reverse=True)
            _, best_node, best_route, best_pos = regret_scores[0]
            best_route.nodes.insert(best_pos, best_node)
            uninserted.remove(best_node)
        return solution


class DemandFailureSortingInsertion(InsertionOperator):
    """Insert nodes sorted by demand and route failure probability."""

    def __init__(self, operator_calculator: 'CostCalculator' = None):
        self.operator_calculator = operator_calculator

    def insert(
        self,
        solution: Solution,
        nodes: List[Node],
    ) -> Solution:
        """Sort nodes by expected demand and insert into routes by failure risk."""

        def route_cost(route: Route) -> float:
            if self.operator_calculator is None:
                return route.travel_cost()
            return self.operator_calculator.total_expected_cost(route)

        def best_insertion_in_route(route: Route, node: Node):
            """Returns (best_increase, best_pos) for inserting node into route."""
            base_cost = route_cost(route)
            best_pos = None
            best_increase = float("inf")
            for pos in range(1, len(route.nodes)):
                temp_nodes = route.nodes.copy()
                temp_nodes.insert(pos, node)
                temp_route = Route(temp_nodes, route.instance)
                if not temp_route.is_feasible():
                    continue
                increase = route_cost(temp_route) - base_cost
                if increase < best_increase:
                    best_increase = increase
                    best_pos = pos
            return best_increase, best_pos

        def route_failure_probability(route: Route) -> float:
            return sum(route.failure_probabilities())

        uninserted = sorted(
            [n for n in nodes if not solution.customer_present(getattr(n, "original_id", n.id))],
            key=lambda n: n.mean_demand, reverse=True,
        )

        for node in uninserted:
            sorted_routes = sorted(solution.routes, key=route_failure_probability)

            inserted = False
            for route in sorted_routes:
                inc, pos = best_insertion_in_route(route, node)
                if pos is not None:
                    route.nodes.insert(pos, node)
                    inserted = True
                    break

            if not inserted:
                if not solution.routes:
                    raise ValueError("Cannot insert into empty solution without a depot route.")
                depot = solution.routes[0].nodes[0]
                new_route = Route([depot, node, depot], solution.routes[0].instance)
                solution.routes.append(new_route)

        return solution
       
            
            

        
