from typing import List, Optional, Tuple
import random
import numpy as np
from scipy.optimize import minimize_scalar, brentq
from scipy.stats import poisson as poisson_dist
from core.instance import ProblemInstance, Node
from core.route import Route
from core.solution import Solution
from core.recourse import PairedVehicleRecourse
from utils import Configuration
from cost.calculator import ExactCostCalculator
from cost.sampling import SamplingCostCalculator
from cost.sampling_strategy import MonteCarloStrategy
from cost.sample_bank import DemandSampleBank
from algorithms.initial_solution import InitialSolutionBuilder
from algorithms.operators import (
    RandomRemoval,
    SimilarityRemoval,
    DeterministicWorstRemoval,
    RecourseWorstRemoval,
    GreedyInsertion,
    RegretInsertion,
    DemandFailureSortingInsertion,
    SplitInsertion,
    GreedyInsertionEC,
    RegretInsertionEC,
)


class ALNSSolver:
    def __init__(
        self,
        instance: ProblemInstance,
        config: Configuration,
        extra_removal_operators: list = None,
        extra_insertion_operators: list = None,
    ):
        """Initialize solver with problem instance and configuration parameters."""
        self.instance = instance
        self.config = config
        self.rng = random.Random(config.seed)
        self.np_rng = np.random.default_rng(config.seed)

        self.recourse_policy = PairedVehicleRecourse()

        self.sample_bank = None
        if config.cost_method == "exact":
            self.operator_calculator = ExactCostCalculator(self.recourse_policy)
            self.evaluation_calculator = ExactCostCalculator(self.recourse_policy)
        else:
            bank_path = getattr(config, "sample_bank_path", None) or "data/samples/sample_bank.npz"
            self.sample_bank = DemandSampleBank.load_or_create(
                bank_path, instance, seed=config.seed, verbose=config.verbose
            )
            self.operator_strategy = MonteCarloStrategy(self.recourse_policy)
            self.evaluation_strategy = MonteCarloStrategy(self.recourse_policy)
            # Seed operator with segment 0
            self.operator_strategy.set_samples(self.sample_bank.operator_slice(0))
            # Seed evaluation with first eval slice
            self.evaluation_strategy.set_samples(self.sample_bank.eval_slice())
            self.operator_calculator = SamplingCostCalculator(
                self.recourse_policy, self.operator_strategy
            )
            self.evaluation_calculator = SamplingCostCalculator(
                self.recourse_policy, self.evaluation_strategy
            )

        self.removal_operators = [
            RandomRemoval(),
            SimilarityRemoval(),
            DeterministicWorstRemoval(),
            RecourseWorstRemoval(),
        ] + (extra_removal_operators or [])

        ec_operators = (
            [GreedyInsertionEC(self.operator_calculator), RegretInsertionEC(self.operator_calculator)]
            if config.use_ec_operators else []
        )
        self.insertion_operators = [
            GreedyInsertion(),
            RegretInsertion(),
            DemandFailureSortingInsertion(
                operator_calculator=self.operator_calculator,
            ),
            SplitInsertion(
                operator_calculator=self.operator_calculator,
                alpha_policy=config.alpha_policy,
                alpha_grid=config.alpha_grid,
            ),
        ] + ec_operators + (extra_insertion_operators or [])

        self.removal_weights = [1.0] * len(self.removal_operators)
        self.insertion_weights = [1.0] * len(self.insertion_operators)
        self.removal_scores = [0.0] * len(self.removal_operators)
        self.insertion_scores = [0.0] * len(self.insertion_operators)
        self.removal_counts = [0] * len(self.removal_operators)
        self.insertion_counts = [0] * len(self.insertion_operators)

    def solve(self, initial_solution: Solution = None) -> Solution:
        """Main ALNS optimization loop."""
        initial_solution = initial_solution or InitialSolutionBuilder(self.instance).build()

        current_solution = initial_solution.copy()
        best_solution = initial_solution.copy()
        record_cost = best_solution.get_total_cost(self.evaluation_calculator)
        deviation = self.config.rrt_deviation_factor * record_cost
        current_cost = record_cost
        iterations_without_improvement = 0
        lock_splits = self.config.lock_splits

        for iteration in range(self.config.alns_iterations):
            # Refresh operator samples at the start of each segment
            if self.sample_bank is not None and iteration % self.config.alns_segment_length == 0:
                segment_idx = iteration // self.config.alns_segment_length
                self.operator_strategy.set_samples(self.sample_bank.operator_slice(segment_idx))
                if isinstance(self.operator_calculator, SamplingCostCalculator):
                    self.operator_calculator.invalidate_cache()

            removal_idx = self._select_operator(self.removal_weights)
            insertion_idx = self._select_operator(self.insertion_weights)
            removal_op = self.removal_operators[removal_idx]
            insertion_op = self.insertion_operators[insertion_idx]

            new_solution, removed = self._apply_removal(
                current_solution.copy(), removal_op, lock_splits=lock_splits
            )
            new_solution = self._apply_insertion(new_solution, insertion_op, removed)

            new_cost = new_solution.get_total_cost(self.operator_calculator)

            score = self._compute_score(new_cost, current_cost, record_cost)
            self.removal_scores[removal_idx] += score
            self.insertion_scores[insertion_idx] += score
            self.removal_counts[removal_idx] += 1
            self.insertion_counts[insertion_idx] += 1

            if self._accept_solution(new_cost, record_cost, deviation):
                current_solution = new_solution
                current_cost = new_cost
                if new_cost < record_cost:
                    if self.config.alpha_reoptimize:
                        self._reoptimize_alphas(current_solution, label="new-best")
                    # Advance eval to a fresh slice before re-evaluating
                    if self.sample_bank is not None:
                        self.evaluation_strategy.set_samples(self.sample_bank.eval_slice())
                        self.evaluation_calculator.invalidate_cache()
                    accurate_cost = current_solution.get_total_cost(self.evaluation_calculator)
                    current_cost = accurate_cost
                    # Only update record if accurate eval confirms improvement
                    if accurate_cost < record_cost:
                        best_solution = current_solution.copy()
                        record_cost = accurate_cost
                        deviation = self.config.rrt_deviation_factor * record_cost
                    iterations_without_improvement = 0
                else:
                    iterations_without_improvement += 1
            else:
                iterations_without_improvement += 1

            if (iteration + 1) % self.config.alns_segment_length == 0:
                if self.config.alpha_reoptimize:
                    self._reoptimize_alphas(current_solution, label="segment")
                    current_cost = current_solution.get_total_cost(self.evaluation_calculator)
                self._update_weights()
                self._reset_scores()

            if self.config.verbose and (iteration + 1) % self.config.log_frequency == 0:
                print(
                    f"Iteration {iteration + 1}/{self.config.alns_iterations} | "
                    f"record={record_cost:.4f} | deviation={deviation:.4f} | "
                    f"no_improve={iterations_without_improvement}"
                )


        if self.config.alpha_reoptimize:
            self._reoptimize_alphas(best_solution, label="final")

        if self.config.find_split_post:
            split_sol, split_delta = self._find_best_split(best_solution)
            if split_sol is not None:
                if self.config.verbose:
                    print(f"  [split-post] delta={split_delta:+.4f}", flush=True)
                best_solution = split_sol
            elif self.config.verbose:
                print("  [split-post] no improving split found", flush=True)

        return best_solution

    def _apply_removal(
        self, solution: Solution, removal_op, lock_splits: bool = False
    ) -> Tuple[Solution, List[Node]]:
        """Apply removal operator and return updated solution and removed nodes."""
        n_customers = sum(
            1 for route in solution.routes
            for node in route.nodes
            if not node.is_depot
        )
        k = self.rng.randint(
            max(self.config.removal_min, int(0.1 * n_customers)),
            min(self.config.removal_max, int(0.2 * n_customers)),
        )
        k = max(k, 1)

        if isinstance(removal_op, RecourseWorstRemoval):
            precomputed_costs = self._compute_vertex_recourse_costs(solution)
            removed = removal_op.remove(solution, k, precomputed_costs, _rng=self.rng, lock_splits=lock_splits)
        else:
            removed = removal_op.remove(solution, k, rng=self.rng, lock_splits=lock_splits)

        return solution, removed

    def _apply_insertion(
        self, solution: Solution, insertion_op, removed: List[Node]
    ) -> Solution:
        """Apply insertion operator and return updated solution."""
        if isinstance(insertion_op, SplitInsertion):
            operator_samples = self._draw_operator_samples(solution)
            return insertion_op.insert(
                solution, removed, samples=operator_samples
            )
        return insertion_op.insert(solution, removed)

    def _select_operator(self, weights: List[float]) -> int:
        """Roulette wheel selection."""
        total = sum(weights)
        r = self.rng.uniform(0, total)
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return i
        return len(weights) - 1

    def _accept_solution(
        self, new_cost: float, record: float, deviation: float
    ) -> bool:
        """RRT acceptance criterion: accept if new_cost < record + deviation."""
        return new_cost < record + deviation

    def _compute_score(
        self, new_cost: float, current_cost: float, record_cost: float
    ) -> float:
        """Assign score based on outcome of iteration."""
        if new_cost < record_cost:
            return self.config.score_increment["new_best"]
        if new_cost < current_cost:
            return self.config.score_increment["improving"]
        if new_cost < record_cost + self.config.rrt_deviation_factor * record_cost:
            return self.config.score_increment["accepted"]
        return 0.0

    def _update_weights(self) -> None:
        """Update operator weights using Lei et al. formula."""
        chi = self.config.weight_update_decay
        for i in range(len(self.removal_weights)):
            if self.removal_counts[i] > 0:
                self.removal_weights[i] = (
                    self.removal_weights[i] * (1 - chi)
                    + chi * self.removal_scores[i] / self.removal_counts[i]
                )
        for i in range(len(self.insertion_weights)):
            if self.insertion_counts[i] > 0:
                self.insertion_weights[i] = (
                    self.insertion_weights[i] * (1 - chi)
                    + chi * self.insertion_scores[i] / self.insertion_counts[i]
                )

    def _reset_scores(self) -> None:
        """Reset scores and counts after weight update."""
        self.removal_scores = [0.0] * len(self.removal_operators)
        self.insertion_scores = [0.0] * len(self.insertion_operators)
        self.removal_counts = [0] * len(self.removal_operators)
        self.insertion_counts = [0] * len(self.insertion_operators)

    def _reset_weights(self) -> None:
        """Reset operator weights to uniform and clear scores/counts."""
        self.removal_weights = [1.0] * len(self.removal_operators)
        self.insertion_weights = [1.0] * len(self.insertion_operators)
        self._reset_scores()

    def _try_split(
        self, solution: Solution, node: Node, r1, r2, alpha: float
    ) -> Tuple[Optional[Solution], Optional[float]]:
        """Try inserting a split of node across r1/r2 at given alpha. Returns (trial, delta) or (None, None)."""
        trial = solution.copy()
        r1_idx = solution.routes.index(r1)
        r2_idx = solution.routes.index(r2)
        trial_r1 = trial.routes[r1_idx]
        trial_r2 = trial.routes[r2_idx]

        trial_node = next(
            (n for n in trial_r1.nodes if not n.is_depot and n.id == node.id), None
        )
        if trial_node is None:
            return None, None
        trial_node.is_split = True
        trial_node.alpha = alpha
        trial_node.original_id = node.id
        if not trial_r1.is_feasible():
            return None, None

        partner = Node(
            id=node.id, x=node.x, y=node.y, mean_demand=node.mean_demand,
            is_split=True, alpha=round(1.0 - alpha, 10),
        )
        partner.original_id = node.id

        best_pos, best_inc = None, float("inf")
        base_r2 = self.evaluation_calculator.total_expected_cost(trial_r2)
        for pos in range(1, len(trial_r2.nodes)):
            tmp = trial_r2.nodes.copy()
            tmp.insert(pos, partner)
            t = Route(tmp, trial_r2.instance)
            if not t.is_feasible():
                continue
            inc = self.evaluation_calculator.total_expected_cost(t) - base_r2
            if inc < best_inc:
                best_inc, best_pos = inc, pos

        if best_pos is None:
            return None, None

        trial_r2.nodes.insert(best_pos, partner)
        trial.paired_routes[trial_r1] = trial_r2
        trial.paired_routes[trial_r2] = trial_r1

        old_cost = solution.get_total_cost(self.evaluation_calculator)
        new_cost = trial.get_total_cost(self.evaluation_calculator)
        return trial, new_cost - old_cost

    def _find_best_split(self, solution: Solution) -> Tuple[Optional[Solution], float]:
        """Exhaustive search over (customer, route-pair, alpha-grid) for best improving split."""
        alpha_grid = self.config.alpha_grid or [round(i * 0.1, 1) for i in range(1, 10)]
        unpaired = [r for r in solution.routes if r not in solution.paired_routes]
        best_delta, best_trial = 0.0, None
        for r1 in unpaired:
            customers = [n for n in r1.nodes if not n.is_depot and not n.is_split]
            for node in customers:
                for r2 in unpaired:
                    if r2 is r1:
                        continue
                    for alpha in alpha_grid:
                        trial, delta = self._try_split(solution, node, r1, r2, alpha)
                        if delta is not None and delta < best_delta:
                            best_delta, best_trial = delta, trial
        return best_trial, best_delta

    def _compute_vertex_recourse_costs(
        self, solution: Solution
    ) -> dict:
        """Compute per-vertex expected recourse costs for RWR operator."""
        vertex_costs = {}
        for route in solution.routes:
            customers = [n for n in route.nodes if not n.is_depot]
            if len(customers) <= 1:
                continue
            failure_probs = route.failure_probabilities()
            depot = route.nodes[0]
            for i, node in enumerate(customers):
                next_node = route.nodes[i + 2] if i + 2 < len(route.nodes) else depot
                s_i = 2 * route.instance.get_distance(node, depot)
                s_bar = (
                    route.instance.get_distance(node, depot)
                    + route.instance.get_distance(depot, next_node)
                    - route.instance.get_distance(node, next_node)
                )
                prob_second = route.second_type_failure_probability(i + 1)
                prob_first = failure_probs[i] - prob_second
                vertex_costs[node] = prob_first * s_i + prob_second * s_bar
        return vertex_costs

    def _reoptimize_alphas(self, solution: Solution, label: str = "") -> float:
        """For each split pair, find optimal alpha via 1D minimization over (0,1).
        Returns total cost improvement (positive = got cheaper)."""
        visited = set()
        total_improvement = 0.0
        pairs_improved = 0

        for r1, r2 in solution.paired_routes.items():
            key = (min(id(r1), id(r2)), max(id(r1), id(r2)))
            if key in visited:
                continue
            visited.add(key)

            split_r1 = [n for n in r1.nodes if n.is_split]
            split_r2 = [n for n in r2.nodes if n.is_split]
            if not split_r1 or not split_r2:
                continue

            node1, node2 = split_r1[0], split_r2[0]
            if getattr(node1, 'original_id', node1.id) != getattr(node2, 'original_id', node2.id):
                continue

            cost_before = (self.evaluation_calculator.total_expected_cost(r1) +
                           self.evaluation_calculator.total_expected_cost(r2))

            # Compute feasible alpha range: Assumption 3 requires P(Poisson(load) <= 2Q) > 0.9
            Q = self.instance.vehicle_capacity
            m = node1.mean_demand
            load_r1 = sum(n.mean_demand for n in r1.nodes if not n.is_depot and not n.is_split)
            load_r2 = sum(n.mean_demand for n in r2.nodes if not n.is_depot and not n.is_split)
            max_load = brentq(lambda mu: poisson_dist.cdf(2 * int(Q), mu) - 0.9, 0, 2 * Q)
            alpha_max = min(0.99, (max_load - load_r1) / m) if m > 0 else 0.99
            alpha_min = max(0.01, 1.0 - (max_load - load_r2) / m) if m > 0 else 0.01
            if alpha_min >= alpha_max:
                continue

            def cost_fn(alpha):
                node1.alpha = alpha
                node2.alpha = 1.0 - alpha
                return (self.evaluation_calculator.total_expected_cost(r1) +
                        self.evaluation_calculator.total_expected_cost(r2))

            old_alpha = node1.alpha
            result = minimize_scalar(cost_fn, bounds=(alpha_min, alpha_max), method='bounded')
            node1.alpha = result.x
            node2.alpha = 1.0 - result.x

            improvement = cost_before - result.fun
            total_improvement += improvement
            cid = getattr(node1, 'original_id', node1.id)
            if improvement > 1e-6:
                pairs_improved += 1
                if self.config.verbose:
                    print(f"    c{cid}: alpha {old_alpha:.3f} -> {result.x:.3f}  saved {improvement:.4f}", flush=True)
            else:
                if self.config.verbose:
                    print(f"    c{cid}: alpha {old_alpha:.3f} -> {result.x:.3f}  no improvement", flush=True)

        tag = f" [{label}]" if label else ""
        if self.config.verbose:
            if total_improvement > 1e-6:
                print(f"  [alpha-reopt{tag}] {pairs_improved}/{len(visited)} pairs improved, "
                      f"saved {total_improvement:.4f}", flush=True)
            elif visited:
                print(f"  [alpha-reopt{tag}] no improvement ({len(visited)} pairs checked)", flush=True)

        return total_improvement

    def _draw_operator_samples(self, solution: Solution) -> Optional[dict]:
        """Draw demand samples per route for operator-level cost evaluation."""
        if self.config.cost_method == "exact":
            return None
        samples_by_route = {}
        strategy = self.operator_calculator.sampling_strategy
        for route in solution.routes:
            route_samples = []
            for _ in range(self.config.operator_num_samples):
                route_samples.append(strategy.generate_demands(route, rng=self.np_rng))
            samples_by_route[route] = route_samples
        return samples_by_route
