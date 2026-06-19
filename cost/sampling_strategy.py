from abc import ABC, abstractmethod
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from core.route import Route
from core.recourse import RecoursePolicy


class SamplingStrategy(ABC):
    @abstractmethod
    def sample(self, route: Route, num_samples: int) -> List[float]:
        """Sample recourse costs for a route."""
        pass


class MonteCarloStrategy(SamplingStrategy):
    def __init__(self, recourse_policy: RecoursePolicy, num_samples: int = 1000,
                 seed: Optional[int] = None, parallel: bool = False,
                 num_threads: Optional[int] = None):
        """
        Initialize Monte Carlo sampling strategy.

        Args:
            recourse_policy: Policy to compute recourse cost for each sample.
            num_samples: Number of demand realizations per route.
            seed: Random seed for reproducibility.
            parallel: If True, use thread pool for parallel sampling.
            num_threads: Number of threads (default: CPU count).
        """
        self.recourse_policy = recourse_policy
        self.num_samples = num_samples
        self.seed = seed
        self.parallel = parallel
        self.num_threads = num_threads or 4
        self.rng = np.random.default_rng(self.seed)
        # Precomputed demand samples: {customer_id: np.array(num_samples,)}
        self._precomputed: Optional[dict] = None

    def sample(
        self,
        route: Route,
        num_samples: Optional[int] = None,
        samples: Optional[np.ndarray] = None,
        paired_route: Optional[Route] = None,
    ) -> List[float]:
        """Perform Monte Carlo sampling of recourse costs."""
        if samples is not None:
            costs = []
            for demands in samples:
                costs.append(self.recourse_policy.compute_cost(
                    route, list(demands), paired_route=paired_route
                ))
            return costs

        ns = num_samples or self.num_samples
        if self.parallel:
            return self._parallel_sample(route, ns, paired_route=paired_route)
        else:
            return self._sequential_sample(route, ns, paired_route=paired_route)

    def set_samples(self, sample_slice: dict) -> None:
        """
        Provide a precomputed slice {customer_id: np.array(N,)} from DemandSampleBank.
        After calling this, num_samples is updated to match the slice length.
        """
        self._precomputed = sample_slice
        first = next(iter(sample_slice.values()))
        self.num_samples = len(first)

    def _sequential_sample(
        self, route: Route, num_samples: int, paired_route: Optional[Route] = None
    ) -> List[float]:
        """Sequential sampling, using precomputed slice when available."""
        customers = [n for n in route.nodes if not n.is_depot]
        costs = []

        from core.recourse import AdaptivePairedVehicleRecourse
        is_adaptive = isinstance(self.recourse_policy, AdaptivePairedVehicleRecourse)

        if self._precomputed is not None:
            for i in range(num_samples):
                demands = []
                for node in customers:
                    cid = getattr(node, "original_id", node.id)
                    d = float(self._precomputed[cid][i])
                    # adaptive policy needs full unscaled demand — it applies alpha itself
                    if node.is_split and not is_adaptive:
                        d *= node.alpha
                    demands.append(d)
                costs.append(self.recourse_policy.compute_cost(route, demands, paired_route=paired_route))
        else:
            rng = self.rng
            for _ in range(num_samples):
                demands = self._generate_demands(route, rng)
                costs.append(self.recourse_policy.compute_cost(route, demands, paired_route=paired_route))
        return costs

    def _parallel_sample(
        self, route: Route, num_samples: int, paired_route: Optional[Route] = None
    ) -> List[float]:
        """Parallelized sampling using ThreadPoolExecutor."""
        costs = []
        samples_per_thread = num_samples // self.num_threads
        remainder = num_samples % self.num_threads

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for t in range(self.num_threads):
                n = samples_per_thread + (1 if t < remainder else 0)
                thread_seed = (self.seed + t) if self.seed is not None else None
                futures.append(executor.submit(
                    self._sample_chunk, route, n, thread_seed, paired_route
                ))

            for future in as_completed(futures):
                costs.extend(future.result())
        return costs

    def _sample_chunk(
        self, route: Route, num_samples: int, seed: Optional[int],
        paired_route: Optional[Route] = None,
    ) -> List[float]:
        """Generate a chunk of samples in a single thread."""
        costs = []
        rng = np.random.default_rng(seed)
        for _ in range(num_samples):
            demands = self._generate_demands(route, rng)
            cost = self.recourse_policy.compute_cost(route, demands, paired_route=paired_route)
            costs.append(cost)
        return costs

    def _generate_demands(self, route: Route, rng: np.random.Generator) -> List[float]:
        """
        Generate one demand realization for all customers on the route.
        For non-adaptive policies: split node demands are pre-scaled by node.alpha.
        For adaptive policies: full unscaled demand is returned — the policy applies alpha itself.
        Works for any scipy demand distribution stored on the node.
        """
        from core.recourse import AdaptivePairedVehicleRecourse
        is_adaptive = isinstance(self.recourse_policy, AdaptivePairedVehicleRecourse)
        customers = [n for n in route.nodes if not n.is_depot]
        demands = []
        for node in customers:
            dist = route.instance.get_demand_distribution(node)
            demand = float(dist.rvs(random_state=rng))
            if node.is_split and not is_adaptive:
                demand = demand * node.alpha
            demands.append(demand)
        return demands

    def generate_demands(
        self, route: Route, rng: Optional[np.random.Generator] = None
    ) -> List[float]:
        """Public wrapper to generate a single demand realization."""
        rng = rng or self.rng
        return self._generate_demands(route, rng)