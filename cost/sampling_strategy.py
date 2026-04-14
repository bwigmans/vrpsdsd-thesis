from abc import ABC, abstractmethod
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy.stats import poisson

from core.route import Route
from core.recourse import RecoursePolicy, PairedVehicleRecourse


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

    def sample(self, route: Route, num_samples: Optional[int] = None) -> List[float]:
        """Perform Monte Carlo sampling of recourse costs."""
        ns = num_samples or self.num_samples
        if self.parallel:
            return self._parallel_sample(route, ns)
        else:
            return self._sequential_sample(route, ns)

    def _sequential_sample(self, route: Route, num_samples: int) -> List[float]:
        """Sequential sampling (single thread)."""
        costs = []
        rng = np.random.default_rng(self.seed)
        for _ in range(num_samples):
            demands = self._generate_demands(route, rng)
            cost = self.recourse_policy.compute_cost(route, demands)
            costs.append(cost)
        return costs

    def _parallel_sample(self, route: Route, num_samples: int) -> List[float]:
        """Parallelized sampling using ThreadPoolExecutor."""
        costs = []
        # Split samples among threads
        samples_per_thread = num_samples // self.num_threads
        remainder = num_samples % self.num_threads

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for t in range(self.num_threads):
                n = samples_per_thread + (1 if t < remainder else 0)
                # Different seed per thread to avoid collisions
                thread_seed = (self.seed + t) if self.seed is not None else None
                futures.append(executor.submit(
                    self._sample_chunk, route, n, thread_seed
                ))

            for future in as_completed(futures):
                costs.extend(future.result())
        return costs

    def _sample_chunk(self, route: Route, num_samples: int, seed: Optional[int]) -> List[float]:
        """Generate a chunk of samples in a single thread."""
        costs = []
        rng = np.random.default_rng(seed)
        for _ in range(num_samples):
            demands = self._generate_demands(route, rng)
            cost = self.recourse_policy.compute_cost(route, demands)
            costs.append(cost)
        return costs

    def _generate_demands(self, route: Route, rng: np.random.Generator) -> List[float]:
        """
        Generate one demand realization for all customers on the route.
        Returns list of total demands (not split‑fraction) in customer order.
        """
        customers = [n for n in route.nodes if not n.is_depot]
        demands = []
        for node in customers:
            # Poisson demand with lambda = node.mean_demand
            lam = route.instance.get_expected_demand(node)
            # Use rng.poisson for reproducibility
            demand = rng.poisson(lam)
            demands.append(float(demand))
        return demands