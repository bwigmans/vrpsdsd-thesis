"""
Precomputes all demand samples upfront for the entire ALNS run.
Slices are handed out per stage so dist.rvs() is never called during search.

Layout per customer (total=50k):
  [0      : 1000 ) — operator pool  (20 segments x 50)
  [1000   : 41000) — eval pool      (up to 80 new-best events x 500)
  [41000  : 42000) — post stage 1   (1000 fixed)
  [42000  : 47000) — post stage 2   (5000 fixed)
"""
import os
import time
import numpy as np
from typing import Dict, Optional


TOTAL_SAMPLES = 50_000

_OP_OFFSET   = 0
_OP_SIZE     = 50       # per segment
_EVAL_OFFSET = 1_000
_EVAL_SIZE   = 500      # per new-best event
_EVAL_MAX    = 80       # max new-best events
_POST1_OFFSET = 41_000
_POST1_SIZE   = 1_000
_POST2_OFFSET = 42_000
_POST2_SIZE   = 5_000


class DemandSampleBank:
    """
    Holds TOTAL_SAMPLES pre-drawn demand values per customer.
    All slices are views (no copy) into the underlying array.
    """

    def __init__(self, instance, seed: Optional[int] = None, verbose: bool = True):
        rng = np.random.default_rng(seed)
        customers = [n for n in instance.nodes if not n.is_depot]

        if verbose:
            print(f"  [sample-bank] precomputing {TOTAL_SAMPLES:,} samples x {len(customers)} customers...", flush=True)
        t0 = time.perf_counter()

        self._bank: Dict[int, np.ndarray] = {}
        for node in customers:
            dist = instance.get_demand_distribution(node)
            self._bank[node.id] = dist.rvs(size=TOTAL_SAMPLES, random_state=rng).astype(np.float64)

        if verbose:
            elapsed = time.perf_counter() - t0
            print(f"  [sample-bank] done in {elapsed:.1f}s", flush=True)

        self._eval_counter = 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save bank to a .npz file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, **{str(cid): arr for cid, arr in self._bank.items()})
        print(f"  [sample-bank] saved to {path}", flush=True)

    @classmethod
    def load(cls, path: str, verbose: bool = True) -> "DemandSampleBank":
        """Load a previously saved bank from a .npz file."""
        obj = cls.__new__(cls)
        data = np.load(path)
        obj._bank = {int(k): data[k].astype(np.float64) for k in data.files}
        obj._eval_counter = 0
        if verbose:
            n_customers = len(obj._bank)
            n_samples = next(iter(obj._bank.values())).shape[0]
            print(f"  [sample-bank] loaded {n_samples:,} samples x {n_customers} customers from {path}", flush=True)
        return obj

    @classmethod
    def load_or_create(cls, path: str, instance, seed: Optional[int] = None, verbose: bool = True) -> "DemandSampleBank":
        """Load from disk if available, otherwise generate and save."""
        if os.path.exists(path):
            return cls.load(path, verbose=verbose)
        bank = cls(instance, seed=seed, verbose=verbose)
        bank.save(path)
        return bank

    def operator_slice(self, segment_idx: int) -> Dict[int, np.ndarray]:
        """50 samples for segment `segment_idx` (wraps if > 20 segments)."""
        start = _OP_OFFSET + (segment_idx % 20) * _OP_SIZE
        return {cid: arr[start: start + _OP_SIZE] for cid, arr in self._bank.items()}

    def eval_slice(self) -> Dict[int, np.ndarray]:
        """500 samples for the next new-best eval event."""
        idx = self._eval_counter % _EVAL_MAX
        self._eval_counter += 1
        start = _EVAL_OFFSET + idx * _EVAL_SIZE
        return {cid: arr[start: start + _EVAL_SIZE] for cid, arr in self._bank.items()}

    @property
    def post_stage1(self) -> Dict[int, np.ndarray]:
        """1000 fixed samples for post-processing split search."""
        return {cid: arr[_POST1_OFFSET: _POST1_OFFSET + _POST1_SIZE] for cid, arr in self._bank.items()}

    @property
    def post_stage2(self) -> Dict[int, np.ndarray]:
        """5000 fixed samples for final solution quality evaluation."""
        return {cid: arr[_POST2_OFFSET: _POST2_OFFSET + _POST2_SIZE] for cid, arr in self._bank.items()}
