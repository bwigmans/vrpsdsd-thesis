"""
Precomputes all demand samples upfront for the entire ALNS run.
Slices are handed out per stage so dist.rvs() is never called during search.

Layout per customer (total=50k), parameterised by op_size and eval_size:
  [0                        : 20*op_size ) — operator pool  (20 segments x op_size)
  [20*op_size               : post1_start) — eval pool      (up to 60 events x eval_size)
  [post1_start              : post1_start+1000) — post stage 1 (1000 fixed)
  [post1_start+1000         : post1_start+6000) — post stage 2 (5000 fixed)
"""
import os
import time
import numpy as np
from typing import Dict, Optional

TOTAL_SAMPLES = 50_000
_NUM_SEGMENTS = 20
_EVAL_MAX     = 60
_POST1_SIZE   = 1_000
_POST2_SIZE   = 5_000


class DemandSampleBank:
    """
    Holds TOTAL_SAMPLES pre-drawn demand values per customer.
    op_size and eval_size are wired from config at construction time.
    """

    def __init__(
        self,
        instance,
        op_size: int = 500,
        eval_size: int = 500,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        self.op_size   = op_size
        self.eval_size = eval_size
        self._op_offset   = 0
        self._eval_offset = _NUM_SEGMENTS * op_size
        self._post1_offset = self._eval_offset + _EVAL_MAX * eval_size
        self._post2_offset = self._post1_offset + _POST1_SIZE

        needed = self._post2_offset + _POST2_SIZE
        total  = max(TOTAL_SAMPLES, needed)

        rng = np.random.default_rng(seed)
        customers = [n for n in instance.nodes if not n.is_depot]

        if verbose:
            print(f"  [sample-bank] precomputing {total:,} samples x {len(customers)} customers "
                  f"(op={op_size}, eval={eval_size})...", flush=True)
        t0 = time.perf_counter()

        self._bank: Dict[int, np.ndarray] = {}
        for node in customers:
            dist = instance.get_demand_distribution(node)
            self._bank[node.id] = dist.rvs(size=total, random_state=rng).astype(np.float64)

        if verbose:
            print(f"  [sample-bank] done in {time.perf_counter() - t0:.1f}s", flush=True)

        self._eval_counter = 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            _meta_op_size=np.array([self.op_size]),
            _meta_eval_size=np.array([self.eval_size]),
            **{str(cid): arr for cid, arr in self._bank.items()},
        )
        print(f"  [sample-bank] saved to {path}", flush=True)

    @classmethod
    def load(cls, path: str, verbose: bool = True) -> "DemandSampleBank":
        obj = cls.__new__(cls)
        data = np.load(path)
        op_size   = int(data["_meta_op_size"][0])
        eval_size = int(data["_meta_eval_size"][0])
        obj.op_size   = op_size
        obj.eval_size = eval_size
        obj._op_offset    = 0
        obj._eval_offset  = _NUM_SEGMENTS * op_size
        obj._post1_offset = obj._eval_offset + _EVAL_MAX * eval_size
        obj._post2_offset = obj._post1_offset + _POST1_SIZE
        obj._bank = {
            int(k): data[k].astype(np.float64)
            for k in data.files if not k.startswith("_meta")
        }
        obj._eval_counter = 0
        if verbose:
            n = next(iter(obj._bank.values())).shape[0]
            print(f"  [sample-bank] loaded {n:,} samples x {len(obj._bank)} customers "
                  f"(op={op_size}, eval={eval_size}) from {path}", flush=True)
        return obj

    @classmethod
    def load_or_create(
        cls,
        path: str,
        instance,
        op_size: int = 500,
        eval_size: int = 500,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> "DemandSampleBank":
        if os.path.exists(path):
            bank = cls.load(path, verbose=verbose)
            if bank.op_size == op_size and bank.eval_size == eval_size:
                return bank
            if verbose:
                print(f"  [sample-bank] config mismatch (op={bank.op_size}→{op_size}, "
                      f"eval={bank.eval_size}→{eval_size}), regenerating...", flush=True)
        bank = cls(instance, op_size=op_size, eval_size=eval_size, seed=seed, verbose=verbose)
        bank.save(path)
        return bank

    # ------------------------------------------------------------------
    # Slices
    # ------------------------------------------------------------------

    def operator_slice(self, segment_idx: int) -> Dict[int, np.ndarray]:
        start = self._op_offset + (segment_idx % _NUM_SEGMENTS) * self.op_size
        return {cid: arr[start: start + self.op_size] for cid, arr in self._bank.items()}

    def eval_slice(self) -> Dict[int, np.ndarray]:
        idx = self._eval_counter % _EVAL_MAX
        self._eval_counter += 1
        start = self._eval_offset + idx * self.eval_size
        return {cid: arr[start: start + self.eval_size] for cid, arr in self._bank.items()}

    @property
    def post_stage1(self) -> Dict[int, np.ndarray]:
        return {cid: arr[self._post1_offset: self._post1_offset + _POST1_SIZE]
                for cid, arr in self._bank.items()}

    @property
    def post_stage2(self) -> Dict[int, np.ndarray]:
        return {cid: arr[self._post2_offset: self._post2_offset + _POST2_SIZE]
                for cid, arr in self._bank.items()}
