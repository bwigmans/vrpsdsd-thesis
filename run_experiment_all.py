"""
Full experiment pipeline for all Lei et al. (2012) 25-customer [0.51,0.70] instances.
Instances: R (R101), C1 (C101), C2 (C201), RC (RC101) — one seed each.

For each instance:
  1. Phase 0 standard ALNS
  2. Phase 0 oracle ALNS
  3. Phase 1 on standard topology → best split candidate
  4. Phase 1 on oracle topology (unsplit first) → best split candidate
  5. Phase 2 evaluation: oracle_true, oracle_avg, equalize_slack, marginal_cost

Prints a summary table comparing standard vs oracle topologies across all instances.
"""
import sys, json, os, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

from io_thesis.instance_reader import read_solomon_instance
from core.instance import Node
from core.route import Route
from core.solution import Solution
from core.recourse import PairedVehicleRecourse, AdaptivePairedVehicleRecourse
from cost.calculator import ExactCostCalculator
from cost.sample_bank import DemandSampleBank
from algorithms.alns import ALNSSolver
from algorithms.alpha_policies import make_recourse_alpha_policy
from utils import Configuration
from run_phase1 import search_phase1, sample_base_cost, build_split_routes, ALPHA_GRID
from run_phase1_oracle import unsplit_solution

SEED             = 42
VEHICLE_CAPACITY = 70.0
DEMAND_SCALE     = (0.51, 0.70)
NUM_CUSTOMERS    = 25
ALNS_ITERATIONS  = 1000

INSTANCES = [
    ("R",  "data/R101.txt"),
    ("C1", "data/C101.txt"),
    ("C2", "data/C201.txt"),
    ("RC", "data/RC101.txt"),
]

ORACLE_SAMPLES = 500


def make_cfg(seed, bank_path, oracle=False):
    cfg = Configuration(
        vehicle_capacity=VEHICLE_CAPACITY,
        alns_iterations=ALNS_ITERATIONS,
        alns_segment_length=50,
        seed=seed, verbose=False,
        alpha_policy='lei', alpha_reoptimize=False,
        use_ec_operators=True, cost_method='sampling',
        use_oracle_splits=oracle,
    )
    cfg.operator_num_samples = 250
    cfg.evaluation_num_samples = 500
    cfg.sample_bank_path = bank_path
    return cfg


def run_phase0(inst, cfg, out_path, label):
    if os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)
    t0 = time.perf_counter()
    solver = ALNSSolver(inst, cfg)
    sol = solver.solve()
    elapsed = time.perf_counter() - t0
    exact_calc = ExactCostCalculator(PairedVehicleRecourse())
    exact_cost = sol.get_total_cost(exact_calc)
    routes = [[n.id for n in r.nodes if not n.is_depot] for r in sol.routes]
    split_customers = sorted(set(
        getattr(n,'original_id',n.id)
        for r in sol.routes for n in r.nodes
        if not n.is_depot and n.is_split
    ))
    data = {"seed": SEED, "exact_cost": exact_cost, "routes": routes,
            "split_customers": split_customers}
    if cfg.use_oracle_splits:
        paired_idx = {}
        for r1, r2 in sol.paired_routes.items():
            i1, i2 = sol.routes.index(r1), sol.routes.index(r2)
            if str(i1) not in paired_idx:
                paired_idx[str(i1)] = i2
        data["paired_indices"] = paired_idx
        data["split_node_map"] = {str(n.id): getattr(n,'original_id',n.id)
                                   for r in sol.routes for n in r.nodes if n.is_split}
        data["split_alpha_map"] = {str(n.id): float(n.alpha)
                                   for r in sol.routes for n in r.nodes if n.is_split}
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"    {label}: exact={exact_cost:.3f}  routes={len(routes)}"
          f"  splits={split_customers}  t={elapsed:.0f}s", flush=True)
    return data


def load_std_sol(data, node_by_id, depot, inst):
    routes = []
    for cids in data["routes"]:
        nodes = [depot] + [node_by_id[c] for c in cids] + [depot]
        routes.append(Route(nodes, inst))
    return Solution(routes)


def apply_split(sol, cid, r1_idx, r2_idx, calc):
    for alpha in ALPHA_GRID:
        _, tr1, tr2 = build_split_routes(sol, cid, r1_idx, r2_idx, alpha, calc)
        if tr1 is not None:
            split_sol = sol.copy()
            split_sol.routes[r1_idx] = tr1
            split_sol.routes[r2_idx] = tr2
            split_sol.paired_routes[tr1] = tr2
            split_sol.paired_routes[tr2] = tr1
            return split_sol
    raise RuntimeError(f"Could not apply split for c{cid} r{r1_idx}->r{r2_idx} at any alpha")


def greedy_splits(sol, calc, samples, label=""):
    """Greedily add improving splits until none remain. One split max per route."""
    current = sol
    splits_added = []
    while True:
        calc.invalidate_cache()
        _, table = search_phase1(current, calc, samples)
        if not table:
            break
        best = table[0]
        cid, r1, r2 = best["customer"], best["r1"], best["r2"]
        current = apply_split(current, cid, r1, r2, calc)
        splits_added.append(f"c{cid} r{r1}->r{r2} oracle={best['oracle']:+.4f}")
        print(f"    {label}split: c{cid} r{r1}->r{r2} oracle={best['oracle']:+.4f}", flush=True)
    if not splits_added:
        print(f"    {label}no improving split found")
    return current


def phase2_eval(sol, samples):
    if sol is None:
        return {m: float('nan') for m in ['oracle_true','oracle_avg','eq_slack','marg_cost']}
    ot = AdaptivePairedVehicleRecourse(oracle_mode='oracle_true')
    oa = AdaptivePairedVehicleRecourse(oracle_mode='oracle_avg', oracle_samples=ORACLE_SAMPLES)
    eq = AdaptivePairedVehicleRecourse(make_recourse_alpha_policy('equalize_slack'))
    mc = AdaptivePairedVehicleRecourse(make_recourse_alpha_policy('marginal_cost'))
    return {
        'oracle_true': sol.get_total_cost_adaptive(ot, samples),
        'oracle_avg':  sol.get_total_cost_adaptive(oa, samples),
        'eq_slack':    sol.get_total_cost_adaptive(eq, samples),
        'marg_cost':   sol.get_total_cost_adaptive(mc, samples),
    }


def main():
    results = []

    for itype, ifile in INSTANCES:
        iname = os.path.splitext(os.path.basename(ifile))[0]
        print(f"\n{'='*60}")
        print(f"Instance: {iname} ({itype}), seed={SEED}, 25c, [0.51,0.70]")
        print(f"{'='*60}", flush=True)

        inst = read_solomon_instance(ifile, vehicle_capacity=VEHICLE_CAPACITY,
                                     demand_scale=DEMAND_SCALE, num_customers=NUM_CUSTOMERS)
        node_by_id = {n.id: n for n in inst.nodes}
        depot = node_by_id[0]
        calc = ExactCostCalculator(PairedVehicleRecourse(), cache=True)

        bank_path = f"data/samples/sample_bank_{iname.lower()}.npz"
        bank = DemandSampleBank.load_or_create(bank_path, inst, op_size=250, eval_size=500, seed=SEED)
        samples = bank.post_stage1

        os.makedirs(f"solutions/{iname}/sampling", exist_ok=True)
        os.makedirs(f"solutions/{iname}/oracle", exist_ok=True)

        # ── Phase 0 ───────────────────────────────────────────────
        print("  Phase 0 standard...", flush=True)
        std_data = run_phase0(inst, make_cfg(SEED, bank_path, oracle=False),
                              f"solutions/{iname}/sampling/base_solution_{SEED}.json", "std")

        print("  Phase 0 oracle...", flush=True)
        ora_data = run_phase0(inst, make_cfg(SEED, bank_path, oracle=True),
                              f"solutions/{iname}/oracle/base_solution_{SEED}.json", "oracle")

        # ── Phase 1 + greedy splits ───────────────────────────────
        print("  Phase 1 standard...", flush=True)
        std_sol = load_std_sol(std_data, node_by_id, depot, inst)
        std_base = sample_base_cost(std_sol, samples)
        std_split_sol = greedy_splits(std_sol, calc, samples, label="std ")

        print("  Phase 1 oracle topology...", flush=True)
        ora_sol = unsplit_solution(ora_data, inst, node_by_id, depot, calc)
        ora_base = sample_base_cost(ora_sol, samples)
        ora_split_sol = greedy_splits(ora_sol, calc, samples, label="ora ")

        # ── Phase 2 ───────────────────────────────────────────────
        print("  Phase 2 evaluation...", flush=True)
        std_p2 = phase2_eval(std_split_sol, samples)
        ora_p2 = phase2_eval(ora_split_sol, samples)

        results.append({
            "instance": iname, "type": itype,
            "std_base": std_base, "ora_base": ora_base,
            "std_p2": std_p2, "ora_p2": ora_p2,
        })

    # ── Summary ───────────────────────────────────────────────────
    import math

    def _w(s, o):
        if math.isnan(s) and math.isnan(o): return "   —"
        if math.isnan(s): return " ora"
        if math.isnan(o): return " std"
        return " std" if s <= o else " ora"

    print(f"\n{'='*80}")
    print("SUMMARY — Standard vs Oracle topology (std+split vs ora+split, Phase 2)")
    print(f"{'='*80}")

    for metric in ['oracle_true', 'oracle_avg', 'eq_slack', 'marg_cost']:
        print(f"\n[{metric}]")
        print(f"  {'Inst':<6}  {'std+split':>10}  {'ora+split':>10}  {'diff(ora-std)':>13}  {'win':>4}")
        print(f"  {'-'*50}")
        for r in results:
            s, o = r['std_p2'][metric], r['ora_p2'][metric]
            diff = o - s if not (math.isnan(s) or math.isnan(o)) else float('nan')
            sf = f"{s:10.3f}" if not math.isnan(s) else "       nan"
            of = f"{o:10.3f}" if not math.isnan(o) else "       nan"
            df = f"{diff:+13.3f}" if not math.isnan(diff) else "          nan"
            print(f"  {r['instance']:<6}  {sf}  {of}  {df}  {_w(s,o)}")


if __name__ == "__main__":
    main()
