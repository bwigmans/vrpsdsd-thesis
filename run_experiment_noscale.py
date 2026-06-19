"""
Same experiment pipeline as run_experiment_all.py but with NO demand scaling.
Raw Solomon demands (~10-30 range) with Q=70 → demand/Q ratio ~[0.14, 0.43].
Expected: no improving splits (Lei et al. show 0% improvement below [0.51, 0.70]).
Serves as negative control for the thesis.
"""
import sys, json, os, time
sys.stdout.reconfigure(encoding="utf-8")

# Reuse everything from run_experiment_all
from run_experiment_all import (
    run_phase0, load_std_sol, greedy_splits,
    phase2_eval, INSTANCES, SEED,
    VEHICLE_CAPACITY, ALNS_ITERATIONS, ORACLE_SAMPLES,
    make_cfg,
)
from io_thesis.instance_reader import read_solomon_instance
from core.recourse import PairedVehicleRecourse
from cost.calculator import ExactCostCalculator
from cost.sample_bank import DemandSampleBank
from run_phase1 import sample_base_cost
from run_phase1_oracle import unsplit_solution

DEMAND_SCALE  = None   # no scaling — use raw Solomon demands
NUM_CUSTOMERS = 25


def main():
    results = []

    for itype, ifile in INSTANCES:
        iname = os.path.splitext(os.path.basename(ifile))[0]
        print(f"\n{'='*60}")
        print(f"Instance: {iname} ({itype}), seed={SEED}, 25c, NO SCALING")
        print(f"{'='*60}", flush=True)

        inst = read_solomon_instance(ifile, vehicle_capacity=VEHICLE_CAPACITY,
                                     demand_scale=DEMAND_SCALE, num_customers=NUM_CUSTOMERS)
        node_by_id = {n.id: n for n in inst.nodes}
        depot = node_by_id[0]
        calc = ExactCostCalculator(PairedVehicleRecourse(), cache=True)

        avg_demand = sum(n.mean_demand for n in inst.nodes if not n.is_depot) / NUM_CUSTOMERS
        print(f"  avg demand: {avg_demand:.2f}  capacity: {VEHICLE_CAPACITY}  ratio: {avg_demand/VEHICLE_CAPACITY:.3f}")

        bank_path = f"data/samples/sample_bank_{iname.lower()}_noscale.npz"
        bank = DemandSampleBank.load_or_create(bank_path, inst, op_size=250, eval_size=500, seed=SEED)
        samples = bank.post_stage1

        std_dir = f"solutions/{iname}_noscale/sampling"
        ora_dir = f"solutions/{iname}_noscale/oracle"
        os.makedirs(std_dir, exist_ok=True)
        os.makedirs(ora_dir, exist_ok=True)

        # ── Phase 0 ───────────────────────────────────────────────
        print("  Phase 0 standard...", flush=True)
        cfg_std = make_cfg(SEED, bank_path, oracle=False)
        std_data = run_phase0(inst, cfg_std, f"{std_dir}/base_solution_{SEED}.json", "std")

        print("  Phase 0 oracle...", flush=True)
        cfg_ora = make_cfg(SEED, bank_path, oracle=True)
        ora_data = run_phase0(inst, cfg_ora, f"{ora_dir}/base_solution_{SEED}.json", "oracle")

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
            "avg_demand_ratio": avg_demand / VEHICLE_CAPACITY,
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
    print("SUMMARY — No demand scaling (negative control)")
    print("         Standard vs Oracle topology (std+split vs ora+split, Phase 2)")
    print(f"{'='*80}")

    for metric in ['oracle_true', 'oracle_avg']:
        print(f"\n[{metric}]")
        print(f"  {'Inst':<6}  {'d/Q':>6}  {'std+split':>10}  {'ora+split':>10}  {'diff(ora-std)':>13}  {'win':>4}")
        print(f"  {'-'*58}")
        for r in results:
            s, o = r['std_p2'][metric], r['ora_p2'][metric]
            diff = o - s if not (math.isnan(s) or math.isnan(o)) else float('nan')
            sf = f"{s:10.3f}" if not math.isnan(s) else "       nan"
            of = f"{o:10.3f}" if not math.isnan(o) else "       nan"
            df = f"{diff:+13.3f}" if not math.isnan(diff) else "          nan"
            print(f"  {r['instance']:<6}  {r['avg_demand_ratio']:>6.3f}  {sf}  {of}  {df}  {_w(s,o)}")


if __name__ == "__main__":
    main()
