import os
import pickle
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)
from job_id_pair import JobIdPair  # required to load the pickles

policies = [
    "finish_time_fairness",
    "max_min_fairness",
    "min_total_duration",
    "allox",
    "max_sum_throughput_perf",
    "shockwave",
]

trace_name = "120_0.2_5_100_40_25_0,0.5,0.5_0.6,0.3,0.09,0.01_multigpu_dynamic"

for policy in policies:
    physical_pickle = (
        f"pickles/tacc_32gpus_comparison/{policy}_{trace_name}_physical.pickle"
    )
    simulation_pickle = f"pickles/tacc_32gpus_comparison/{policy}_{trace_name}_simulation.pickle"

    with open(physical_pickle, "rb") as f:
        physical_pickle = pickle.load(f)
    with open(simulation_pickle, "rb") as f:
        simulation_pickle = pickle.load(f)

    # makespan
    makespan = [physical_pickle["makespan"], simulation_pickle["makespan"]]
    # NOTE: we came across daylight savings (nov 2021) when running finish time fairness :)
    if policy == "finish_time_fairness":
        makespan[0] += 3600
    print(
        f"{policy} makespan difference: {round(abs(100 - 100 * makespan[0] / makespan[1]), 2)}%"
    )

    # avg jct
    avg_jct = [physical_pickle["avg_jct"], simulation_pickle["avg_jct"]]
    print(
        f"{policy} avg_jct difference: {round(abs(100 - 100 * avg_jct[0] / avg_jct[1]), 2)}%"
    )

    # unfair fraction
    unfair_physical = sum(
        [1 for x in physical_pickle["finish_time_fairness_list"] if x > 1.1]
    )
    unfair_simulation = sum(
        [1 for x in simulation_pickle["finish_time_fairness_list"] if x > 1.1]
    )
    unfair_jobs_diff = abs(unfair_physical - unfair_simulation)
    print(
        f"{policy} unfair fraction difference: {round(unfair_jobs_diff / 120 * 100, 2)}%"
    )  # 120 jobs in the trace
