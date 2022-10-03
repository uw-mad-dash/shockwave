import os
import pickle
import numpy as np

policy = "finish_time_fairness"
# policy='shockwave'

trace_name = "50_0.2_5_100_100_41_multigpu_dynamic"

with open(
    os.path.join(
        "./pickle_output", f"{policy}_{trace_name}_simulation.pickle"
    ),
    "rb",
) as f:
    pickle_simulation = pickle.load(f)

with open(
    os.path.join("./pickle_output", f"{policy}_{trace_name}_physical.pickle"),
    "rb",
) as f:
    pickle_physical = pickle.load(f)

jct_list_sim = pickle_simulation["jct_list"]
jct_list_physical = pickle_physical["jct_list"]
jrt_list_sim = pickle_simulation["job_run_time"]
jrt_list_physical = pickle_physical["job_run_time"]

print(pickle_physical["makespan"])

jrt_diff = []

for ijob, jctsim in enumerate(jct_list_sim):
    jctphsical = jct_list_physical[ijob]
    jrtsim = sum(jrt_list_sim[ijob].values())
    jrtphysical = sum(jrt_list_physical[ijob].values())
    print(f"Job {ijob}")
    print(f"JCT Ratio: {jctphsical/jctsim}={jctphsical}/{jctsim}")
    print(f"JRT Ratio: {jrtphysical/jrtsim}={jrtphysical}/{jrtsim}")
    jrt_diff.append(jrtphysical / jrtsim)
    print("-" * 50)

print(np.mean(jrt_diff))
