import os
import pickle
import numpy as np

# This script dissects a run

# policy='finish_time_fairness'
policy = "shockwave"

# trace_name = '50_0.2_5_100_100_342_0,0.5,0.5_0.6,0.3,0.1,0_multigpu_dynamic'
trace_name = "30_0.2_5_100_100_63_0,0.5,0.5_0.6,0.3,0.1,0_multigpu_dynamic"

with open(
    os.path.join(
        "./pickle_output", f"{policy}_{trace_name}_simulation.pickle"
    ),
    "rb",
) as f:
    # with open(os.path.join('./pickle_output', f'{"shockwave"}_{trace_name}_physical.pickle'),'rb') as f:
    pickle_simulation = pickle.load(f)

with open(
    os.path.join("./pickle_output", f"{policy}_{trace_name}_physical.pickle"),
    "rb",
) as f:
    # with open(os.path.join('./pickle_output', f'{"finish_time_fairness"}_{trace_name}_physical.pickle'),'rb') as f:
    pickle_physical = pickle.load(f)

with open(os.path.join("./traces", f"{trace_name}.pickle"), "rb") as f:
    job_metadata = pickle.load(f)

jct_list_sim = pickle_simulation["jct_list"]
jct_list_physical = pickle_physical["jct_list"]
jrt_list_sim = pickle_simulation["job_run_time"]
jrt_list_physical = pickle_physical["job_run_time"]

# print(pickle_physical['makespan'])

jrt_diff = []
jct_diff = []

for ijob, jctsim in enumerate(jct_list_sim):
    jctphsical = jct_list_physical[ijob]
    jrtsim = (
        sum(jrt_list_sim[ijob].values()) / job_metadata[ijob]["scale_factor"]
    )
    jrtphysical = (
        sum(jrt_list_physical[ijob].values())
        / job_metadata[ijob]["scale_factor"]
    )
    print(f"Job {ijob}")
    # print(f"simulation jrt: {jrt_list_sim[ijob]}")
    print(f"JCT Ratio: {jctphsical/jctsim}={jctphsical}/{jctsim}")
    print(f"JRT Ratio: {jrtphysical/jrtsim}={jrtphysical}/{jrtsim}")
    print(f"Diff between JCT and JRT in simulation: {jctphsical/jctsim}")
    print(f"Diff between JCT and JRT in physical: {jrtphysical/jrtsim}")
    jrt_diff.append(abs(1 - jrtphysical / jrtsim))
    jct_diff.append(abs(1 - jctphsical / jctsim))
    print("-" * 50)

print(f"mean of jrt diff: {np.mean(jrt_diff)}")
print(f"max of jrt diff: {np.max(jrt_diff)}")
print(f"min of jrt diff: {np.min(jrt_diff)}")

print(f"mean of jct diff: {np.mean(jct_diff)}")
print(f"max of jct diff: {np.max(jct_diff)}")
print(f"min of jct diff: {np.min(jct_diff)}")
