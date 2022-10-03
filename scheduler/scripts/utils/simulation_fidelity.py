import os
import pickle
import numpy as np

# 1
jct_jrt_diff_mean_list = []
jct_jrt_diff_std_list = []
# 2
makespan_diff_list = []
# 3
avg_jct_diff_list = []
# 4
ftf_diff_mean_list = []
ftf_diff_std_list = []
# 5
unfair_fraction_diff_list = []
# 6
cluster_util_diff_list = []


# trace_name = '50_0.2_5_100_100_342_0,0.5,0.5_0.6,0.3,0.1,0_multigpu_dynamic'
# trace_name = "30_0.2_5_100_100_302_0,0.5,0.5_0.6,0.3,0.1,0_multigpu_dynamic"

trace_list = [
    "50_0.2_5_100_100_342_0,0.5,0.5_0.6,0.3,0.1,0_multigpu_dynamic",
    # "30_0.2_5_100_100_302_0,0.5,0.5_0.6,0.3,0.1,0_multigpu_dynamic",
    "100_0.2_5_1000_50_992_0,0.5,0.5_0.6,0.3,0.09,0.01_multigpu_dynamic",
]

# policy = "shockwave"
# policy = "finish_time_fairness"
policy_list = [
    # "shockwave",
    # "finish_time_fairness",
    "max_min_fairness",
]

for trace_name in trace_list:
    for policy in policy_list:
        with open(
            os.path.join(
                "./pickle_output", f"{policy}_{trace_name}_simulation.pickle"
            ),
            "rb",
        ) as f:
            pickle_simulation = pickle.load(f)

        with open(
            os.path.join(
                "./pickle_output", f"{policy}_{trace_name}_physical.pickle"
            ),
            "rb",
        ) as f:
            pickle_physical = pickle.load(f)

        with open(
            os.path.join("./traces", f"{trace_name}.pickle"), "rb"
        ) as f:
            job_metadata = pickle.load(f)

        print("=" * 50)
        print(f"Analyzing policy {policy} on trace {trace_name}")

        print("=" * 50)
        print("(1) Mean and std of (jct_phy/jct_sim) /(jrt_phy/jrt_sim)")
        jct_list_physical = pickle_physical["jct_list"]
        jct_list_simulation = pickle_simulation["jct_list"]
        jrt_list_physical_raw = pickle_physical["job_run_time"]
        jrt_list_simulation_raw = pickle_simulation["job_run_time"]
        jrt_list_physical = []
        jrt_list_simulation = []
        for job_id in range(len(jrt_list_physical_raw)):
            jrt_list_physical.append(
                sum(jrt_list_physical_raw[job_id].values())
                / job_metadata[job_id]["scale_factor"]
            )
            jrt_list_simulation.append(
                sum(jrt_list_simulation_raw[job_id].values())
                / job_metadata[job_id]["scale_factor"]
            )

        per_job_jct_diff = [
            i / j for i, j in zip(jct_list_physical, jct_list_simulation)
        ]
        per_job_jrt_diff = [
            i / j for i, j in zip(jrt_list_physical, jrt_list_simulation)
        ]

        per_job_jct_jrt_diff = [
            i / j for i, j in zip(per_job_jct_diff, per_job_jrt_diff)
        ]
        print(
            f"mean of per_job_jct_diff / per_job_jrt_diff is {np.mean(per_job_jct_jrt_diff)}"
        )
        print(
            f"std of per_job_jct_diff / per_job_jrt_diff is {np.std(per_job_jct_jrt_diff)}"
        )
        jct_jrt_diff_mean_list.append(np.mean(per_job_jct_jrt_diff))
        jct_jrt_diff_std_list.append(np.std(per_job_jct_jrt_diff))

        print("=" * 50)
        print("(2) Makespan abs diff in percentage between phy and sim")
        makespan_physical = pickle_physical["makespan"]
        makespan_simulation = pickle_simulation["makespan"]
        print(
            f"makespan_physical / makespan_simulation is {round(100 * makespan_physical / makespan_simulation, 5)}%"
        )
        makespan_diff_list.append(
            round(100 * makespan_physical / makespan_simulation, 5)
        )

        print("=" * 50)
        print("(3) Avg_jct abs diff in percentage between phy and sim")
        avg_jct_physical = pickle_physical["avg_jct"]
        avg_jct_simulation = pickle_simulation["avg_jct"]
        print(
            f"avg_jct_physical / avg_jct_simulation is {round(100 * avg_jct_physical / avg_jct_simulation, 5)}%"
        )
        avg_jct_diff_list.append(
            round(100 * avg_jct_physical / avg_jct_simulation, 5)
        )

        print("=" * 50)
        print("(4) Mean and std of (ftf_phy/ftf_sim)")
        ftf_list_physical = pickle_physical["finish_time_fairness_list"]
        ftf_list_simulation = pickle_simulation["finish_time_fairness_list"]
        per_job_diff = [
            i / j for i, j in zip(ftf_list_physical, ftf_list_simulation)
        ]
        print(
            f"mean of ftf_list_physical / ftf_list_simulation is {np.mean(per_job_diff)}"
        )
        print(
            f"std deviation of ftf_list_physical / ftf_list_simulation is {np.std(per_job_diff)}"
        )
        ftf_diff_mean_list.append(np.mean(per_job_diff))
        ftf_diff_std_list.append(np.std(per_job_diff))

        print("=" * 50)
        print("(5) unfair_fraction abs diff in percentage between phy and sim")
        ftf_list_physical = pickle_physical["finish_time_fairness_list"]
        ftf_list_simulation = pickle_simulation["finish_time_fairness_list"]
        unfair_fraction_physical = sum(ftf > 1.05 for ftf in ftf_list_physical)
        unfair_fraction_simulation = sum(
            ftf > 1.05 for ftf in ftf_list_simulation
        )
        print(
            f"unfair_fraction_physical / unfair_fraction_simulation is {round(100 * unfair_fraction_physical / unfair_fraction_simulation, 5)}%"
        )
        unfair_fraction_diff_list.append(
            round(
                100 * unfair_fraction_physical / unfair_fraction_simulation, 5
            )
        )

        print("=" * 50)
        print("(6) cluster util abs diff in percentage between phy and sim")
        cluster_util_physical = pickle_physical["cluster_util"]
        cluster_util_simulation = pickle_simulation["cluster_util"]
        print(
            f"cluster_util_physical / cluster_util_simulation is {round(100 * cluster_util_physical / cluster_util_simulation, 5)}%"
        )
        cluster_util_diff_list.append(
            round(100 * cluster_util_physical / cluster_util_simulation, 5)
        )


print("=" * 50)
print("=" * 50)

print(f"mean of jct_jrt_diff_mean_list is {np.mean(jct_jrt_diff_mean_list)}")
print(f"mean of jct_jrt_diff_std_list is {np.mean(jct_jrt_diff_std_list)}")
print(f"mean of makespan_diff_list is {np.mean(makespan_diff_list)}")
print(f"mean of avg_jct_diff_list is {np.mean(avg_jct_diff_list)}")
print(f"mean of ftf_diff_mean_list is {np.mean(ftf_diff_mean_list)}")
print(f"mean of ftf_diff_std_list is {np.mean(ftf_diff_std_list)}")
print(
    f"mean of unfair_fraction_diff_list is {np.mean(unfair_fraction_diff_list)}"
)
print(f"mean of cluster_util_diff_list is {np.mean(cluster_util_diff_list)}")
