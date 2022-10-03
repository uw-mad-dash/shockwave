import argparse
import os
import pickle
import sys
from pprint import pprint

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)
from job_id_pair import JobIdPair  # required to load the pickles

policies = [
    "shockwave",
    "min_total_duration",
    "finish_time_fairness",
    "max_min_fairness",
    "allox",
    "max_sum_throughput_perf",
    "gandiva_fair",
]

# TODO: due to the noisy nature of throughputs and how we
# use them to calculate the finish time fairness, we regard
# jobs with ftf < 1.05 as receiving fair allocations
FTF_FAIRNESS_THRESHOLD = 1.05


def main(args):
    makespans = {p: None for p in policies}
    avg_jcts = {p: None for p in policies}
    worst_ftf = {p: None for p in policies}
    unfair_fractions = {p: None for p in policies}

    for policy in policies:
        with open(
            os.path.join(args.dir, f"{policy}_{args.trace_name}.pickle"), "rb"
        ) as f:
            result = pickle.load(f)
            makespans[policy] = result["makespan"]
            avg_jcts[policy] = result["avg_jct"]
            ftf_list = result["finish_time_fairness_list"]
            worst_ftf[policy] = max(ftf_list)
            unfair_fractions[policy] = round(
                sum(ftf > FTF_FAIRNESS_THRESHOLD for ftf in ftf_list)
                / len(ftf_list)
                * 100,
                2,
            )

    for metric in [makespans, avg_jcts, worst_ftf, unfair_fractions]:
        print_relative_value(metric)


def print_relative_value(metrics):
    pprint(metrics)

    baseline = metrics["shockwave"]
    for policy, value in metrics.items():
        metrics[policy] = round(value / baseline, 2)

    pprint(metrics)
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate and analyze experiment results of running multiple policies over the same trace"
    )
    parser.add_argument(
        "--dir", type=str, default="./", help="Directory of the pickle files"
    )
    parser.add_argument(
        "--trace-name",
        type=str,
        help="File name of the trace",
    )
    parser.add_argument(
        "--is-physical",
        action="store_true",
        default=False,
        help="Set to True if the pickle files are produced by a physical run",
    )

    args = parser.parse_args()
    args.trace_name += "_physical" if args.is_physical else "_simulation"

    main(args)
