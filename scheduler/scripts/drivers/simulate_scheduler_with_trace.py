import os
import sys
import json

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    )
)

import argparse
import pickle

from job_id_pair import JobIdPair
import scheduler
import utils

root_dir = os.path.dirname(os.path.realpath(__file__))


def main(args):
    """Creates a scheduler in simulation and run a pre-defined job trace,
    then dumps the simulation statistics into a pickle file.

    Args:
        args (argparse.Namespace): Command line arguments
    """
    trace_name = os.path.splitext(os.path.basename(args.trace_file))[0]
    trace_dir = os.path.dirname(args.trace_file)

    jobs, arrival_times = utils.generate_pickle_file(
        args.trace_file, args.throughputs_file
    )
    policy = utils.get_policy(args.policy, solver=args.solver, seed=args.seed)
    pickle_filepath = os.path.join(trace_dir, f"{trace_name}.pickle")

    with open(pickle_filepath, "rb") as trace_pickle_file:
        trace_pickle = pickle.load(trace_pickle_file)
        for job_id, job in enumerate(jobs):
            # use the run time duration after dynamic adaptation for each job
            # print(f"Updating job {job_id} duration from {job.duration} to {sum(trace_pickle[job_id]['duration_every_epoch'])}")
            job.duration = sum(trace_pickle[job_id]["duration_every_epoch"])

    num_gpus = args.cluster_spec.split(":")
    cluster_spec = {
        "v100": int(num_gpus[0]),
        "p100": int(num_gpus[1]),
        "k80": int(num_gpus[2]),
    }
    num_gpus_per_server_split = args.num_gpus_per_server.split(":")
    num_gpus_per_server = {
        "v100": int(num_gpus_per_server_split[0]),
        "p100": int(num_gpus_per_server_split[1]),
        "k80": int(num_gpus_per_server_split[2]),
    }
    if args.window_start is not None and args.window_end is not None:
        jobs_to_complete = set()
        for i in range(args.window_start, args.window_end):
            jobs_to_complete.add(JobIdPair(i, None))
    else:
        jobs_to_complete = None

    if args.policy == "shockwave":
        assert (
            args.config
        ), "A .json configuration file is needed when running the shockwave policy"
        shockwave_config = json.load(open(args.config, "r"))
        # TODO: error check for missing configuration knobs
        shockwave_config["time_per_iteration"] = args.time_per_iteration
        # NOTE: In our project, we assume homogeneous hardware
        # and leave the extension to heterogeneous accelerators as future work.
        shockwave_config["num_gpus"] = (
            cluster_spec["v100"] * num_gpus_per_server["v100"]
        )
    else:
        shockwave_config = None

    # construct the scheduler
    sched = scheduler.Scheduler(
        policy,
        throughputs_file=args.throughputs_file,
        simulate=True,
        seed=args.seed,
        time_per_iteration=args.time_per_iteration,
        pickle_file=pickle_filepath,
        shockwave_config=shockwave_config,
    )

    # run the trace through the scheduler in simulation
    makespan = sched.simulate(
        cluster_spec,
        arrival_times,
        jobs,
        debug=args.debug,
        checkpoint_threshold=args.checkpoint_threshold,
        checkpoint_file=args.checkpoint_file,
        num_gpus_per_server=num_gpus_per_server,
        jobs_to_complete=jobs_to_complete,
    )

    # retrieve the experiment statistics
    (
        avg_jct,
        geometric_mean_jct,
        harmonic_mean_jct,
        jct_list,
    ) = sched.get_average_jct(job_ids=jobs_to_complete)
    cluster_util, utilization_list = sched.get_cluster_utilization()
    (
        extension_percentage,
        num_lease_extensions,
        num_lease_extension_opportunities,
    ) = sched.get_num_lease_extensions()
    per_round_schedule = sched.get_per_round_schedule()
    (
        finish_time_fairness_list,
        finish_time_fairness_themis_list,
    ) = sched.get_finish_time_fairness(
        job_ids=jobs_to_complete, pickle_file_name=f"{args.trace_file}.pickle"
    )
    # sched.get_completed_steps(jobs_to_complete)
    envy_ratios, envy_list = sched.get_envy_list()
    throughput_timeline = sched.get_throughput_timeline()
    job_run_time = sched.get_job_run_time()

    sched.shutdown()

    # dump the simulation statistics into a pickle file for future plotting
    pickle_object = {
        "trace_file": args.trace_file,
        "policy": args.policy,
        "makespan": makespan,
        "avg_jct": avg_jct,
        "jct_list": jct_list,
        "finish_time_fairness_list": finish_time_fairness_list,
        "finish_time_fairness_themis_list": finish_time_fairness_themis_list,
        # finish_time_fairness_isolated_list is added in plotting.py
        "cluster_util": cluster_util,
        "utilization_list": utilization_list,
        "envy_list": envy_list,
        "envy_ratios": envy_ratios,
        "extension_percentage": extension_percentage,
        "num_lease_extensions": num_lease_extensions,
        "num_lease_extension_opportunities": num_lease_extension_opportunities,
        "per_round_schedule": per_round_schedule,
        "time_per_iteration": args.time_per_iteration,
        "throughput_timeline": throughput_timeline,
        "job_run_time": job_run_time,
        "geometric_mean_jct": geometric_mean_jct,
        "harmonic_mean_jct": harmonic_mean_jct,
    }

    if not os.path.isdir(os.path.join(root_dir, args.pickle_output_dir)):
        os.mkdir(os.path.join(root_dir, args.pickle_output_dir))
    with open(
        os.path.join(
            root_dir,
            args.pickle_output_dir,
            f"{args.policy}_{trace_name}_simulation.pickle",
        ),
        "wb",
    ) as f:
        pickle.dump(pickle_object, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run scheduler with trace in simulation"
    )
    parser.add_argument(
        "-t", "--trace_file", type=str, required=True, help="Trace file"
    )
    parser.add_argument(
        "-p",
        "--policy",
        type=str,
        default="fifo",
        choices=utils.get_available_policies(),
        help="Scheduler policy",
    )
    parser.add_argument(
        "--throughputs_file",
        type=str,
        required=True,
        help="Oracle throughputs file",
    )
    parser.add_argument(
        "-c",
        "--cluster_spec",
        type=str,
        default="16:0:0",
        help=("Cluster specification in the form of " "#v100s:#p100s:#k80s"),
    )
    parser.add_argument(
        "--num_gpus_per_server",
        type=str,
        default="1:1:1",
        help=("Cluster specification in the form of " "#v100s:#p100s:#k80s"),
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--solver",
        type=str,
        choices=["ECOS", "GUROBI", "SCS"],
        default="ECOS",
        help="CVXPY solver",
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", default=False, help="Debug"
    )
    parser.add_argument(
        "--checkpoint_threshold",
        type=int,
        default=None,
        help="Create checkpoint when this job ID comes in",
    )
    parser.add_argument(
        "--checkpoint_file",
        default=None,
        help=("Load checkpoint located at passed in" "checkpoint_file"),
    )
    parser.add_argument(
        "--time_per_iteration",
        type=int,
        default=120,
        help="Time per iteration in seconds",
    )
    parser.add_argument(
        "-s",
        "--window-start",
        type=int,
        default=None,
        help="measurement window start (job id)",
    )
    parser.add_argument(
        "-e",
        "--window-end",
        type=int,
        default=None,
        help="Measurement window end (job ID)",
    )
    parser.add_argument(
        "--pickle_output_dir",
        type=str,
        default="../../pickle_output_nsdi23",
        help="Path of the directory in which summary files of experiments (*.pickle) are stored",
    )
    # shockwave additions
    parser.add_argument(
        "--config",
        type=str,
        help=".json configuration file that holds the shockwave hyperparameters",
    )
    args = parser.parse_args()

    if args.policy == "finish_time_fairness":
        args.solver = "GUROBI"  # XXX: Is it fair for other policies?

    main(args)
