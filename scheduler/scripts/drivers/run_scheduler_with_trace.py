import os, sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    )
)

import argparse
import datetime
import json
import pickle
import queue
import time
import datetime

import job
from job_id_pair import JobIdPair
import policies
import scheduler
import utils

root_dir = os.path.dirname(
    os.path.dirname(os.path.split(os.path.abspath(__file__))[0])
)

SLEEP_TIME = 10

# TODO: add shockwave config setup
# TODO: set up pickle output

# pickle_output_dir = "simulator_logs"
pickle_output_dir = "pickle_output"

if not os.path.isdir(os.path.join(root_dir, pickle_output_dir)):
    os.mkdir(os.path.join(root_dir, pickle_output_dir))


def main(args):
    trace_name = os.path.splitext(os.path.basename(args.trace_file))[0]
    trace_dir = os.path.dirname(args.trace_file)

    # Set up jobs.
    jobs_to_complete = set()
    # jobs, arrival_times = utils.generate_pickle_file(args.trace_file, args.throughputs_file, args.mode)
    jobs, arrival_times = utils.generate_pickle_file(
        args.trace_file, args.throughputs_file
    )
    if args.window_start is not None and args.window_end is not None:
        for i in range(args.window_start, args.window_end):
            jobs_to_complete.add(JobIdPair(i, None))
    else:
        for i in range(len(jobs)):
            jobs_to_complete.add(JobIdPair(i, None))
    job_queue = queue.Queue()
    for (job, arrival_time) in zip(jobs, arrival_times):
        job_queue.put((job, arrival_time))

    # Instantiate scheduler.
    trace_name = os.path.splitext(os.path.basename(args.trace_file))[0]
    policy = utils.get_policy(args.policy, solver=args.solver, seed=args.seed)
    pickle_filepath = os.path.join(trace_dir, f"{trace_name}.pickle")

    # read in shockwave json config file
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

    sched = scheduler.Scheduler(
        policy,
        seed=args.seed,
        throughputs_file=args.throughputs_file,
        time_per_iteration=args.time_per_iteration,
        expected_num_workers=args.expected_num_workers,
        max_rounds=args.max_rounds,
        pickle_file=pickle_filepath,
    )

    with open(pickle_filepath, "rb") as trace_pickle_file:
        trace_pickle = pickle.load(trace_pickle_file)
        job_id = 0
        try:
            # Submit jobs to the scheduler.
            start_time = datetime.datetime.now()
            while not job_queue.empty() and not sched.is_done(
                jobs_to_complete
            ):
                job, arrival_time = job_queue.get()
                print(
                    f"Updating job {job_id} duration from {job.duration} to {sum(trace_pickle[job_id]['duration_every_epoch'])}"
                )
                job.duration = sum(
                    trace_pickle[job_id]["duration_every_epoch"]
                )
                job_id += 1
                while True:
                    current_time = datetime.datetime.now()
                    elapsed_seconds = (current_time - start_time).seconds
                    remaining_time = arrival_time - elapsed_seconds
                    if remaining_time <= 0:
                        # job_id = sched.add_job(job)
                        sched.add_job(job)
                        break
                    elif sched.is_done(jobs_to_complete):
                        break
                    else:
                        time.sleep(SLEEP_TIME)

            # Wait for scheduler to complete.
            while not sched.is_done(jobs_to_complete):
                time.sleep(SLEEP_TIME)

            # Print summary information.
            (
                avg_jct,
                geometric_mean_jct,
                harmonic_mean_jct,
                jct_list,
            ) = sched.get_average_jct(job_ids=jobs_to_complete)
            # FTF is now not calculated in a separate plotting script
            (
                finish_time_fairness_list,
                finish_time_fairness_themis_list,
            ) = sched.get_finish_time_fairness(
                job_ids=jobs_to_complete,
                pickle_file_name=f"{args.trace_file}.pickle",
            )
            sched.get_completed_steps(jobs_to_complete)
            cluster_util, utilization_list = sched.get_cluster_utilization()
            (
                extension_percentage,
                num_lease_extensions,
                num_lease_extension_opportunities,
            ) = sched.get_num_lease_extensions()
            per_round_schedule = sched.get_per_round_schedule()
            envy_ratios, envy_list = sched.get_envy_list()
            throughput_timeline = sched.get_throughput_timeline()
            job_run_time = sched.get_job_run_time()

            if args.timeline_dir is not None:
                sched.save_job_timelines(args.timeline_dir)
            elapsed_time = (datetime.datetime.now() - start_time).seconds
            print("Total time taken: %d seconds" % (elapsed_time))
            # dump simulation statistics into pickle file for future plotting
            pickle_object = {
                "trace_file": args.trace_file,
                "policy": args.policy,
                "makespan": elapsed_time,
                "avg_jct": avg_jct,
                "jct_list": jct_list,
                "finish_time_fairness_list": finish_time_fairness_list,
                "finish_time_fairness_themis_list": finish_time_fairness_themis_list,
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

            trace_name = args.trace_file[
                args.trace_file.rfind("/") + 1 : args.trace_file.rfind(".")
            ]
            with open(
                os.path.join(
                    root_dir,
                    pickle_output_dir,
                    f"{args.policy}_{trace_name}_physical.pickle",
                ),
                "wb",
            ) as f:
                pickle.dump(pickle_object, f)
        except KeyboardInterrupt as e:
            pass
        finally:
            sched.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scheduler with trace")
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
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--solver",
        type=str,
        choices=["ECOS", "GUROBI", "SCS"],
        default="ECOS",
        help="CVXPY solver",
    )
    parser.add_argument(
        "--throughputs_file",
        type=str,
        default=None,
        help="Oracle throughputs file",
    )
    parser.add_argument(
        "--expected_num_workers",
        type=int,
        default=None,
        help="Total number of workers expected",
    )
    parser.add_argument(
        "--time_per_iteration",
        type=int,
        default=360,
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
        "--max_rounds",
        type=int,
        default=None,
        help="Maximum number of rounds to run",
    )
    # shockwave additions
    parser.add_argument(
        "--pickle_output_dir",
        type=str,
        default="../../pickle_output_nsdi23",
        help="Path of the directory in which summary files of experiments (*.pickle) are stored",
    )
    parser.add_argument(
        "--config",
        type=str,
        help=".json configuration file that holds the shockwave hyperparameters",
    )
    parser.add_argument(
        "--timeline_dir",
        type=str,
        default=None,
        help="Directory to save timelnes to",
    )
    main(parser.parse_args())
