import argparse
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import pickle
import sys
import utils

from collections import OrderedDict
from PIL import Image

colors = ["r", "g", "b", "c", "m", "y", "k"]

policy_color_dict = {
    "finish_time_fairness": "r",
    "shockwave": "g",
    "max_min_fairness": "b",
    "min_total_duration": "c",
    "gandiva": "Gandiva",
    "allox": "m",
    "max_sum_throughput_perf": "y",
    "fifo": "k",
    "isolated_plus": "tan",
    "gandiva_fair": "silver",
}

policy_name_dict = {
    "finish_time_fairness": "Themis",
    "shockwave": "Shockwave",
    "max_min_fairness": "LAS",
    "min_total_duration": "Job Shop Scheduling/Min Makespan",
    "gandiva": "Gandiva",
    "allox": "Allox",
    "max_sum_throughput_perf": "Max Sum Throughput",
    "fifo": "FIFO",
    "isolated_plus": "Fair Share (isolated_plus)",
    "gandiva_fair": "Gandiva-Fair",
}


def load_pickle(args, policy, is_simulation=True):
    """Loads pickle file.

    Args:
        policy (str): Name of policy

    Returns:
        dict: Summary dictionary of a simulation run
    """
    filename = f"{policy}_{args.expr_name}.pickle"
    filepath = os.path.join(args.pickle_dir, filename)
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except:
        assert policy == "finish_time_fairness"


def write_pickle(args, policy, pickle_dict, is_simulation=True):
    """Writes to pickle file.

    Args:
        policy (str): Name of policy

    Returns:
        dict: Summary dictionary of a simulation run
    """
    filename = f"{policy}_{args.expr_name}.pickle"
    filepath = os.path.join(args.pickle_dir, filename)
    with open(filepath, "wb") as f:
        pickle.dump(pickle_dict, f)


def add_ftf_list(policy, summary_dict):
    """do an element-wise division on the JCT list of current policy
    using the JCT list of the "isolated" policy
    to calculate the finish time fairness rho value 
    for each policy

    Args:
        summary_dict (dict): Summary statistics of an experiment,
        loaded from pickle file
    """
    jct_list = summary_dict["jct_list"]
    # run isolated policy in simulation
    # optionally read from physical experiment run time by changing
    # is_simulation to False
    isolated_jct_list = load_pickle(args, "isolated_plus", is_simulation=True)[
        "jct_list"
    ]
    print(
        f"Calculating FTF list for policy {policy} using isolated_plus as the fair share baseline"
    )
    finish_time_fairness_list = utils.get_finish_time_fairness_list(
        jct_list, isolated_jct_list
    )
    # if "finish_time_fairness_list" in summary_dict.keys():
    #     print(f"WARNING: Overwriting scheduler-generated finish time fairness list")
    summary_dict[
        "finish_time_fairness_isolated_list"
    ] = finish_time_fairness_list
    # TODO: write the isolated jct list to pickle file
    write_pickle(args, policy, summary_dict)
    return summary_dict


def values_to_cdf(values):
    """Turns a list of values into a list of cumulative probabilities.

    Args:
        values (list): List of values

    Returns:
        list: List of cumulative probabilities
    """
    cdf_list = []
    values.sort()
    count = 0
    for v in values:
        count += 1
        cdf_list.append(count / len(values))
    return cdf_list


def plot_cdf(args, pickle_dict, metric):
    """Plot the CDF figure of a given metric.

    Args:
        pickle_dict (dict): Pickle dict loaded using load_pickle()
        pickle_dict (dict): Pickle dict loaded using load_pickle()
        metric (str, optional): One of ["jct", "finish_time_fairness", "utilization"]. Defaults to "jct".
    """
    pickle_key = f"{metric}_list"
    # generate dict of (policy, metric_list)
    metric_dict = {}
    # for policy in args.policies:
    for policy in pickle_dict.keys():
        # if pickle_key not in pickle_dict[policy]:
        # print(f"[Warning] Key {pickle_key} is not in the pickle file of policy {policy}")
        # continue
        # print(f"Plotting {metric} cdf for policy {policy}")
        metric_dict[policy] = pickle_dict[policy][pickle_key]

    for i, (policy, metric_list) in enumerate(metric_dict.items()):
        # x should be values (finish_time_fairness, jct), y should be fraction
        if "fairness" in metric:
            metric_list = [
                (1 if 0.95 <= x <= 1.05 else x) for x in metric_list
            ]
        plt.plot(
            metric_list,
            values_to_cdf(metric_list),
            policy_color_dict[policy],
            # label=f"{policy} (avg: {pickle_dict[policy]['avg_jct']}s)")
            label=policy_name_dict[policy],
        )
        if "finish_time_fairness" in metric or "envy" in metric:
            print(f"{metric} tail of policy {policy} is {max(metric_list)}")
    if "finish_time_fairness" in metric:
        plt.axvline(
            x=1, color="grey", linestyle="dotted", linewidth=1
        )  # plot line that marks ftf rho = 1
    plt.legend(
        loc="upper left"
        if metric
        not in [
            "finish_time_fairness",
            "finish_time_fairness_isolated",
            "finish_time_fairness_themis",
            "envy",
        ]
        else "lower right",
        prop={"size": 8},
    )
    num_jobs, min_duration, max_duration, num_choices = args.trace_name.split(
        "_"
    )[0:4]
    has_multigpu_jobs = "multigpu" in args.trace_name.split("_")
    has_dynamic_jobs = "dynamic" in args.trace_name.split("_")
    plt.title(
        f"{metric} CDF on {args.is_simulation_str} trace {args.trace_name}\n({num_jobs} {'distributed ' if has_multigpu_jobs else ''}{'dynamic ' if has_dynamic_jobs else ''}jobs, {min_duration}hrs ~ {max_duration} hrs each)"
    )
    plt.ylabel("Cumulative Probability")
    xlabel_dict = {
        "jct": "Job Completion Time (s)",
        "finish_time_fairness": "Finish Time Fairness Rho",
        "finish_time_fairness_isolated": "Finish Time Fairness Rho (using isolated_plus as baseline)",
        "finish_time_fairness_themis": "Finish Time Fairness Rho (contention factor is averaged over time)",
        "utilization": "GPU Utilization",
        "envy": "Pairwise Enviness",
    }
    # if metric == "envy":
    #     plt.xlim([0, 0.3])
    if metric in ["finish_time_fairness_isolated", "finish_time_fairness"]:
        plt.xlim([0, 5])
    if metric == "finish_time_fairness_themis":
        plt.xlim([0, 10])
    plt.xlabel(xlabel_dict[metric])
    figpath = pathlib.Path(__file__).parent.joinpath(
        args.figure_dir, f"{args.expr_name}/cdf_{metric}.png"
    )
    plt.savefig(figpath, dpi=300)
    plt.close()


def plot_barchart(args, pickle_dict, metric):
    """Plot a barchart representing the makespan/avg_jct
    of each experiment

    Args:
        pickle_dict (dict): Pickle dict
    """
    if metric == "unfair_fraction":
        metric_dict = {}
        for policy in pickle_dict.keys():
            ftf_list = pickle_dict[policy]["finish_time_fairness_list"]
            # num_unfair_jobs = sum(ftf > 1.05 for ftf in ftf_list)
            num_unfair_jobs = sum(ftf > 1.1 for ftf in ftf_list)
            unfair_fraction = 100 * num_unfair_jobs / len(ftf_list)
            metric_dict[policy] = unfair_fraction
    else:
        metric_dict = {
            policy: pickle_dict[policy][metric]
            for policy in pickle_dict.keys()
        }

    metric_value_list = []

    for i, (policy, value) in enumerate(metric_dict.items()):
        metric_value_list.append(value)
        plt.bar(
            i,
            value,
            color=policy_color_dict[policy],
            label=policy_name_dict[policy],
        )
        print(f"{metric} of policy {policy} is {value}")

    plt.legend(loc="upper left", prop={"size": 8})
    num_jobs, min_duration, max_duration, num_choices = args.trace_name.split(
        "_"
    )[0:4]
    has_multigpu_jobs = "multigpu" in args.trace_name.split("_")
    has_dynamic_jobs = "dynamic" in args.trace_name.split("_")
    plt.title(
        f"{metric} on {args.is_simulation_str} trace {args.trace_name}\n({num_jobs} {'distributed ' if has_multigpu_jobs else ''}{'dynamic ' if has_dynamic_jobs else ''}jobs, {min_duration}hrs ~ {max_duration} hrs each)"
    )
    plt.ylabel(
        f"{metric} ({'%' if metric in ['cluster_util', 'unfair_fraction'] else 's'})"
    )
    plt.xlabel("Policy")

    ylim_min = min(metric_value_list)
    ylim_max = max(metric_value_list)
    vertical_boundary = (ylim_max - ylim_min) / 7

    plt.ylim([ylim_min - vertical_boundary, ylim_max + vertical_boundary])
    figpath = pathlib.Path(__file__).parent.joinpath(
        args.figure_dir, f"{args.expr_name}/bar_{metric}.png"
    )
    plt.savefig(figpath, dpi=300)
    plt.close()


def plot_per_round_schedule(args, policy, pickle_object):
    """Visualizes per-round scheduling decisions by
    plotting a colorful table

    Args:
        policy (str): Name of the policy
        pickle_object (dict): Pickle object
    """
    # get meta statistics of experiment run
    global per_round_schedule
    global num_jobs
    global num_rounds
    global num_gpus
    per_round_schedule = pickle_object["per_round_schedule"]
    num_jobs = len(pickle_object["jct_list"])
    num_rounds = len(per_round_schedule)
    num_gpus = len(pickle_object["utilization_list"])

    print("=" * 50)
    print(f"Experiment with policy {policy} has the following parameters")
    print(f"num_jobs={num_jobs}, num_rounds={num_rounds}, num_gpus={num_gpus}")

    # set up colormap for mapping job_ids to colors
    # cmap = plt.cm.get_cmap("hsv", num_jobs)
    global job_runtime_dict
    global job_duration_cutoff
    job_runtime_dict = [
        pickle_object["job_run_time"][job_id] for job_id in range(num_jobs)
    ]
    job_runtime_dict = [
        sum(all_runtime.values()) / trace_pickle[job_id]["scale_factor"]
        for job_id, all_runtime in enumerate(job_runtime_dict)
    ]  # TODO: divide by scale factor
    print(job_runtime_dict)
    num_cutoff = 20
    import math

    job_duration_cutoff = np.percentile(
        np.array(job_runtime_dict),
        list(range(0, 100, math.floor(100 / num_cutoff))),
    )
    print(
        f"job_duration_cutoff at range(0, 100, 100/num_cutoff) of all job runtime is {job_duration_cutoff}"
    )
    cmap = plt.cm.get_cmap("summer", num_cutoff)

    columns = ["GPU ID \ round ID"] + list(
        range(num_rounds)
    )  # this is the first row
    cell_text, colors_table = parse_schedule(cmap)

    fig, ax = plt.subplots(figsize=(num_rounds / 2, num_gpus / 3))
    ax.axis("tight")
    ax.axis("off")
    the_table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        loc="center",
        cellColours=colors_table,
    )

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.auto_set_column_width(col=list(range(num_rounds)))

    print(
        f"Writing scheduling visualization graph of policy {policy}, this might take a while..."
    )
    figpath = pathlib.Path(__file__).parent.joinpath(
        args.figure_dir,
        f"{args.expr_name}/schedule_visualizations/{policy}.png",
    )
    plt.title(
        f"Per-round schedules using policy {policy_name_dict[policy]} on {args.is_simulation_str} trace {args.trace_name}"
    )
    plt.savefig(figpath, dpi=2 ** 16 / (max(num_rounds, num_gpus) + 1))
    plt.close()


def plot_schedules_of_all_policies(args):
    """Stitches the schedule visualizations of different policies
    """
    images = []
    for policy in args.policies:
        figpath = pathlib.Path(__file__).parent.joinpath(
            args.figure_dir,
            f"{args.expr_name}/schedule_visualizations/{policy}.png",
        )
        images.append(Image.open(figpath))

    widths, heights = zip(*(i.size for i in images))

    total_height = sum(heights)
    max_width = max(widths)

    new_im = Image.new("RGB", (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]

    figpath = pathlib.Path(__file__).parent.joinpath(
        args.figure_dir, f"{args.expr_name}/schedule_visualizations.png"
    )
    new_im.save(figpath)


def parse_schedule(cmap):
    """Parses the scheduling decisions from the 
    Gavel scheduler into list of lists that can
    be directly passed into the matplotlib plotting
    function

    Args:
        cmap: matplotlib colormap

    Returns:
        list, list: Text/color of the table
    """
    # {worker_id: [job_id_in_round_0, job_id_in_round_1, ...], ...}
    schedule_dict = {}

    for gpu_id in range(num_gpus):
        schedule_dict[gpu_id] = []

    for round_id, schedule_of_current_round in enumerate(per_round_schedule):
        # schedule_of_current_round is an OrderedDict that maps job_ids to lists of worker_ids
        # e.g., OrderedDict([(13, (0, 1, 2, 3)), (19, (4, 5, 6, 7)), (21, (8,)), (25, (9,))])
        # print(f"Round {round_id}, schedule_of_current_round is {schedule_of_current_round}")
        overall_worker_job_mappings = []
        for job_id, worker_ids in schedule_of_current_round.items():
            worker_ids = list(worker_ids)
            worker_job_mappings = [
                (worker_id, job_id) for worker_id in worker_ids
            ]
            overall_worker_job_mappings += worker_job_mappings

        # overall_worker_job_mappings maps worker_ids to job_ids
        overall_worker_job_mappings = {
            worker_id: job_id
            for worker_id, job_id in overall_worker_job_mappings
        }
        # print(f"Round {round_id}, overall_worker_job_mappings is {overall_worker_job_mappings}")

        for worker_id in range(num_gpus):
            if worker_id in overall_worker_job_mappings.keys():
                # worker is not idle in this round
                job_id = overall_worker_job_mappings[worker_id]
                schedule_dict[worker_id].append(job_id)
            else:
                # no jobs are scheduled on this GPU in this round
                schedule_dict[worker_id].append(" ")

    cell_text = []
    colors_table = []

    for gpu_id in range(num_gpus):
        schedule = schedule_dict[gpu_id]
        cell_text.append([gpu_id] + schedule_dict[gpu_id])
        # if no jobs are scheduled in a round, make the entry white
        # otherwise, use the colormap to randomly pick a color that corresponds to a job_id
        # colors_table.append(['w'] + [cmap(job_id) if type(job_id) is int else 'w' for job_id in schedule_dict[gpu_id]])
        colors_table.append(
            ["w"]
            + [
                find_color(cmap, job_id) if type(job_id) is int else "w"
                for job_id in schedule_dict[gpu_id]
            ]
        )

    return cell_text, colors_table


def find_color(cmap, job_id):
    job_run_time = job_runtime_dict[job_id]
    i = 0
    while (
        i != len(job_duration_cutoff) and job_run_time > job_duration_cutoff[i]
    ):
        i += 1
    # print(f"Job {job_id}, i is {i}")
    return cmap(i)


def plot(args):
    print("=" * 50)
    print(f"Plotting figures for trace {args.trace_name}")
    pickle_dict = {
        policy: load_pickle(args, policy) for policy in args.policies
    }

    if (
        "finish_time_fairness" in pickle_dict.keys()
        and pickle_dict["finish_time_fairness"] is None
    ):
        print(f"Warning: Themis did not finish")
        pickle_dict.pop("finish_time_fairness")

    # # uncomment to overwrite the scheduler-calculated FTF list
    # # with the JCTs of the isolated policy as the baseline
    # {policy: add_ftf_list(policy, pickle_object)
    #    for policy, pickle_object in pickle_dict.items()}

    # for metric in ["jct", "finish_time_fairness", "utilization", "envy", "finish_time_fairness_isolated", "finish_time_fairness_themis"]:
    for metric in ["jct", "finish_time_fairness", "utilization", "envy"]:
        plot_cdf(args, pickle_dict, metric)
    for metric in [
        "avg_jct",
        "cluster_util",
        "makespan",
        "geometric_mean_jct",
        "harmonic_mean_jct",
        "unfair_fraction",
    ]:
        plot_barchart(args, pickle_dict, metric)

    # # NOTE: trace search helper

    # allox_makespan = pickle_dict["allox"]["makespan"]
    # allox_avgjct = pickle_dict["allox"]["avg_jct"]
    # jobshop_makespan = pickle_dict["min_total_duration"]["makespan"]
    # jobshop_avgjct = pickle_dict["min_total_duration"]["avg_jct"]
    # finish_time_fairness_makespan = pickle_dict["finish_time_fairness"]["makespan"]
    # finish_time_fairness_avgjct = pickle_dict["finish_time_fairness"]["avg_jct"]
    # max_min_fairness_makespan = pickle_dict["max_min_fairness"]["makespan"]
    # max_min_fairness_avgjct = pickle_dict["max_min_fairness"]["avg_jct"]

    # allox_makespan_win = round(allox_makespan / jobshop_makespan, 5)
    # finish_time_fairness_makespan_win = round(finish_time_fairness_makespan / jobshop_makespan, 5)
    # max_min_fairness_makespan_win = round(max_min_fairness_makespan / jobshop_makespan, 5)
    # # print(f"The three win are {allox_win}, {finish_time_fairness_win}, {max_min_fairness_win}")
    # # if allox_win > 1.3 and finish_time_fairness_win > 1.25 and max_min_fairness_win > 1.25:
    # #     print(f"Hello there!")

    # allox_avgjct_win = round(allox_avgjct / jobshop_avgjct, 5)
    # finish_time_fairness_avgjct_win = round(finish_time_fairness_avgjct / jobshop_avgjct, 5)
    # max_min_fairness_avgjct_win = round(max_min_fairness_avgjct / jobshop_avgjct, 5)

    # print(f"Makespan - Allox/Jobshop:{allox_makespan_win}, Themis/Jobshop:{finish_time_fairness_makespan_win},LAS/Jobshop:{max_min_fairness_makespan_win}")
    # print(f"AvgJCT - Allox/Jobshop:{allox_avgjct_win}, Themis/Jobshop:{finish_time_fairness_avgjct_win},LAS/Jobshop:{max_min_fairness_avgjct_win}")

    # mkspwin=1.25
    # ajctwin=1.1
    # if allox_makespan_win > mkspwin and finish_time_fairness_makespan_win > mkspwin and max_min_fairness_makespan_win > mkspwin:
    #     print(f"Bravo!!!!!!!!!!")

    # finish_time_fairness_makespan = pickle_dict["finish_time_fairness"]["makespan"]
    # shockwave_makespan = pickle_dict["shockwave"]["makespan"]
    # finish_time_fairness_makespan_win = finish_time_fairness_makespan / shockwave_makespan
    # shockwave_ftf_list = pickle_dict["shockwave"]["finish_time_fairness_list"]
    # shockwave_fairness_tail = max(shockwave_ftf_list)
    # num_unfair_jobs = sum(ftf > 1.1 for ftf in shockwave_ftf_list)
    # shockwave_unfair_fraction = 100 * num_unfair_jobs / len(shockwave_ftf_list)
    # print(f"finish_time_fairness_makespan_win {finish_time_fairness_makespan_win}, shockwave_fairness_tail {shockwave_fairness_tail}, shockwave_unfair_fraction {shockwave_unfair_fraction}")
    # if finish_time_fairness_makespan_win > 1.1 and shockwave_fairness_tail < 1.5 and shockwave_unfair_fraction <= 10:
    #     print(f"Bravo!!!!!")

    # # scalability
    # shockwave_makespan = pickle_dict["shockwave"]["makespan"]
    # themis_makespan = pickle_dict["finish_time_fairness"]["makespan"]
    # print(f"Shockwave win over Themis in makespan is {round(themis_makespan / shockwave_makespan, 5)}")
    # shockwave_ftf_list = pickle_dict["shockwave"]["finish_time_fairness_list"]
    # themis_ftf_list = pickle_dict["finish_time_fairness"]["finish_time_fairness_list"]
    # shockwave_fairness_tail = max(shockwave_ftf_list)
    # themis_fairness_tail = max(themis_ftf_list)
    # print(f"Shockwave win over Themis in ftf tail is {round(themis_fairness_tail / shockwave_fairness_tail, 5)} = {themis_fairness_tail} / {shockwave_fairness_tail}")
    # num_unfair_jobs = sum(ftf > 1.05 for ftf in shockwave_ftf_list)
    # shockwave_unfair_fraction = 100 * num_unfair_jobs / len(shockwave_ftf_list)
    # num_unfair_jobs = sum(ftf > 1.05 for ftf in themis_ftf_list)
    # themis_unfair_fraction = 100 * num_unfair_jobs / len(shockwave_ftf_list)
    # print(f"Shockwave win over Themis in unfair fracton is {round(themis_unfair_fraction / shockwave_unfair_fraction, 5)} = {themis_unfair_fraction} / {shockwave_unfair_fraction}")

    if args.plot_schedule:
        for policy, pickle_object in pickle_dict.items():
            plot_per_round_schedule(args, policy, pickle_object)
        plot_schedules_of_all_policies(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate figures on statistics of scheduling experiments"
    )
    # required args
    parser.add_argument(
        "--trace_name", type=str, help="Name of the trace to be plotted"
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        default=False,
        help="Set to True if the current experiment is simulation, False otherwise",
    )
    parser.add_argument(
        "--plot_schedule",
        action="store_true",
        default=False,
        help="Set to True to plot the per-round schedule visualization",
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="finish_time_fairness,shockwave,max_min_fairness,min_total_duration,allox,max_sum_throughput_perf",
        help="Comma-separated policies to be plotted",
    )

    # default args
    parser.add_argument(
        "--traces_dir",
        type=str,
        default="traces/simulation",
        help="Path of the directory in which traces are stored",
    )
    parser.add_argument(
        "--pickle_dir",
        type=str,
        default="pickle_output",
        help="Path of the directory in which summary files of experiments (*.pickle) are stored",
    )
    parser.add_argument(
        "--figure_dir",
        type=str,
        default="batch_simulation_figures",
        help="Path of the directory in which the figures will be written to",
    )

    args = parser.parse_args()

    # # # manual override if necessary
    # # args.trace_name = "50_0.2_5_100_100_342_0,0.5,0.5_0.6,0.3,0.1,0_multigpu_dynamic"
    # # args.trace_name = "30_0.2_5_100_100_302_0,0.5,0.5_0.6,0.3,0.1,0_multigpu_dynamic"
    # # args.trace_name = "20_0.2_5_100_150_161_0,0.5,0.5_0.6,0.3,0.1,0_multigpu_dynamic"
    # args.trace_name = "120_0.2_5_100_40_25_0,0.5,0.5_0.6,0.3,0.09,0.01_multigpu_dynamic"
    # args.policies = "shockwave,finish_time_fairness,allox,max_min_fairness,min_total_duration,max_sum_throughput_perf"
    # # args.policies = "shockwave,finish_time_fairness,allox,max_min_fairness,min_total_duration"
    # # args.policies = "shockwave,finish_time_fairness,max_min_fairness,min_total_duration"
    # # args.policies = "shockwave,finish_time_fairness,max_min_fairness,allox"
    # # args.simulation = True
    # # args.plot_schedule = True
    # # # args.traces_dir = "traces/simulation"
    # args.traces_dir = "traces"
    # args.pickle_dir = "pickle_output"
    # # args.pickle_dir = "simulator_logs"
    # # args.figure_dir = "batch_simulation_figures"
    # # args.figure_dir = "figures"
    # args.figure_dir = "figures_test"

    global trace_pickle
    pickle_filepath = os.path.join(
        args.traces_dir, f"{args.trace_name}.pickle"
    )
    trace_pickle = pickle.load(open(pickle_filepath, "rb"))
    # print(trace_pickle[0])

    args.policies = args.policies.split(",")
    args.is_simulation_str = "simulation" if args.simulation else "physical"
    home_dir = os.path.dirname(os.path.abspath(__file__))
    args.expr_name = f"{args.trace_name}_{args.is_simulation_str}"
    args.output_dir = os.path.join(home_dir, args.figure_dir, args.expr_name)
    args.visualization_dir = os.path.join(
        args.output_dir, "schedule_visualizations"
    )

    try:
        os.mkdir(os.path.join(home_dir, args.figure_dir))
    except:
        pass
    try:
        os.mkdir(args.output_dir)
    except:
        pass
    try:
        os.mkdir(args.visualization_dir)
    except:
        pass

    plot(args)
