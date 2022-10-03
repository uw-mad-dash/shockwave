import csv
from datetime import datetime
import json
import math
import numpy as np
import os
import pickle
import psutil
import random
import re
import socket
import subprocess

from job import Job
from job_table import JobTable
from policies import (
    allox,
    fifo,
    finish_time_fairness,
    gandiva,
    gandiva_fair_proportional,
    isolated,
    isolated_plus,
    max_min_fairness,
    max_min_fairness_water_filling,
    max_sum_throughput,
    min_total_duration,
    shockwave,
)

from itertools import groupby
from collections import Counter


root_dir = os.path.split(os.path.abspath(__file__))[0]

model_dataset_mapping = {
    "ResNet-18": "CIFAR-10",
    "ResNet-50": "ImageNet",
    "Transformer": "Multi30k",
    "LM": "Wikitext-2",
    "Recommendation": "ML-20M",
    "A3C": "Pong",
    "CycleGAN": "monet2photo",
}
dataset_len = {
    "CIFAR-10": 50000,
    "ImageNet": 100000,
    "Multi30k": 10000,
    "Wikitext-2": 59675,
    "ML-20M": 117907,
    "Pong": 4,
    "monet2photo": 6287,
}

parsed_throughputs = None


def get_finish_time_fairness_list(jct_list, isolated_jct_list):
    """Generate and print a list of finish time fairness rhos
    for a trace of jobs based on their isolated JCT 
    (1/N of cluster) and actual JCT on a policy

    Args:
        jct_list (list): List of floats of JCTs using target policy
        isolated_jct_list (list): List of floats of JCTs using isolated policy

    Returns:
        list: List of finish time fairness rhos
    """
    assert len(jct_list) == len(isolated_jct_list)

    finish_time_fairness_list = []

    for i in range(len(jct_list)):
        jct = jct_list[i]
        isolated_jct = isolated_jct_list[i]
        # TODO: should we do a "*= (self._num_jobs_in_trace / num_gpus)" on isolated_jct?
        print(
            f"Job {i} FTF Rho: {round(jct / isolated_jct, 2)}={jct}/{isolated_jct}"
        )
        rho = round(jct / isolated_jct, 5)
        print(
            f"Job {i}, rho: {rho} = {round(jct, 5)} / {round(isolated_jct, 5)}"
        )
        finish_time_fairness_list.append(rho)

    # print(f"Finish time fairness rho values for all jobs: {finish_time_fairness_list}")
    print(
        f"Finish time percentiles: {np.percentile(np.array(finish_time_fairness_list), [0, 25, 50, 75, 100])}"
    )

    return finish_time_fairness_list


def _generate_scale_factor(rng):
    # Sample the scale factor from the Philly distribution.
    scale_factor = 1
    r = rng.uniform(0, 1)
    if 0.7 <= r <= 0.8:
        scale_factor = 2
    elif 0.8 <= r <= 0.95:
        scale_factor = 4
    elif 0.95 <= r:
        scale_factor = 8
    return scale_factor


def _generate_duration(rng):
    # Sample the job duration from the Philly distribution.
    if rng.random() >= 0.8:
        run_time = 60 * (10 ** rng.uniform(3, 4))
    else:
        run_time = 60 * (10 ** rng.uniform(1.5, 3))
    return run_time


def generate_job(
    throughputs,
    reference_worker_type="v100",
    rng=None,
    job_id=None,
    fixed_job_duration=None,
    generate_multi_gpu_jobs=False,
    generate_multi_priority_jobs=False,
    generate_dynamic_jobs=False,
    run_dir=None,
    scale_factor_generator_func=_generate_scale_factor,
    scale_factor_mix=None,
    #  duration_generator_func=_generate_duration,
    duration_generator_func=None,
    mode_generator_func=None,
    mode_mix=None,
    single_mode=None,
    scale_factor_rng=None,
    duration_rng=None,
    mode_rng=None,
    SLO_rng=None,
    always_generate_scale_factor=True,
):
    """Generates a new job.

       Args:
         throughputs: A dict containing pre-measured throughputs.
         reference_worker_type: The worker type to use when calculating steps.
         rng: A random number generator for selecting job parameters.
         job_id: The job's ID.
         fixed_job_duration: If set, fixes the duration to the specified value.
         generate_multi_gpu_jobs: If set, generate a scale factor >= 1.
         generate_multi_priority_jobs: If set, generate a priority >= 1.
         run_dir: The directory to run the job from.
         scale_factor_generator_func: A function that accepts an RNG parameter
                                      and returns a job size.
         duration_generator_func: A function that accepts an RNG parameter and
                                  returns a job duration in seconds.
         scale_factor_rng: A random number generator specifically for
                           generating scale factors.
         duration_rng: A random number generator specifically for generating
                       durations.
         SLO_rng: If set, generate an SLO >= 1 using this RNG.
         always_generate_scale_factor: If set, generate a scale factor
                                       regardless of whether user has
                                       requested multi-GPU jobs.
      Returns:
        The generated Job.
    """
    if rng is None:
        rng = random.Random()
    if scale_factor_rng is None:
        scale_factor_rng = rng
    if duration_rng is None:
        duration_rng = rng
    if mode_rng is None:
        mode_rng = rng

    job_template = None

    if always_generate_scale_factor:
        scale_factor = scale_factor_generator_func(
            scale_factor_rng, scale_factor_mix
        )
    else:
        # NOTE: We select the job template here to maintain backwards
        # compatability with scripts/utils/generate_trace.py
        job_template = rng.choice(JobTable)
        if generate_multi_gpu_jobs and job_template.distributed:
            scale_factor = scale_factor_generator_func(
                scale_factor_rng, scale_factor_mix
            )
        else:
            scale_factor = 1

    if fixed_job_duration:
        run_time = fixed_job_duration
    elif (type(duration_generator_func) is not float) and (
        type(duration_generator_func) is not int
    ):
        run_time = duration_generator_func(duration_rng)
    else:
        # use pre-profiled job duration from the pollux trace
        # to my future self, sorry about the terrible code change
        run_time = int(duration_generator_func)
    if not generate_multi_gpu_jobs:
        scale_factor = 1
    if single_mode is not None:
        mode = single_mode
    elif generate_dynamic_jobs:
        mode = mode_generator_func(mode_rng, mode_mix)
    else:
        mode = "static"
    assert run_time > 0
    assert scale_factor >= 1 and scale_factor <= 8

    # prevent short jobs from accordion-ing into super-short jobs, ruining tail ftf for all policies
    if run_time < 1000 and mode == "accordion":
        mode = "static"

    # Sample the job type.
    if job_template is None:
        while True:
            job_template = rng.choice(JobTable)
            if scale_factor == 1 or (
                scale_factor > 1 and job_template.distributed
            ):
                break
    job_type = job_template.model

    # Complete the job command with the run directory.
    command = job_template.command
    if run_dir is not None:
        if job_template.needs_data_dir:
            command = command % (run_dir, run_dir)
        else:
            command = command % (run_dir)

    # Compute the number of steps the job will run for given its duration.
    key = (job_type, scale_factor)
    assert key in throughputs[reference_worker_type]
    num_steps = run_time * throughputs[reference_worker_type][key]["null"]
    assert num_steps > 0

    # Optionally assign a priority to the job.
    priority_weight = 1.0
    if generate_multi_priority_jobs:
        r = rng.uniform(0, 1)
        if 0.0 <= r <= 0.2:
            priority_weight = 5.0

    # Optionally assign an SLO to the job.
    SLO = None
    if SLO_rng is not None:
        r = SLO_rng.uniform(0, 1)
        if 0.0 <= r < 0.33:
            SLO = 1.2
        elif 0.33 <= r < 0.67:
            SLO = 2.0
        else:
            SLO = 10.0

    job = Job(
        job_id=job_id,
        job_type=job_type,
        command=command,
        working_directory=job_template.working_directory,
        num_steps_arg=job_template.num_steps_arg,
        total_steps=num_steps,
        duration=run_time,
        scale_factor=scale_factor,
        mode=mode,
        priority_weight=priority_weight,
        SLO=SLO,
        needs_data_dir=job_template.needs_data_dir,
    )

    return job


def load_philly_job_distribution():
    with open("philly_job_distribution.pickle", "rb") as f:
        return pickle.load(f)


def get_ip_address():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address


def get_num_gpus():
    command = "nvidia-smi -L"
    output = (
        subprocess.run(command, stdout=subprocess.PIPE, check=True, shell=True)
        .stdout.decode("utf-8")
        .strip()
    )
    return len(output.split("\n"))


def get_pid_for_job(command):
    pids = []
    for proc in psutil.process_iter():
        cmdline = " ".join(proc.cmdline())
        if cmdline == command:
            pids.append(proc.pid)
    return min(pids)


def get_gpu_processes():
    output = subprocess.check_output("nvidia-smi").decode("utf-8")
    gpu_processes = {}
    processes_flag = False
    for line in output.split("\n"):
        if "Processes" in line:
            processes_flag = True
            continue
        if processes_flag:
            res = re.search("(\d+) +(\d+) +(\w+) +(.+) +(\d+)MiB", line)
            if res is not None:
                gpu_id = int(res.group(1))
                if gpu_id not in gpu_processes:
                    gpu_processes[gpu_id] = []
                pid = int(res.group(2))
                process_name = res.group(4)
                if process_name != "nvidia-cuda-mps-server":
                    gpu_processes[gpu_id].append(pid)
    return gpu_processes


def get_available_policies():
    return [
        "allox",
        "fifo",
        "fifo_perf",
        "fifo_packed",
        "finish_time_fairness",
        "finish_time_fairness_perf",
        "finish_time_fairness_packed",
        "gandiva",
        "gandiva_fair",
        "isolated",
        "isolated_plus",
        "max_min_fairness",
        "max_min_fairness_perf",
        "max_min_fairness_packed",
        "max_min_fairness_water_filling",
        "max_min_fairness_water_filling_perf",
        "max_min_fairness_water_filling_packed",
        "max_sum_throughput_perf",
        "max_sum_throughput_normalized_by_cost_perf",
        "max_sum_throughput_normalized_by_cost_perf_SLOs",
        "max_sum_throughput_normalized_by_cost_packed_SLOs",
        "min_total_duration",
        "min_total_duration_perf",
        "min_total_duration_packed",
        "shockwave",
    ]


def read_per_instance_type_spot_prices_aws(directory):
    # TODO: Make this flexible.
    directory = os.path.join(directory, "us-east-1")
    per_instance_type_spot_prices = {}
    for filename in os.listdir(directory):
        full_filepath = os.path.join(directory, filename)
        with open(full_filepath, "r") as f:
            json_obj = json.load(f)
            for x in json_obj["SpotPriceHistory"]:
                instance_type = x["InstanceType"]
                if instance_type not in per_instance_type_spot_prices:
                    per_instance_type_spot_prices[instance_type] = []
                per_instance_type_spot_prices[instance_type].append(x)
    return per_instance_type_spot_prices


def read_per_instance_type_spot_prices_azure(directory):
    per_instance_type_spot_prices = {}
    for filename in os.listdir(directory):
        full_filepath = os.path.join(directory, filename)
        with open(full_filepath, "r") as f:
            zone = filename.replace(".csv", "")
            reader = csv.reader(f)
            i = 0
            for row in reader:
                if i == 0:
                    header = row
                    for header_elem in header[1:]:
                        if header_elem not in per_instance_type_spot_prices:
                            per_instance_type_spot_prices[header_elem] = {}
                else:
                    for (header_elem, row_elem) in zip(header[1:], row[1:]):
                        if (
                            zone
                            not in per_instance_type_spot_prices[header_elem]
                        ):
                            per_instance_type_spot_prices[header_elem][
                                zone
                            ] = []
                        date = datetime.strptime(row[0], "%m/%d/%Y")
                        per_instance_type_spot_prices[header_elem][
                            zone
                        ].append((date, row_elem))
                i += 1
    return per_instance_type_spot_prices


def read_per_instance_type_spot_prices_json(directory):
    per_instance_type_spot_prices = {}
    per_instance_type_spot_prices[
        "aws"
    ] = read_per_instance_type_spot_prices_aws(
        os.path.join(directory, "aws/logs")
    )
    per_instance_type_spot_prices[
        "azure"
    ] = read_per_instance_type_spot_prices_azure(
        os.path.join(directory, "azure/logs")
    )
    per_instance_type_spot_prices["gcp"] = {
        "v100": 0.74,
        "p100": 0.43,
        "k80": 0.135,
    }
    return per_instance_type_spot_prices


def get_latest_price_for_worker_type_aws(
    worker_type, current_time, per_instance_type_spot_prices
):
    # TODO: Make this function more efficient.
    if worker_type == "v100":
        instance_type = "p3.2xlarge"
    elif worker_type == "p100":
        # NOTE: AWS does not have single P100 instances, use 1.5x K80 price
        # as a proxy.
        instance_type = "p2.xlarge"
    elif worker_type == "k80":
        instance_type = "p2.xlarge"

    timestamps = [
        datetime.strptime(x["Timestamp"], "%Y-%m-%dT%H:%M:%S.000Z")
        for x in per_instance_type_spot_prices[instance_type]
    ]
    timestamps.sort()

    availability_zones = [
        x["AvailabilityZone"]
        for x in per_instance_type_spot_prices[instance_type]
    ]
    latest_prices = []
    for availability_zone in set(availability_zones):
        per_instance_type_spot_prices[instance_type].sort(
            key=lambda x: datetime.strptime(
                x["Timestamp"], "%Y-%m-%dT%H:%M:%S.000Z"
            )
        )
        latest_price = None
        for x in per_instance_type_spot_prices[instance_type]:
            if x["AvailabilityZone"] != availability_zone:
                continue
            timestamp = (
                datetime.strptime(x["Timestamp"], "%Y-%m-%dT%H:%M:%S.000Z")
                - timestamps[0]
            ).total_seconds()
            if timestamp > current_time and latest_price is not None:
                break
            latest_price = float(x["SpotPrice"])
        assert latest_price is not None
        latest_prices.append(latest_price)

    # NOTE: AWS does not have single P100 instances, use 1.5x K80 price
    # as a proxy.
    if worker_type == "p100":
        return min(latest_prices) * 1.5
    else:
        return min(latest_prices)


def get_latest_price_for_worker_type_gcp(
    worker_type, current_time, per_instance_type_spot_prices
):
    return per_instance_type_spot_prices[worker_type]


def get_latest_price_for_worker_type_azure(
    worker_type, current_time, per_instance_type_spot_prices
):
    if worker_type == "k80":
        instance_type = "NC6"
    elif worker_type == "p100":
        instance_type = "NC6s v2"
    elif worker_type == "v100":
        instance_type = "NC6s v3"

    earliest_timestamps = []
    for zone in per_instance_type_spot_prices[instance_type]:
        per_instance_type_spot_prices[instance_type][zone].sort(
            key=lambda x: x[0]
        )
        earliest_timestamps.append(
            per_instance_type_spot_prices[instance_type][zone][0][0]
        )
    earliest_timestamp = min(earliest_timestamps)
    latest_prices = []
    for zone in per_instance_type_spot_prices[instance_type]:
        latest_price = None
        for x in per_instance_type_spot_prices[instance_type][zone]:
            timestamp = (x[0] - earliest_timestamp).total_seconds()
            if timestamp > current_time and latest_price is not None:
                break
            elif x[1] == "":
                continue
            else:
                # Remove '$' character.
                latest_price = float(x[1][1:])
    return latest_price


def get_latest_price_for_worker_type(
    worker_type, current_time, per_instance_type_spot_prices, available_clouds
):
    assert len(available_clouds) > 0
    prices = []
    if "aws" in available_clouds:
        aws_price = get_latest_price_for_worker_type_aws(
            worker_type, current_time, per_instance_type_spot_prices["aws"]
        )
        prices.append(aws_price)
    if "gcp" in available_clouds:
        gcp_price = get_latest_price_for_worker_type_gcp(
            worker_type, current_time, per_instance_type_spot_prices["gcp"]
        )
        prices.append(gcp_price)
    if "azure" in available_clouds:
        azure_price = get_latest_price_for_worker_type_azure(
            worker_type, current_time, per_instance_type_spot_prices["azure"]
        )
        prices.append(azure_price)

    return min(prices)


def parse_job_type_str(job_type):
    if job_type is None:
        return None
    match = re.match("(.*) \(scale factor (\d+)\)", job_type)
    if match is None:
        return (job_type, 1)
    model = match.group(1)
    scale_factor = int(match.group(2))
    return (model, scale_factor)


def parse_job_type_tuple(job_type):
    match = re.match("\('(.*)', (\d+)\)", job_type)
    if match is None:
        return None
    model = match.group(1)
    scale_factor = int(match.group(2))
    return (model, scale_factor)


def stringify_throughputs(throughputs):
    stringified_throughputs = {}
    for worker_type in throughputs:
        stringified_throughputs[worker_type] = {}
        for key in throughputs[worker_type]:
            stringified_throughputs[worker_type][str(key)] = {}
            for other_key in throughputs[worker_type][key]:
                stringified_throughputs[worker_type][str(key)][
                    str(other_key)
                ] = throughputs[worker_type][key][other_key]
    return stringified_throughputs


def read_all_throughputs_json_v2(file_name):
    with open(file_name, "r") as f:
        raw_throughputs = json.load(f)
    parsed_throughputs = {}
    for worker_type in raw_throughputs:
        parsed_throughputs[worker_type] = {}
        for job_type in raw_throughputs[worker_type]:
            key = parse_job_type_tuple(job_type)
            assert key is not None
            parsed_throughputs[worker_type][key] = {}
            for other_job_type in raw_throughputs[worker_type][job_type]:
                if other_job_type == "null":
                    other_key = other_job_type
                else:
                    other_key = parse_job_type_tuple(other_job_type)
                    assert other_key is not None
                parsed_throughputs[worker_type][key][
                    other_key
                ] = raw_throughputs[worker_type][job_type][other_job_type]
    return parsed_throughputs


def read_all_throughputs_json(throughputs_file):
    with open(throughputs_file, "r") as f:
        throughputs = json.load(f)
    return throughputs


def get_policy(
    policy_name, solver=None, seed=None, priority_reweighting_policies=None
):
    if policy_name.startswith("allox"):
        if policy_name == "allox":
            alpha = 0.2
        else:
            alpha = float(policy_name.split("allox_alpha=")[1])
        policy = allox.AlloXPolicy(alpha=alpha)
    elif policy_name == "fifo":
        policy = fifo.FIFOPolicy(seed=seed)
    elif policy_name == "fifo_perf":
        policy = fifo.FIFOPolicyWithPerf()
    elif policy_name == "fifo_packed":
        policy = fifo.FIFOPolicyWithPacking()
    elif policy_name == "finish_time_fairness":
        # policy = finish_time_fairness.FinishTimeFairnessPolicy(solver=solver)
        policy = finish_time_fairness.FinishTimeFairnessPolicy(solver="GUROBI")
    elif policy_name == "finish_time_fairness_perf":
        policy = finish_time_fairness.FinishTimeFairnessPolicyWithPerf(
            solver=solver
        )
    elif policy_name == "finish_time_fairness_packed":
        policy = finish_time_fairness.FinishTimeFairnessPolicyWithPacking(
            solver=solver
        )
    elif policy_name == "gandiva":
        policy = gandiva.GandivaPolicy(seed=seed)
    elif policy_name == "gandiva_fair":
        policy = gandiva_fair_proportional.GandivProportionalPolicy()
    elif policy_name == "isolated":
        policy = isolated.IsolatedPolicy()
    elif policy_name == "isolated_plus":
        policy = isolated_plus.IsolatedPolicy()
    elif policy_name == "max_min_fairness":
        policy = max_min_fairness.MaxMinFairnessPolicy(solver=solver)
    elif policy_name == "max_min_fairness_perf":
        policy = max_min_fairness.MaxMinFairnessPolicyWithPerf(solver=solver)
    elif policy_name == "max_min_fairness_packed":
        policy = max_min_fairness.MaxMinFairnessPolicyWithPacking(
            solver=solver
        )
    elif policy_name == "max_min_fairness_water_filling":
        policy = max_min_fairness_water_filling.MaxMinFairnessWaterFillingPolicy(
            priority_reweighting_policies=priority_reweighting_policies
        )
    elif policy_name == "max_min_fairness_water_filling_perf":
        policy = max_min_fairness_water_filling.MaxMinFairnessWaterFillingPolicyWithPerf(
            priority_reweighting_policies=priority_reweighting_policies
        )
    elif policy_name == "max_min_fairness_water_filling_packed":
        policy = max_min_fairness_water_filling.MaxMinFairnessWaterFillingPolicyWithPacking(
            priority_reweighting_policies=priority_reweighting_policies
        )
    elif policy_name == "max_sum_throughput_perf":
        policy = max_sum_throughput.ThroughputSumWithPerf(solver=solver)
    elif policy_name == "max_sum_throughput_normalized_by_cost_perf":
        policy = max_sum_throughput.ThroughputNormalizedByCostSumWithPerf(
            solver=solver
        )
    elif policy_name == "max_sum_throughput_normalized_by_cost_perf_SLOs":
        policy = max_sum_throughput.ThroughputNormalizedByCostSumWithPerfSLOs(
            solver=solver
        )
    elif policy_name == "max_sum_throughput_normalized_by_cost_packed_SLOs":
        policy = max_sum_throughput.ThroughputNormalizedByCostSumWithPackingSLOs(
            solver=solver
        )
    elif policy_name == "min_total_duration":
        policy = min_total_duration.MinTotalDurationPolicy(solver=solver)
    elif policy_name == "min_total_duration_perf":
        policy = min_total_duration.MinTotalDurationPolicyWithPerf(
            solver=solver
        )
    elif policy_name == "min_total_duration_packed":
        policy = min_total_duration.MinTotalDurationPolicyWithPacking(
            solver=solver
        )
    elif policy_name == "shockwave":
        policy = shockwave.ShockwavePolicy()
    else:
        raise ValueError("Unknown policy!")
    return policy


def get_metric(
    model, batch_size, metric, parsed_throughputs=None, scale_factor=None
):
    """Returns the pre-profiled per-epoch training time of a job
    on a V100 GPU
    """
    if parsed_throughputs is not None and metric == "duration":
        # print(parsed_throughputs["v100"].keys())
        # print(parse_job_type_tuple(job_type))
        job_type = f"{model} (batch size {batch_size})"
        throughput = parsed_throughputs["v100"][(job_type, scale_factor)][
            "null"
        ]
        dataset_size = dataset_len[model_dataset_mapping[model]]
        num_iters_per_epoch = dataset_size / batch_size
        # print(f"round(num_iters_per_epoch / throughput) = {num_iters_per_epoch} / {throughput} = {round(num_iters_per_epoch / throughput)}")
        return num_iters_per_epoch / throughput

    metadata_dict = {
        "mem": {
            "ResNet-18": {16: 1771, 32: 1857, 64: 2925, 128: 4137, 256: 3581},
            "ResNet-50": {16: 3279, 32: 4597, 64: 4949, 128: 10289},
            "Transformer": {16: 3145, 32: 4219, 64: 7199, 128: 12197},
            "LM": {5: 1687, 10: 1789, 20: 1983, 40: 2415, 80: 3337},
            "Recommendation": {
                512: 1751,
                1024: 2373,
                2048: 3559,
                4096: 6565,
                8192: 7699,
            },
            "CycleGAN": {1: 7901, 2: 8435, 4: 12291},
            "A3C": {4: 5880},
        },
        "util": {
            "ResNet-18": {16: 76.8, 32: 87.6, 64: 95.5, 128: 98.0, 256: 98.8},
            "ResNet-50": {16: 96.0, 32: 96.4, 64: 98.8, 128: 99.2},
            "Transformer": {16: 76.7, 32: 82.0, 64: 88.8, 128: 93.8},
            "LM": {5: 71.5, 10: 67.6, 20: 60.8, 40: 58.9, 80: 60.0},
            "Recommendation": {
                512: 12.3,
                1024: 8.9,
                2048: 12.2,
                4096: 10.9,
                8192: 15.3,
            },
            "CycleGAN": {1: 96.0, 2: 98.0, 4: 98.0},
            "A3C": {4: 88.0},
        },
    }
    return metadata_dict[metric][model][batch_size]


def get_accordion_bs_pattern(job_type, initial_batch_size, num_epochs, job_id):
    """Takes in the specifications of a job, return the batch size
    pattern as a list after applying Accordion
    """
    model = job_type[: job_type.find(" ")]
    bs_every_epoch = [initial_batch_size] * num_epochs

    critical_regime = []

    if model == "ResNet-18":
        if initial_batch_size in [16, 32, 64, 128]:
            regime = 10
        elif initial_batch_size in [256]:
            regime = 20
        critical_regime = (
            list(range(regime)) + list(range(150, 160)) + list(range(250, 260))
        )
    elif model == "ResNet-50":
        if initial_batch_size in [16, 32, 64, 128]:
            regime = 10
        critical_regime = [x for x in range(600) if x % 30 < 10]
    elif model == "LM":
        regime = 10
        critical_regime = list(range(regime))
    elif model in ["Transformer", "CycleGAN", "A3C"]:
        return bs_every_epoch
    elif model == "Recommendation":
        if initial_batch_size in [512, 1024]:
            regime = 30
        elif initial_batch_size in [2048]:
            regime = 40
        elif initial_batch_size in [4096, 8192]:
            regime = 10
        critical_regime = (
            list(range(regime)) + list(range(60, 70)) + list(range(80, 90))
        )

    max_bs_dict = {  # max possible bs of jobs that support accordion bs scaling
        "LM": 80,
        "ResNet-18": 256,
        "ResNet-50": 128,
        "Recommendation": 8192,
    }

    max_bs = (
        max_bs_dict[model]
        if model in max_bs_dict.keys()
        else initial_batch_size
    )

    for (epoch, bs) in enumerate(bs_every_epoch):
        # switch to large batch size when outside of critical regime
        # if epoch not in critical_regime:
        if (epoch not in critical_regime) and (epoch > num_epochs * 0.3):
            # force the first 30% of a job to be in critical regime to preserve accuracy
            bs_every_epoch[epoch] = max_bs

    return bs_every_epoch


def get_gns_bs_pattern(job_type, batch_size, num_epochs, scale_factor):
    """Takes in the specifications of a job, return the batch size
    pattern as a list after applying GNS
    """
    # bs_limit = {"ResNet-18": 256,"ResNet-50": 128,"Transformer": 128,"LM": 80,"Recommendation": 8192}
    model = job_type[: job_type.find(" ")]
    # print("num_epochs are ",num_epochs)
    regular_mode_bs = [batch_size] * num_epochs
    # print(len(regular_mode_bs))
    bs_limit = {
        "ResNet-18": 256,
        "ResNet-50": 128,
        "Transformer": 128,
        "LM": 80,
        "Recommendation": 8192,
    }
    if (
        model == "ResNet-18"
        and batch_size == 16
        and scale_factor == 1
        and num_epochs > 31
    ):
        for epoch in range(31, 41):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
        for epoch in range(41, 51):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 4
        for epoch in range(51, 71):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 8
        for epoch in range(71, num_epochs):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 16
    elif (
        model == "ResNet-18"
        and batch_size == 32
        and scale_factor == 1
        and num_epochs > 21
    ):
        for epoch in range(21, 31):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
        for epoch in range(31, 51):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 4
        for epoch in range(51, num_epochs):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 8
    elif (
        model == "ResNet-18"
        and batch_size == 64
        and scale_factor == 1
        and num_epochs > 11
    ):
        for epoch in range(11, 31):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
        for epoch in range(31, num_epochs):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 4
    elif (
        model == "ResNet-18"
        and batch_size == 128
        and scale_factor == 1
        and num_epochs > 11
    ):
        for epoch in range(11, num_epochs):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
    elif (
        model == "ResNet-18"
        and batch_size == 16
        and scale_factor == 2
        and num_epochs > 21
    ):
        for epoch in range(21, 31):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
        for epoch in range(31, 91):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 4
        for epoch in range(91, 111):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 8
        for epoch in range(111, num_epochs):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 16
    elif (
        model == "ResNet-18"
        and batch_size == 32
        and scale_factor == 2
        and num_epochs > 11
    ):
        for epoch in range(11, 21):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
        for epoch in range(21, 41):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 4
        for epoch in range(41, num_epochs):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 8
    elif (
        model == "ResNet-18"
        and batch_size == 64
        and scale_factor == 2
        and num_epochs > 21
    ):
        for epoch in range(21, 41):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
        for epoch in range(41, num_epochs):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 4
    elif (
        model == "ResNet-18"
        and batch_size == 128
        and scale_factor == 2
        and num_epochs > 41
    ):
        for epoch in range(41, num_epochs):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
    elif (
        model == "ResNet-18"
        and batch_size == 16
        and scale_factor == 4
        and num_epochs > 11
    ):
        for epoch in range(11, 21):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
        for epoch in range(21, 81):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 4
        for epoch in range(81, 91):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 8
        for epoch in range(91, num_epochs):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 16
    elif (
        model == "ResNet-18"
        and batch_size == 32
        and scale_factor == 4
        and num_epochs > 21
    ):
        for epoch in range(21, 31):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
        for epoch in range(31, 61):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 4
        for epoch in range(61, num_epochs):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 8
    elif (
        model == "ResNet-18"
        and batch_size == 64
        and scale_factor == 4
        and num_epochs > 11
    ):
        for epoch in range(11, 61):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
        for epoch in range(61, num_epochs):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 4
    elif (
        model == "ResNet-18"
        and batch_size == 128
        and scale_factor == 4
        and num_epochs > 11
    ):
        for epoch in range(11, num_epochs):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
    elif (
        model == "ResNet-50"
        and batch_size == 64
        and scale_factor == 1
        and num_epochs > 101
    ):
        for epoch in range(101, num_epochs):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
    elif (
        model == "ResNet-50"
        and batch_size == 32
        and scale_factor == 2
        and num_epochs > 101
    ):
        for epoch in range(101, 111):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
        for epoch in range(111, num_epochs):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 4
    elif (
        model == "ResNet-50"
        and batch_size == 64
        and scale_factor == 2
        and num_epochs > 81
    ):
        for epoch in range(81, num_epochs):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
    elif (
        model == "ResNet-50"
        and batch_size == 32
        and scale_factor == 4
        and num_epochs > 131
    ):
        for epoch in range(131, 221):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
        for epoch in range(221, num_epochs):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 4
    elif (
        model == "ResNet-50"
        and batch_size == 64
        and scale_factor == 4
        and num_epochs > 191
    ):
        for epoch in range(191, num_epochs):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
    elif (
        model == "LM"
        and batch_size == 5
        and scale_factor == 1
        and num_epochs > 31
    ):
        for epoch in range(31, 41):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
        for epoch in range(41, 61):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 4
        for epoch in range(61, 71):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 8
        for epoch in range(71, num_epochs):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 16
    elif (
        model == "LM"
        and batch_size == 10
        and scale_factor == 1
        and num_epochs > 11
    ):
        for epoch in range(11, 21):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
        for epoch in range(21, 41):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 4
        for epoch in range(41, num_epochs):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 8
    elif (
        model == "LM"
        and batch_size == 20
        and scale_factor == 1
        and num_epochs > 11
    ):
        for epoch in range(11, 41):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
        for epoch in range(41, num_epochs):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 4
    elif (
        model == "LM"
        and batch_size == 40
        and scale_factor == 1
        and num_epochs > 11
    ):
        for epoch in range(11, num_epochs):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
    elif (
        model == "LM"
        and batch_size == 5
        and scale_factor == 2
        and num_epochs > 31
    ):
        for epoch in range(31, 51):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
        for epoch in range(51, 61):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 4
        for epoch in range(61, 71):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 8
        for epoch in range(71, num_epochs):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 16
    elif (
        model == "LM"
        and batch_size == 10
        and scale_factor == 2
        and num_epochs > 11
    ):
        for epoch in range(11, 31):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
        for epoch in range(31, 41):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 4
        for epoch in range(41, num_epochs):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 8
    elif (
        model == "LM"
        and batch_size == 20
        and scale_factor == 2
        and num_epochs > 31
    ):
        for epoch in range(31, 41):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
        for epoch in range(41, num_epochs):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 4
    elif (
        model == "LM"
        and batch_size == 40
        and scale_factor == 2
        and num_epochs > 11
    ):
        for epoch in range(11, num_epochs):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
    elif (
        model == "LM"
        and batch_size == 5
        and scale_factor == 4
        and num_epochs > 11
    ):
        for epoch in range(11, 31):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
        for epoch in range(31, 71):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 4
        for epoch in range(71, 91):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 8
        for epoch in range(91, num_epochs):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 16
    elif (
        model == "LM"
        and batch_size == 10
        and scale_factor == 4
        and num_epochs > 11
    ):
        for epoch in range(11, 31):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
        for epoch in range(31, 61):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 4
        for epoch in range(61, num_epochs):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 8
    elif (
        model == "LM"
        and batch_size == 20
        and scale_factor == 4
        and num_epochs > 11
    ):
        for epoch in range(11, 61):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
        for epoch in range(61, num_epochs):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 4
    elif (
        model == "LM"
        and batch_size == 40
        and scale_factor == 4
        and num_epochs > 61
    ):
        for epoch in range(61, num_epochs):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
    elif model in ["Transformer", "CycleGAN", "A3C"]:
        return regular_mode_bs
    elif (
        model == "Recommendation"
        and batch_size == 512
        and scale_factor == 1
        and num_epochs > 21
    ):
        for epoch in range(21, 41):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
        for epoch in range(41, 71):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 4
        for epoch in range(71, 91):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 8
        for epoch in range(91, num_epochs):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 16
    elif (
        model == "Recommendation"
        and batch_size == 1024
        and scale_factor == 1
        and num_epochs > 21
    ):
        for epoch in range(21, 51):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
        for epoch in range(51, 91):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 4
        for epoch in range(91, num_epochs):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 8
    elif (
        model == "Recommendation"
        and batch_size == 2048
        and scale_factor == 1
        and num_epochs > 21
    ):
        for epoch in range(21, 41):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
        for epoch in range(41, num_epochs):
            if epoch + 1 >= num_epochs:
                break
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 4
    elif (
        model == "Recommendation"
        and batch_size == 4096
        and scale_factor == 1
        and num_epochs > 41
    ):
        for epoch in range(41, num_epochs):
            regular_mode_bs[epoch] = regular_mode_bs[epoch] * 2
            if epoch + 1 >= num_epochs:
                break
    for epoch in range(0, num_epochs):
        if regular_mode_bs[epoch] > bs_limit[model]:
            regular_mode_bs[epoch] = bs_limit[model]
    return regular_mode_bs


def generate_pickle_file(trace_file, throughput_file):
    """Uses parse_trace to return lists of jobs & arrival_times,
    generate pickle file with more metadata for use in shockwave 
    scheduler and the calculation of finish time fairness

    Args:
        trace_file (str): File name of .trace file used by Gavel
        throughput_file (str): File name of .json throughput file

    Returns:
        list, list: Lists of jobs/arrival_times (same as parse_trace)
    """
    parsed_throughputs = read_all_throughputs_json_v2(throughput_file)
    jobs, arrival_times = parse_trace(trace_file)
    shockwave_job_metadata = []

    max_prob = []
    static_fraction = []

    for ijob, job in enumerate(jobs):
        job_type = job.job_type
        total_steps = job.total_steps
        scale_factor = job.scale_factor
        model = job.model
        dataset = model_dataset_mapping[model]
        batch_size = job.batch_size
        len_dataset = dataset_len[dataset]
        num_epochs = math.ceil(
            total_steps / math.ceil(len_dataset / batch_size)
        )
        num_samples_per_epoch = len_dataset
        if job.mode == "static":
            bs_every_epoch = [batch_size] * num_epochs
        elif job.mode == "accordion":
            bs_every_epoch = get_accordion_bs_pattern(
                job_type, batch_size, num_epochs, ijob
            )
        elif job.mode == "gns":
            bs_every_epoch = get_gns_bs_pattern(
                job_type, batch_size, num_epochs, scale_factor
            )

        bs_pmf = Counter(bs_every_epoch)
        for k, v in bs_pmf.items():
            bs_pmf[k] = round(v / len(bs_every_epoch), 5)
        print(
            f"Batch size probability mass for job {ijob} (model: {model}, adaptation mode: {job.mode}):{dict(bs_pmf)}"
        )

        max_prob.append(max([x for x in dict(bs_pmf).values()]))
        if len(dict(bs_pmf).keys()) == 1:
            static_fraction.append(0)
        else:
            static_fraction.append(
                (1 - max([x for x in dict(bs_pmf).values()]))
                / (len(dict(bs_pmf).keys()) - 1)
            )

        # if sum(bs_pmf.values()) != 1: print("oof")

        mem_every_epoch = [
            get_metric(
                model, bs, metric="mem", parsed_throughputs=parsed_throughputs
            )
            for bs in bs_every_epoch
        ]
        util_every_epoch = [
            get_metric(
                model, bs, metric="util", parsed_throughputs=parsed_throughputs
            )
            for bs in bs_every_epoch
        ]
        duration_every_epoch = [
            get_metric(
                model,
                bs,
                metric="duration",
                scale_factor=int(scale_factor),
                parsed_throughputs=parsed_throughputs,
            )
            for bs in bs_every_epoch
        ]

        # FIXME
        # slowdown_factor = np.random.normal(loc=1.0, scale=1)
        # duration_every_epoch = [x * slowdown_factor for x in duration_every_epoch]

        new_metadata = {
            "model": model,
            "dataset": dataset,
            "num_epochs": num_epochs,
            "num_samples_per_epoch": num_samples_per_epoch,
            "bs_every_epoch": bs_every_epoch,
            "mem_every_epoch": mem_every_epoch,
            "util_every_epoch": util_every_epoch,
            "duration_every_epoch": duration_every_epoch,
            "scale_factor": scale_factor,
            "duration": job.duration,
        }
        shockwave_job_metadata.append(new_metadata)

    trace_name = os.path.splitext(os.path.basename(trace_file))[0]
    full_filepath = os.path.join(
        os.path.dirname(trace_file), f"{trace_name}.pickle"
    )
    with open(full_filepath, "wb") as dump_file:
        pickle.dump(shockwave_job_metadata, dump_file)

    # TODO: clean these up
    print(f"max_prob is {max_prob}")
    print(f"static_fraction is {static_fraction}")

    return jobs, arrival_times


def parse_trace(trace_file):
    jobs = []
    arrival_times = []
    shockwave_job_metadata = []
    with open(trace_file, "r") as f:
        for line in f:
            # if len(line.split('\t')) == 11:
            #     (job_type, command, working_directory, num_steps_arg,
            #     needs_data_dir, total_steps, scale_factor, mode, priority_weight, SLO,
            #     arrival_time) = line.split('\t')
            # elif len(line.split('\t')) == 10:
            #     (job_type, command, working_directory, num_steps_arg,
            #     needs_data_dir, total_steps, scale_factor, priority_weight, SLO,
            #     arrival_time) = line.split('\t')
            #     mode = "static"

            (
                job_type,
                command,
                working_directory,
                num_steps_arg,
                needs_data_dir,
                total_steps,
                scale_factor,
                mode,
                priority_weight,
                SLO,
                duration,
                arrival_time,
            ) = line.split("\t")

            assert int(scale_factor) >= 1
            jobs.append(
                Job(
                    job_id=None,
                    job_type=job_type,
                    command=command,
                    working_directory=working_directory,
                    needs_data_dir=bool(int(needs_data_dir)),
                    num_steps_arg=num_steps_arg,
                    total_steps=int(total_steps),
                    # duration=None,
                    duration=duration,
                    scale_factor=int(scale_factor),
                    mode=mode,
                    priority_weight=float(priority_weight),
                    SLO=float(SLO),
                )
            )
            arrival_times.append(float(arrival_time))

    return jobs, arrival_times


def print_allocation(allocation, current_time=None):
    """Prints the allocation.

       Debug method used for printing the allocation of each job on each
       worker type.
    """
    print("=" * 80)
    if current_time is not None:
        print("Allocation\t(Current_time: %f)" % (current_time))
        print("-" * 80)
    for job_id in sorted(list(allocation.keys())):
        allocation_str = "Job ID %s:" % (job_id)
        for worker_type in sorted(list(allocation[job_id].keys())):
            value = allocation[job_id][worker_type]
            allocation_str += " [%s: %f]" % (worker_type, value)
        print(allocation_str)
    print("=" * 80)
