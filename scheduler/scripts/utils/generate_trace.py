import os, sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    )
)

import argparse
import math
import numpy as np
import random
from collections import OrderedDict

from job import Job
from job_table import JobTable
import utils
import json

pollux_arrival_time = [
    53,
    154,
    326,
    368,
    735,
    1315,
    1585,
    1817,
    1916,
    1924,
    2438,
    2518,
    2535,
    2735,
    2850,
    3073,
    3107,
    3403,
    3472,
    3687,
    3733,
    4003,
    4007,
    4078,
    4101,
    4948,
    4949,
    4955,
    5172,
    5277,
    5285,
    6136,
    6324,
    6406,
    6731,
    6897,
    7752,
    8287,
    8444,
    8532,
    8614,
    9226,
    9550,
    9612,
    9732,
    9876,
    10151,
    10318,
    10319,
    11033,
    11256,
    11288,
    11292,
    11446,
    11491,
    11515,
    11852,
    12204,
    12558,
    12866,
    12938,
    14331,
    14377,
    14377,
    14708,
    14817,
    15115,
    15563,
    15879,
    16110,
    16319,
    16331,
    16437,
    16560,
    16585,
    16925,
    16946,
    16957,
    17490,
    17515,
    17628,
    18010,
    18159,
    18289,
    18391,
    18510,
    18699,
    18780,
    18785,
    18806,
    18877,
    19165,
    19202,
    19239,
    19306,
    19316,
    19332,
    19335,
    19393,
    19414,
    19450,
    19689,
    20051,
    20103,
    20139,
    20269,
    20313,
    20463,
    20574,
    20613,
    20643,
    20665,
    20669,
    20703,
    20726,
    20851,
    21008,
    21017,
    21061,
    21157,
    21190,
    21209,
    21433,
    21686,
    21706,
    21738,
    21776,
    21835,
    21886,
    22332,
    22769,
    22783,
    23032,
    23040,
    23136,
    23293,
    23478,
    23900,
    24040,
    24132,
    24362,
    24755,
    24780,
    24855,
    24869,
    25125,
    25129,
    25679,
    25875,
    25897,
    25898,
    25952,
    26080,
    26243,
    26853,
    26876,
    27132,
    27142,
    27229,
    28189,
]
pollux_arrival_time = [x / 5 for x in pollux_arrival_time]
pollux_job_duration = [
    4037,
    4056,
    19504,
    4082,
    13689,
    4035,
    1505,
    1513,
    4034,
    1526,
    13666,
    4032,
    1515,
    13669,
    1500,
    4077,
    4043,
    1487,
    79878,
    4063,
    4077,
    4047,
    4043,
    1472,
    4069,
    1502,
    1501,
    4055,
    4078,
    4033,
    4085,
    4074,
    1506,
    19484,
    1519,
    1473,
    4078,
    4083,
    4046,
    1518,
    4056,
    14312,
    33828,
    26220,
    1518,
    1494,
    13693,
    1472,
    1471,
    1477,
    13668,
    19522,
    13692,
    4044,
    19499,
    4035,
    19498,
    1506,
    1512,
    4064,
    4052,
    4039,
    4053,
    4053,
    3669,
    4033,
    1475,
    1507,
    4051,
    4060,
    1471,
    4079,
    1473,
    1470,
    1505,
    4085,
    13678,
    4053,
    13674,
    1475,
    19482,
    79920,
    4051,
    1481,
    1499,
    1230,
    19491,
    1470,
    676,
    1504,
    4053,
    19505,
    1528,
    19491,
    4044,
    4034,
    1518,
    1515,
    4077,
    13670,
    1520,
    13695,
    4079,
    19527,
    4051,
    1211,
    4057,
    4087,
    19476,
    3644,
    4087,
    4065,
    1501,
    1527,
    4064,
    1499,
    1522,
    4073,
    4089,
    6207,
    4080,
    4061,
    1517,
    1504,
    4044,
    4072,
    4034,
    26177,
    4044,
    4078,
    13675,
    19487,
    4038,
    1470,
    4054,
    19517,
    1512,
    79910,
    1490,
    4078,
    1528,
    1495,
    1470,
    4075,
    1501,
    4045,
    4041,
    4031,
    4075,
    1493,
    1492,
    13672,
    19490,
    19507,
    1497,
    14302,
    1518,
    13682,
    1481,
    4041,
]
pollux_job_duration = [
    (x if x < 20000 else 20000) for x in pollux_job_duration
]


def generate_interarrival_time(rng, lam):
    return -math.log(1.0 - rng.random()) * lam


def generate_duration(durations, rng):
    return round(3600 * rng.choice(durations))


def generate_mode(rng, mix):
    r = rng.uniform(0, 1)

    assert abs(sum(mix) - 1.0) <= 1e-3

    if r < mix[0]:
        return "static"
    elif mix[0] <= r < mix[0] + mix[1]:
        return "accordion"
    else:
        return "gns"


def generate_pollux_duration(durations, rng):
    """
    Pollux: 72% small jobs, 20% medium jobs, 6% large jobs, 2% extra large jobs
    Shockwave: 70% small jobs, 20% medium jobs, 10% large jobs
    """
    duration_prob = [0.72, 0.2, 0.05, 0.03]
    duration_boundary = [0.2, 0.5, 0.9, 1.0]

    # duration_prob = [0.72, 0.2, 0.06, 0.02]  # pengfei
    # # duration_prob = [0.8, 0.1, 0.07, 0.03]
    # duration_boundary = [0.2, 0.4, 0.8, 0.95]  # pengfei
    # # duration_boundary = [0.3, 0.4, 0.8, 0.95]

    num_durations = len(durations)
    num_small_durations = round(num_durations * duration_boundary[0])
    num_medium_durations = round(num_durations * duration_boundary[1])
    num_large_durations = round(num_durations * duration_boundary[2])

    r = rng.uniform(0, 1)

    if r < duration_prob[0]:
        durations = durations[:num_small_durations]
    elif duration_prob[0] <= r < sum(duration_prob[:2]):
        durations = durations[num_small_durations:num_medium_durations]
    elif sum(duration_prob[:2]) <= r < sum(duration_prob[:3]):
        durations = durations[num_medium_durations:num_large_durations]
    else:
        durations = durations[num_large_durations:]

    choice = np.random.choice(durations)

    # print(f"Invoking generate_pollux_duration with durations {durations}")
    return round(3600 * choice)


def generate_scale_factor(rng, mix):
    r = rng.uniform(0, 1)
    assert abs(sum(mix) - 1) <= 1e-3

    if r <= mix[0]:
        scale_factor = 1
    elif mix[0] < r <= sum(mix[:2]):
        scale_factor = 2
    elif sum(mix[:2]) < r <= sum(mix[:3]):
        scale_factor = 4
    else:
        scale_factor = 8
    return scale_factor


def construct_duration_space(min_duration, max_duration, nchoices, base=1.5):
    assert base > 1.0
    power_space = np.linspace(start=1, stop=nchoices, num=nchoices - 1)
    power_space = base ** power_space
    power_space = np.insert(arr=power_space, obj=0, values=0)
    power_space = power_space / np.max(power_space)
    space = np.round(
        power_space * (max_duration - min_duration) + min_duration, 2
    )
    return space


def main(args):
    np.random.seed(args.seed)

    job_generator = random.Random()
    job_generator.seed(args.seed)

    interarrival_time_generator = random.Random()
    interarrival_time_generator.seed(args.seed + 1)

    duration_generator = random.Random()
    duration_generator.seed(args.seed + 2)

    scale_factor_generator = random.Random()
    scale_factor_generator.seed(args.seed + 3)

    mode_generator = random.Random()
    mode_generator.seed(args.seed + 4)

    throughputs = utils.read_all_throughputs_json_v2(args.throughputs_file)

    if args.duration_logspace:
        print(f"Generating durations using logspace")
        durations = construct_duration_space(
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            nchoices=args.num_durations,
            base=1.5,
        )
    else:
        print(f"Generating durations using linspace")
        durations = np.linspace(
            args.min_duration, args.max_duration, args.num_durations
        )

    if not args.pollux_duration:
        duration_generator_func = lambda rng: generate_pollux_duration(
            durations, rng
        )
    else:
        duration_generator_func = pollux_job_duration

    prev_arrival_time = None

    temp_durations = []
    temp_arrivals = []

    with open(args.output_file, "w") as f:
        duration_count = {}
        scale_factor_count = {}
        mode_count = {}
        for i in range(args.num_jobs):
            while True:
                job = utils.generate_job(
                    throughputs=throughputs,
                    reference_worker_type="v100",
                    rng=job_generator,
                    job_id=None,
                    fixed_job_duration=None,
                    generate_multi_gpu_jobs=args.generate_multi_gpu_jobs,
                    generate_multi_priority_jobs=args.generate_multi_priority_jobs,
                    generate_dynamic_jobs=args.generate_dynamic_jobs,
                    scale_factor_generator_func=generate_scale_factor,
                    scale_factor_mix=args.scale_factor_mix,
                    duration_generator_func=duration_generator_func.pop(0),
                    mode_generator_func=generate_mode,
                    mode_mix=args.mode_mix,
                    single_mode=args.single_mode,
                    scale_factor_rng=scale_factor_generator,
                    duration_rng=duration_generator,
                    mode_rng=mode_generator,
                    always_generate_scale_factor=False,
                )
                model = job.model
                batch_size = job.batch_size
                total_steps = job.total_steps
                dataset_size_dict = {
                    "ResNet-18": 50000,  # cifar10
                    "ResNet-50": 100000,  # imagenet
                    "Transformer": 10000,  # multi30k
                    "LM": 59675,  # wikitext2
                    "Recommendation": 117907,  # ml-20m
                    "CycleGAN": 6287,  # monet2photo
                    "A3C": 4,  # no dataset
                }
                dataset_size = dataset_size_dict[model]
                iters_per_epoch = math.ceil(dataset_size / batch_size)
                num_epochs = math.ceil(total_steps / iters_per_epoch)
                # make sure all jobs last for at least 30 epochs
                # if num_epochs >= 30:
                #     print(f"Generated job {job.model} (bs = {job.batch_size}) with {num_epochs} epochs, duration {job.duration}")
                #     break
                print("-" * 50)
                print(
                    f"Generated job {job.model} (bs = {job.batch_size}) with {num_epochs} epochs, duration {job.duration}s, scale factor = {job.scale_factor}, mode is {job.mode}"
                )

                if job.duration not in duration_count.keys():
                    duration_count[job.duration] = 0
                duration_count[job.duration] += 1

                temp_durations.append(job.duration)

                if job.scale_factor not in scale_factor_count.keys():
                    scale_factor_count[job.scale_factor] = 0
                scale_factor_count[job.scale_factor] += 1

                if job.mode not in mode_count.keys():
                    mode_count[job.mode] = 0
                mode_count[job.mode] += 1

                break

            if prev_arrival_time is None:
                arrival_time = 0
            elif args.pollux_arrival:
                arrival_time = pollux_arrival_time.pop(0)
            elif args.lam > 0:
                interarrival_time = generate_interarrival_time(
                    interarrival_time_generator, args.lam
                )
                arrival_time = prev_arrival_time + interarrival_time

            temp_arrivals.append(arrival_time)

            prev_arrival_time = arrival_time
            f.write("%s\t%d\n" % (str(job), arrival_time))
            print(f"Job ID: {i}")
            print(f"Arrival Timestamp: {arrival_time}")
            print("Job Command: %s" % str(job))

    print("=" * 50)
    print(f"Finished trace generation. Statistics are as follows:")
    print(f"Duration count: {sorted(duration_count.items())}")
    print(f"Scale factor count: {sorted(scale_factor_count.items())}")
    print(f"Mode count: {sorted(mode_count.items())}")
    print(temp_durations)
    print(temp_arrivals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic trace")
    parser.add_argument(
        "--num_jobs",
        type=int,
        required=True,
        help="Number of jobs to generate",
    )
    parser.add_argument(
        "-l",
        "--lam",
        type=float,
        default=0.0,
        help="Lambda for Poisson arrival rate",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--throughputs_file",
        type=str,
        # default=('./ec2_throughputs.json'),
        help="Oracle throughputs file",
    )
    parser.add_argument(
        "-a",
        "--min_duration",
        type=float,
        default=1,
        help="Minimum job duration in hours",
    )
    parser.add_argument(
        "-b",
        "--max_duration",
        type=float,
        default=4,
        help="Maximum job duration in hours",
    )
    parser.add_argument(
        "-n",
        "--num_durations",
        type=int,
        default=4,
        help="Number of possible job durations",
    )
    parser.add_argument(
        "--duration_logspace",
        action="store_true",
        default=False,
        help=(
            "If set, use logspace for job duration distribution, instead of uniform distribution"
        ),
    )
    parser.add_argument(
        "-m",
        "--generate-multi-gpu-jobs",
        action="store_true",
        default=False,
        help=(
            "If set, generates multi-GPU jobs according to "
            "a pre-defined distribution"
        ),
    )
    parser.add_argument(
        "--generate-multi-priority-jobs",
        action="store_true",
        default=False,
        help=("If set, generates some jobs with higher priority"),
    )
    parser.add_argument(
        "-d",
        "--generate-dynamic-jobs",
        action="store_true",
        default=False,
        help=("If set, generates some jobs with accordion/gns mode"),
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Output file name"
    )
    parser.add_argument(
        "--single-mode",
        type=str,
        default=None,
        choices=["static", "accordion", "gns"],
        help=("If set, for all jobs, only use the specified dynamic mode"),
    )

    parser.add_argument(
        "--mode-mix",
        type=eval,
        default="0.333,0.333,0.333",
        help=(
            "A string dictionary to indicate the probabilistic mix of static, accordion and gns jobs."
        ),
    )
    parser.add_argument(
        "--scale-factor-mix",
        type=eval,
        default="0.6,0.3,0.09,0.01",
        help=(
            "A string dictionary to indicate the probabilistic mix of 1-GPU, 2-GPU, 4-GPU and 8-GPU jobs."
        ),
    )
    parser.add_argument(
        "--pollux-arrival",
        default=False,
        action="store_true",
        help="If set, use the arrival timestamps from the pollux trace",
    )
    parser.add_argument(
        "--pollux-duration",
        default=False,
        action="store_true",
        help="If set, use the job durations from the pollux trace",
    )
    # https://github.com/petuum/adaptdl/tree/osdi21-artifact/benchmark/workloads-realistic

    args = parser.parse_args()
    main(args)
