import math


def parse_trace(trace_file):
    num_gpus = 24
    total_time = 0
    num_epochs_dict = {
        "ResNet-18": list(),  # cifar10
        "ResNet-50": list(),  # imagenet
        "Transformer": list(),  # multi30k
        "LM": list(),  # wikitext2
        "Recommendation": list(),  # ml-20m
        # "CycleGAN": None,         # monet2photo
    }
    with open(trace_file, "r") as f:
        for line in f:
            (
                job_type,
                command,
                working_directory,
                num_steps_arg,
                needs_data_dir,
                total_steps,
                scale_factor,
                priority_weight,
                SLO,
                arrival_time,
            ) = line.split("\t")
            assert int(scale_factor) >= 1
            total_steps = int(total_steps)
            batch_size = get_bs_from_command(command)
            num_epochs = get_num_epochs(job_type, batch_size, total_steps)
            model = job_type[: job_type.find(" ")]
            num_epochs_dict[model].append(num_epochs)
            total_time += num_epochs * get_per_epoch_time(job_type, batch_size)
    total_time /= num_gpus
    print(f"Total time on {num_gpus} GPUs is {round(total_time)}s")
    for model, num_epochs in num_epochs_dict.items():
        avg_num_epochs = sum(num_epochs) / len(num_epochs)
        print(f"Model: {model}, average num epochs: {avg_num_epochs}")
    return total_time


def get_bs_from_command(command):
    """Takes in a training command, extract and return the batch size
    """
    if "translation" in command or "imagenet" in command:
        second_last_space_index = command[: command.rfind(" ")].rfind(" ")
        last_space_index = command.rfind(" ")
        bs = int(command[second_last_space_index:last_space_index])
    else:
        bs = int(command[(command.rfind(" ") + 1) :])
    return bs


def get_num_epochs(job_type, batch_size, num_steps):
    """Takes in the specifications of a job, calculate and 
    return the number of total epochs
    """
    model = job_type[: job_type.find(" ")]
    dataset_size_dict = {
        "ResNet-18": 50000,  # cifar10
        "ResNet-50": 100000,  # imagenet
        "Transformer": 10000,  # multi30k
        "LM": 59675,  # wikitext2
        "Recommendation": 117907,  # ml-20m
        # "CycleGAN": None,         # monet2photo
    }
    dataset_size = dataset_size_dict[model]
    return math.ceil(num_steps / math.ceil(dataset_size / batch_size))


def get_per_epoch_time(job_type, batch_size):
    """Returns the pre-profiled per-epoch training time of a job
    on a V100 GPU
    """
    # TODO: Add more jobs
    model = job_type[: job_type.find(" ")]
    time_per_epoch = {
        "ResNet-18": {16: 42, 32: 65, 64: 53, 128: 45, 256: 41,},  # ???
        "ResNet-50": {16: 367, 32: 320, 64: 299, 128: 288,},
        "Transformer": {16: 60.16, 32: 49.88, 64: 41.59, 128: 36.67,},
        "LM": {5: 74.47, 10: 44.96, 20: 32.84, 40: 24.97, 80: 19.26,},
        "Recommendation": {
            512: 0.93,
            1024: 0.67,
            2048: 0.61,
            4096: 0.59,
            8192: 0.64,
        },
        # "CycleGAN": {
        # },
    }
    return time_per_epoch[model][batch_size]


def main():
    filename = "/home/ruipan/gavel/scheduler/traces/2hours.trace"
    # filename = "/home/ruipan/gavel/scheduler/traces/test.trace"
    parse_trace(filename)


if __name__ == "__main__":
    main()
