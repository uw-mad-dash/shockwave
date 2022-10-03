import os
import re
import numpy as np
from datetime import datetime

root_dir = os.getcwd()

"""
place me in the same directory as the one Gavel jobs use as shared checkpoint
"""


def analyze_file(filename):
    with open(filename) as f:
        lines = f.readlines()
        filtered = list(filter(lambda x: "CHECKPOINT" in x, lines))
        # print(filtered)
        timestamps = []
        for line in filtered:
            timestamp = re.match("\[(.*?)\]", line).group(0)[1:-1]
            # print(timestamp)
            datetime_object = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            timestamps.append(datetime_object)
        if len(filtered) == 2:
            # initial round, no reading from checkpoint
            return None, timestamps[1] - timestamps[0]
        elif len(filtered) == 4:
            # read from & write to checkpoint
            return timestamps[1] - timestamps[0], timestamps[3] - timestamps[2]
        elif len(filtered) == 0:
            # distributed jobs, only 1 worker write to checkpoint
            return None, None


load_times = []
save_times = []

for job_dir in os.listdir(root_dir):
    if os.path.isdir(job_dir):
        # print(job_dir)
        job_id = job_dir[job_dir.find("=") + 1 :]
        # print(root_dir, job_dir, '.gavel')
        rounds_dir = os.path.join(root_dir, job_dir, ".gavel")
        # print(rounds_dir)
        for round_dir in os.listdir(rounds_dir):
            round_dir = os.path.join(rounds_dir, round_dir)
            if os.path.isdir(round_dir):
                round_id = round_dir[round_dir.rfind("=") + 1 :]
                # print(round_dir)
                for log_file in os.listdir(round_dir):
                    if log_file.endswith(".log"):
                        log_file = os.path.join(round_dir, log_file)
                        worker_id = log_file[log_file.rfind("=") + 1 :]
                        # print(log_file)
                        load_time, save_time = analyze_file(log_file)
                        # print(load_time, save_time)
                        if load_time is not None:
                            load_times.append(load_time)
                        if save_time is not None:
                            save_times.append(save_time)

load_times = [x.total_seconds() for x in load_times if type(x) is not int]
save_times = [x.total_seconds() for x in save_times if type(x) is not int]


print(len(load_times), len(save_times))
print(max(load_times), max(save_times))
print(sum(load_times), sum(save_times))
print(np.mean(load_times), np.mean(save_times))


print(np.percentile(np.array(load_times), [0, 25, 50, 75, 100]))
print(np.percentile(np.array(save_times), [0, 25, 50, 75, 100]))

print(np.percentile(np.array(load_times), list(range(101))))
print(np.percentile(np.array(save_times), list(range(101))))
