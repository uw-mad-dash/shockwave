import os
import subprocess
import pandas as pd
from progressbar import progressbar

# put me under /adaptdl/simulator

workload_trace = "workloads-realistic/workload-6.csv"
temp_single_job_csv = "workloads-realistic/temp_single_job.csv"
temp_simulator_log = "temp.txt"

durations = []

with open(workload_trace, "r") as f:
    df = pd.read_csv(f)

# get job arrival timestamps
print(list(df["time"]))

# get job durations
for i in progressbar(range(df.shape[0])):
    # create a csv workload trace with a single job
    single_job_df = df.iloc[[i], :]
    # print(single_job_df)
    with open(temp_single_job_csv, "w") as f:
        single_job_df.to_csv(f, index=False)

    # run the job in pollux simulator
    command = f"python3 simulator.py --policy tiresias {temp_single_job_csv}"

    # get its exclusive run time
    with open(temp_simulator_log, "w") as f:
        subprocess.run(command, stdout=f, shell=True)
    with open(temp_simulator_log, "r") as f:
        second_last_line = f.readlines()[-1]
        durations.append(int(float(second_last_line[13:-1])))

os.remove(temp_simulator_log)
os.remove(temp_single_job_csv)
print(durations)
