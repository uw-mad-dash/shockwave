import json
import numpy as np

with open("./actual_throughputs.json", "rb") as f:
    throughputs = json.load(f)

np.random.seed(0)

for job_type in throughputs["v100"].keys():
    orig_throughput = throughputs["v100"][job_type]["null"]
    noise = np.random.normal(loc=1.0, scale=0.1)
    noisy_throughput = orig_throughput * noise
    print(
        f"{job_type}: changing throughput from {orig_throughput} to {noisy_throughput}"
    )
    throughputs["v100"][job_type]["null"] = noisy_throughput

with open("./noisy_throuhgputs.json", "w") as f:
    json.dump(throughputs, f, indent=4)
