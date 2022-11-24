# Shockwave: Fair and Efficient Cluster Scheduling for Dynamic Adaptation in Machine Learning

This repository contains the source code implementation of the [NSDI '23](https://www.usenix.org/conference/nsdi23) paper [Shockwave: Fair and Efficient Cluster Scheduling for Dynamic Adaptation in Machine Learning](https://arxiv.org/abs/2210.00093).

We built our implementation atop [Gavel](https://github.com/stanford-futuredata/gavel), the open-sourced codebase of the [OSDI '20](https://www.usenix.org/conference/osdi20) paper [Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads](https://www.usenix.org/conference/osdi20/presentation/narayanan-deepak). We would like to thank the Gavel authors for open-sourcing their implementation!

## Release notes

Sep 2022: We have released the first version of Shockwave! Please see the documentation below to get started. In the upcoming months, we will gradually make the following updates:

* Add shell scripts and documentation for deploying Shockwave on a physical cluster
* Add shell scripts and documentation for more simulation experiments
* Add bibtex information and hyperlinks for the arXiv release
* Make cleanups to the Shockwave codebase for better readability
* Add plotting scripts

## Directory Structure

### `scheduler`
Code for the scheduler, including the scheduling mechanism and simulator (`scheduler.py`), implementations of scheduling policies (`policies/`), `GavelIterator` as a Python module, and a communication stack between the scheduler and workers that uses [gRPC](https://grpc.io/) (`runtime/`).

### `workloads`
Implementations of target workloads in PyTorch, including changes needed to integrate with the `GavelIterator`.

### `accordion_workloads` and `gns_workloads`
Workload scripts built on top of those in [`workloads`](workloads), with respective dynamic adaptation optimizations implemented, namely [Accordion](https://github.com/uw-mad-dash/Accordion) and [Gradient Noise Scale (GNS)](https://openai.com/blog/science-of-ai/).


## Setting up the Software Dependencies

Shockwave/Gavel is implemented in Python. We have tested Shockwave/Gavel on Ubuntu 18.04 with Python 3.6.9.
Python can be installed using [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

Required software dependencies can be installed using:

```bash
apt-get -y install cmake g++ gcc libnuma-dev make numactl zlib1g-dev
pip install -r scheduler/requirements.txt
cd scheduler; make
```

In addition to the software dependencies required to run [Gavel](https://github.com/stanford-futuredata/gavel), running Shockwave also requires the [Gurobi Optimizer](https://www.gurobi.com/). An academic license can be requested [here](https://www.gurobi.com/features/academic-named-user-license/). Note that you might need to connect to your university's network or use a VPN to download Gurobi. Please see the Gurobi website for more details.

## Getting Started

Gavel's policies (including Shockwave) and scheduling mechanism can be evaluated either in simulation or on a physical cluster.

To reproduce our canonical results in simulation in ~10 minutes, run [`scheduler/reproduce/tacc_32gpus.sh`](scheduler/reproduce/tacc_32gpus.sh). For detailed instructions on how to reproduce more results from the NSDI paper, see [EXPERIMENTS.md](EXPERIMENTS.md).


## References

```
@misc{zheng2022shockwave,
      title={Shockwave: Fair and Efficient Cluster Scheduling for Dynamic Adaptation in Machine Learning}, 
      author={Pengfei Zheng and Rui Pan and Tarannum Khan and Shivaram Venkataraman and Aditya Akella},
      year={2022},
      eprint={2210.00093},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```