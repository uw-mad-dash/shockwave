import pickle
from collections import OrderedDict
import numpy as np
import os
import copy
import random

TRACE_FIELD = {
    "model": "model",
    "dataset": "dataset",
    "epoch_nsamples": "num_samples_per_epoch",
    "nepochs": "num_epochs",
    "epoch_gpu_req": "util_every_epoch",
    "epoch_mem_req": "mem_every_epoch",
    "epoch_duration": "duration_every_epoch",
    "nworkers": "scale_factor",
    "bs_schedule": "bs_every_epoch",
}

INFINITY = 1e9


def job_metadata_constructor(
    jobids: list, profiles_pickle, overclock: int = 1.0
) -> OrderedDict:
    assert os.path.isfile(profiles_pickle)
    assert profiles_pickle.lower().endswith(".pickle")
    profiles = pickle.load(open(profiles_pickle, "rb"))
    njobs = len(jobids)
    assert njobs <= len(profiles) and njobs > 0

    metadata = OrderedDict()
    for jobid, profile in zip(jobids, profiles):
        job_metadata = JobMetaData(
            id=jobid, profile=profile, overclock=overclock,
        )
        metadata[jobid] = job_metadata
    return metadata


class JobMetaData(object):
    def __init__(
        self, id, profile, overclock: int = 1.0,
    ):

        self.job_profile = profile
        assert type(self.job_profile) == dict and self.job_profile != {}

        self.jobid = id

        self.model_name = self.job_profile[TRACE_FIELD["model"]]
        self.dataset_name = self.job_profile[TRACE_FIELD["dataset"]]
        self.jobname = "ID_{}_{}_{}".format(
            id, self.model_name, self.dataset_name
        )

        if TRACE_FIELD["nworkers"] in self.job_profile.keys():
            self.nworkers = int(self.job_profile[TRACE_FIELD["nworkers"]])
        else:
            self.nworkers = 1

        self.epochs = self.job_profile[TRACE_FIELD["nepochs"]]
        assert self.epochs > 0
        self.epoch_nsamples = self.job_profile[TRACE_FIELD["epoch_nsamples"]]

        self.epoch_gpu_req = self.job_profile[TRACE_FIELD["epoch_gpu_req"]]
        assert len(self.epoch_gpu_req) == self.epochs

        self.epoch_gram_req = self.job_profile[TRACE_FIELD["epoch_mem_req"]]
        self.convert_RAM_Mb2Gb()
        assert len(self.epoch_gram_req) == self.epochs

        self.epoch_duration = self.job_profile[TRACE_FIELD["epoch_duration"]]
        self.convert_epoch_duration_in_seconds()
        self.epoch_duration_overclock(overclock)
        assert len(self.epoch_duration) == self.epochs

        self.epoch_duration_preprofiled = copy.deepcopy(self.epoch_duration)

        self.bs_schedule = self.job_profile[TRACE_FIELD["bs_schedule"]]
        assert len(self.bs_schedule) == self.epochs
        self.metadata_bs_schedule()

        self.throughput_measurements = None

        self.epoch_progress = 0
        self.epoch_tick = 0

        self.mem_alloc = None
        self.mps_alloc = None

        self.timestamp_submit = None
        # self.timestamp_start = None
        self.timestamp_completion = None

        self.waiting_delay = 0

        self.scaling_flag = False

    def convert_RAM_Mb2Gb(self):
        self.epoch_gram_req = [
            round(gram_usage / 1024.0, 1) for gram_usage in self.epoch_gram_req
        ]

    def convert_epoch_duration_in_seconds(self):
        self.epoch_duration = [
            max(1.0, round(duration)) for duration in self.epoch_duration
        ]

    def epoch_duration_overclock(self, clock_rate=1.0):
        self.epoch_duration = [
            max(1.0, duration / float(clock_rate))
            for duration in self.epoch_duration
        ]

    def mps_alloc_util(self, mps_alloc, mem_alloc):
        mps_req = self.mps_req()
        mem_req = self.mem_req()
        if mem_alloc < mem_req:
            print(
                "[Warning] JobID {} fails to run with insufficient GRAM allocation. GRAM requirement: {}, GRAM allocation: {} .".format(
                    self.jobid, mem_req, mem_alloc
                )
            )
            return 0.0
        assert mps_req > 0
        mps_util = min(1.0, mps_alloc / mps_req)
        return mps_util

    def progress(self, mps_alloc=None, mem_alloc=None, tol=1e-1):
        if self.epoch_progress >= self.epochs:
            print("[Error:] Executing completed job {}".format(self.jobid))
            return

        epoch_duration = self.metadata_epoch_duration()
        alloc_tick = self.mps_alloc_util(
            mps_alloc=mps_alloc, mem_alloc=mem_alloc
        )
        if alloc_tick is None:
            print(
                "[Warning] Job {} has null progress with (MPS,GRAM) allocation ({},{}).".format(
                    self.jobid, mps_alloc, mem_alloc
                )
            )
        self.epoch_tick += alloc_tick

        if self.epoch_tick >= epoch_duration:
            self.epoch_progress += 1
            self.epoch_tick = 0

            if self.epoch_progress > 0 and self.epoch_progress < self.epochs:
                if not self.scaling_flag:
                    mpsreq_prev = self.mps_req(self.epoch_progress - 1)
                    memreq_prev = self.mem_req(self.epoch_progress - 1)
                    if (
                        np.abs(mpsreq_prev - self.mps_req()) > tol
                        or np.abs(memreq_prev - self.mem_req()) > tol
                    ):
                        self.set_scaling_flag()
                else:
                    self.clear_scaling_flag()

    def set_epoch_progress(self, progress):
        assert progress >= 0 and progress <= self.epochs
        self.epoch_progress = progress

    def add_waiting_delay(self, delay):
        self.waiting_delay += delay

    def reset_waiting_delay(self):
        self.waiting_delay = 0

    def set_scaling_flag(self):
        self.scaling_flag = True

    def clear_scaling_flag(self):
        self.scaling_flag = False

    def mps_req(self, epoch_progress=None):
        if epoch_progress is None:
            epoch_progress = self.epoch_progress
        if not (epoch_progress < self.epochs and epoch_progress >= 0):
            raise ValueError(
                "JobID: {}, Inquiry Epoch: {}, Num. Epochs: {}".format(
                    self.jobid, epoch_progress, self.epochs
                )
            )
        return self.epoch_gpu_req[epoch_progress]

    def mem_req(self, epoch_progress=None):
        if epoch_progress is None:
            epoch_progress = self.epoch_progress
        if not (epoch_progress < self.epochs and epoch_progress >= 0):
            raise ValueError(
                "JobID: {}, Inquiry Epoch: {}, Num. Epochs: {}".format(
                    self.jobid, epoch_progress, self.epochs
                )
            )
        return self.epoch_gram_req[epoch_progress]

    def metadata_epoch_duration(self, epoch_progress=None):
        if epoch_progress is None:
            epoch_progress = self.epoch_progress
        if not (epoch_progress < self.epochs and epoch_progress >= 0):
            raise ValueError(
                "JobID: {}, Inquiry Epoch: {}, Num. Epochs: {}".format(
                    self.jobid, epoch_progress, self.epochs
                )
            )
        return self.epoch_duration[epoch_progress]

    def register_job_submit(self, time):
        if self.timestamp_submit is None:
            self.timestamp_submit = time

    def register_job_completion(self, time):
        if self.timestamp_completion is None:
            self.timestamp_completion = time

    def set_throughput_measurments(self, measurements, gavel_round_duration):
        assert type(measurements) == OrderedDict
        self.throughput_measurements = measurements
        self.gavel_round_duration = gavel_round_duration

    def calibrate_profiled_epoch_duration(self):
        # FIXME: temoporary hack on 30-job trace
        # return

        assert self.throughput_measurements is not None
        if len(self.throughput_measurements) <= 0:
            return
        assert self.bs_schedule is not None

        assert self.gavel_round_duration is not None
        timeline = sorted(list(self.throughput_measurements.keys()))
        prev_round = 0
        measured_nsamples = 0
        for cur_round in timeline:
            measured_throughput = self.throughput_measurements[cur_round][0]
            measured_bs = self.throughput_measurements[cur_round][1]
            measured_niters = (
                measured_throughput
                * self.gavel_round_duration
                * (cur_round - prev_round)
            )
            measured_nsamples += measured_bs * measured_niters
            prev_round = cur_round
        end_round = max(timeline)
        measured_time_range = self.gavel_round_duration * end_round

        preprofiled_time_range = 0
        preprofiled_nsamples = 0
        for iepoch, duration in enumerate(self.epoch_duration_preprofiled):
            if preprofiled_time_range + duration > measured_time_range:
                break
            else:
                preprofiled_time_range += duration
                preprofiled_nsamples += self.epoch_nsamples

        in_epoch_deficit = measured_time_range - preprofiled_time_range
        if in_epoch_deficit > 0:
            epoch_duration = self.epoch_duration[iepoch]
            if in_epoch_deficit > self.epoch_nsamples:
                print(
                    f"Warning: Job {self.jobid}, In Epoch Deficit: {in_epoch_deficit}, Epoch Duration: {epoch_duration}"
                )
            preprofiled_nsamples += (
                self.epoch_nsamples * in_epoch_deficit / epoch_duration
            )

        if (
            measured_nsamples <= 0
            or preprofiled_nsamples <= 0
            or abs(measured_nsamples - preprofiled_nsamples)
            / (preprofiled_nsamples)
            <= 0.4
        ):
            # print(f"Info: Preprofiled throughput for {self.jobid} is accurate.")
            return
        else:
            amp_factor = preprofiled_nsamples / measured_nsamples
            # print(f"Warning: Preprofiled throughput for {self.jobid} is calibrated with factor {amp_factor}.\n")
            for iepoch in range(len(self.epoch_duration)):
                self.epoch_duration[iepoch] = (
                    self.epoch_duration_preprofiled[iepoch] * amp_factor
                )

        return

    def metadata_bs_schedule(self):
        assert self.bs_schedule is not None
        assert self.epochs is not None
        assert self.epoch_duration is not None
        assert len(self.bs_schedule) == len(self.epoch_duration) == self.epochs

        self.bs_modes = sorted(list(set(self.bs_schedule)))
        self.bs_dirichlet_prior = {
            bs: self.epochs / len(self.bs_modes) for bs in self.bs_modes
        }

    def get_bs_epoch_duration_map(self):
        self.calibrate_profiled_epoch_duration()
        bs_epoch_duration_map = {}
        for iepoch, duration in enumerate(self.epoch_duration):
            bs = self.bs_schedule[iepoch]
            if bs not in bs_epoch_duration_map:
                bs_epoch_duration_map[bs] = []
            bs_epoch_duration_map[bs].append(duration)
        for bs in bs_epoch_duration_map.keys():
            mean_duration = np.mean(bs_epoch_duration_map[bs])
            assert mean_duration > 0 and mean_duration < INFINITY
            bs_epoch_duration_map[bs] = mean_duration
        return bs_epoch_duration_map

    def dirichlet_posterior_remaining_runtime(
        self, progress=None, oracle=False, noise_level=0.0
    ):
        if progress is None:
            progress = self.epoch_progress
        assert progress >= 0 and progress <= self.epochs

        if oracle:
            return sum(self.epoch_duration[self.epoch_progress :])

        observed_bs_schedule = self.bs_schedule[: progress + 1]
        bs_dirichlet_posterior = copy.deepcopy(self.bs_dirichlet_prior)
        for bs in observed_bs_schedule:
            bs_dirichlet_posterior[bs] += 1

        concentration_sum = sum(list(bs_dirichlet_posterior.values()))
        bs_dirichlet_rebased = {
            bs: self.epochs * concentration / concentration_sum
            for bs, concentration in bs_dirichlet_posterior.items()
        }

        for bs in observed_bs_schedule:
            if bs_dirichlet_rebased[bs] >= 1:
                bs_dirichlet_rebased[bs] -= 1
            else:
                pass
                # print(f"Warning: Job {self.jobid} - Dirichlet rebased underestimates num. epochs for BS={bs}.")

        if len(bs_dirichlet_rebased) <= 0:
            return 1.0

        inflated_remaining_epochs = int(
            sum(list(bs_dirichlet_rebased.values())) + 1
        )
        remaining_epochs = self.epochs - self.epoch_progress
        if inflated_remaining_epochs < remaining_epochs:
            inflated_remaining_epochs = remaining_epochs
        if inflated_remaining_epochs <= 0 or remaining_epochs <= 0:
            return 1.0

        bs_epoch_duration_map = self.get_bs_epoch_duration_map()
        remaining_runtime = 0.0
        for bs in bs_dirichlet_rebased.keys():
            remaining_epochs_bs = bs_dirichlet_rebased[bs]
            remaining_runtime += (
                remaining_epochs_bs * bs_epoch_duration_map[bs]
            )

        remaining_runtime *= remaining_epochs / inflated_remaining_epochs

        remaining_runtime *= 1.0 + random.choice([1, -1]) * noise_level

        if noise_level >= 1.0:
            remaining_runtime = max(remaining_runtime, 1)

        return remaining_runtime
