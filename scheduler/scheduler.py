from __future__ import print_function

import collections
import copy
import faulthandler
import heapq
import numpy as np
import os
import pickle

# from preconditions import preconditions
import queue
import scipy
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import datetime
import random
import sched
import math
import multiprocessing
import logging
from collections import OrderedDict
from copy import deepcopy

import traceback

from job import Job
import job_id_pair
from job_table import JobTable
from runtime.rpc import scheduler_server, scheduler_client
import set_queue
from custom_logging import SchedulerAdapter
from throughput_estimator import ThroughputEstimator
import utils

from shockwave import ShockwaveScheduler
from JobMetaData import JobMetaData
import pathlib

""" Constants """
# Port for scheduler server.
SCHEDULER_PORT = 50070
# Proxy for infinity.
INFINITY = int(1e9)
# Default job throughput.
DEFAULT_THROUGHPUT = 1
# Default number of steps in each iteration.
DEFAULT_NUM_STEPS = 100
# Alpha parameter for exponential moving average.
EMA_ALPHA = 0.5
# Maximum number of times a job is allowed to fail before being dropped.
MAX_FAILED_ATTEMPTS = 5
# Fraction of the round to wait for before re-computing the schedule.
SCHEDULE_RECOMPUTE_FRACTION = 0.5

# Format string for logging.
LOG_FORMAT = "{name}:{levelname} {message}"
# Buffer time for jobs to complete.
JOB_COMPLETION_BUFFER_TIME = 60
# Base port to use for distributed jobs.
BASE_JOB_PORT = 60570
# Maximum port number.
MAX_PORT = 65535
# Shockwave: Allow jobs to ask for initial leases at most
# 3 seconds before a round starts
EARLY_INIT_THRESHOLD = 3.0

# Shockwave fixed hyperparameters
SEED = 0
REOPT_ROUNDS = 8  # TODO: move this into the config json file?

dataset_size_dict = {
    "ResNet-18": 50000,  # cifar10
    "ResNet-50": 100000,  # imagenet
    "Transformer": 10000,  # multi30k
    "LM": 59675,  # wikitext2
    "Recommendation": 117907,  # ml-20m
    "CycleGAN": 6287,  # monet2photo
    "A3C": 4,  # no dataset
}


class Scheduler:

    # TODO: Make assign_SLOs a configurable parameter from scripts.
    def __init__(
        self,
        policy,
        simulate=False,
        throughputs_file=None,
        seed=0,
        time_per_iteration=360,
        profiling_percentage=1.0,
        num_reference_models=len(JobTable),
        per_instance_type_prices_dir=None,
        available_clouds=[],
        assign_SLOs=False,
        enable_global_queue=False,
        expected_num_workers=None,
        minimum_time_between_allocation_resets=1000,
        max_rounds=None,
        pickle_file=None,
        shockwave_config=None,
        log_level="INFO",
    ):
        # Flag to control whether scheduler runs in simulation mode.
        self._simulate = simulate

        # Initial timestamp.
        if self._simulate:
            self._start_timestamp = 0
        else:
            self._start_timestamp = time.time()
        # Latest simulated timestamp.
        self._current_timestamp = self._start_timestamp

        # Configure logger.
        logger = logging.getLogger(__name__)
        logging_level_dict = {"INFO": logging.INFO, "DEBUG": logging.DEBUG}
        logger.setLevel(logging_level_dict[log_level])
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT, style="{"))
        logger.addHandler(ch)

        logger.addHandler(
            logging.FileHandler(
                pathlib.Path(__file__).parent.parent.joinpath(
                    f"console_output_{policy.name}.txt"
                ),
                mode="w",
            )
        )

        self._orig_logger = logger
        self._logger = SchedulerAdapter(
            logger,
            {"scheduler": self, "start_timestamp": datetime.datetime.now()},
        )
        self._logging_handler = ch

        # Print config information.
        if simulate:
            loc = "in simulation"
        else:
            loc = "at {addr}:{port}".format(
                addr=utils.get_ip_address(), port=SCHEDULER_PORT
            )
        self._logger.info(
            "Running scheduler {loc} with the following args: "
            "policy={policy}, seed={seed}, "
            "time_per_iteration={time_per_iteration}, "
            "profiling_percentage={profiling_percentage}, "
            "num_reference_models={num_reference_models}".format(
                loc=loc,
                policy=policy.name,
                seed=seed,
                time_per_iteration=time_per_iteration,
                profiling_percentage=profiling_percentage,
                num_reference_models=num_reference_models,
            )
        )

        # Initialize seeds.
        self._initialize_seeds(seed)
        # Initialize time in seconds each iteration should run for.
        self._time_per_iteration = time_per_iteration

        # Sets whether to use a global queue across all worker types.
        self._enable_global_queue = enable_global_queue

        self._expected_num_workers = expected_num_workers
        self._minimum_time_between_allocation_resets = (
            minimum_time_between_allocation_resets
        )

        # Start and last processed timestamp for each job_id.
        self._per_job_start_timestamps = {}
        self._per_job_latest_timestamps = {}
        # Job completion times.
        self._job_completion_times = {}
        # Job priority weights.
        self._job_priority_weights = {}
        # Queue of events that need to be processed at specific timestamps.
        self._event_queue = []

        # List of worker IDs.
        self._worker_ids = []
        # List of worker types.
        self._worker_types = set()
        # Mapping of worker ID to worker type, and worker type to worker ID.
        self._worker_id_to_worker_type_mapping = {}
        self._worker_type_to_worker_id_mapping = {}
        # Policy instance.
        self._policy = policy
        # Should jobs be packed.
        self._job_packing = "Packing" in policy.name
        # RPC clients.
        self._cluster_spec = {}
        self._worker_connections = {}
        # Next job_id to assign.
        self._job_id_counter = 0
        # Next worker_id to assign.
        self._worker_id_counter = 0
        # Synchronization primitives to ensure thread-safe updates of
        # scheduler metadata.
        self._scheduler_lock = threading.Lock()
        self._scheduler_cv = threading.Condition(self._scheduler_lock)
        # List of available worker IDs.
        self._available_worker_ids = set_queue.SetQueue()
        # Allocations for all current incomplete applications.
        self._allocation = {}
        # Map from job combinations to assigned workers for current round.
        self._current_worker_assignments = collections.OrderedDict()
        # Map from job combinations to assigned workers for the upcoming round.
        self._next_worker_assignments = None
        # Map from job combinations to assigned workers for jobs that need to
        # be re-dispatched on account of finishing early.
        self._redispatched_worker_assignments = collections.OrderedDict()
        # Set of completed jobs in current round.
        self._completed_jobs_in_current_round = set()
        # Set of jobs with an extended lease for the upcoming round.
        self._jobs_with_extended_lease = set()
        # The total number of lease extensions across all jobs.
        self._num_lease_extensions = 0
        # The total number of instances where leasees could have been extended.
        self._num_lease_extension_opportunities = 0
        # Event scheduler to trigger round completions for jobs with
        # extended leases.
        self._completion_event_scheduler = sched.scheduler(
            time.time, time.sleep
        )
        # Map from job ID to completion event.
        self._completion_events = {}
        # Map from job ID to timeline of events.
        self._job_timelines = {}
        # Port offset for distributed jobs.
        self._port_offset = 0
        # Iterations run on each worker_id, for all current incomplete
        # applications.
        self._steps_run_so_far = {}
        # Total number of iterations run for each incomplete job across
        # all worker types.
        self._total_steps_run = {}
        # Time run so far on each worker_id, for all current incomplete
        # applications.
        # _job_time_so_far keeps track of how long job_id has run on
        # worker_type since the last reset event.
        self._job_time_so_far = {}
        # Total cost of each job so far.
        self._job_cost_so_far = {}
        # Time spent running any application on each worker, for all current
        # incomplete applications.
        self._worker_time_so_far = {}
        # Cumulative time spent running any application on each worker.
        self._cumulative_worker_time_so_far = {}
        # Number of jobs to compute fair share.
        self._num_jobs = 0
        # Commands to run for all current incomplete applications.
        self._jobs = {}
        # Priority queues for each worker_type.
        self._priorities = {}
        self._deficits = {}
        # Number of failures per job.
        self._num_failures_per_job = {}
        # Timestamp when data structures recording elapsed time was last reset.
        self._last_reset_time = 0
        self._init_timestamp = time.time()
        # Flag indicating when to update the allocation.
        self._need_to_update_allocation = False
        # Flag indicating whether allocation has been updated since elapsed
        # time was last reset.
        self._allocation_changed_since_last_time_reset = False
        # Measured and predicted throughputs for all current incomplete
        # applications.
        self._throughputs = {}
        # Throughputs measured with respect to job types rather than
        # individual jobs.
        self._job_type_throughputs = {}
        # Map from job ID to application.
        self._job_id_to_job_type = {}
        # Map from application to set of job IDs.
        self._job_type_to_job_ids = {}
        # Throughputs for all job types (pre-measured).
        if throughputs_file is not None:
            self._oracle_throughputs = utils.read_all_throughputs_json_v2(
                throughputs_file
            )
        else:
            self._oracle_throughputs = None
        # Flag to indicate whether throughputs should be estimated online.
        self._estimate_throughputs = self._job_packing and (
            profiling_percentage < 1 or num_reference_models < len(JobTable)
        )
        if self._estimate_throughputs:
            self._throughput_estimator = self._initialize_throughput_estimator(
                seed + 4, num_reference_models, profiling_percentage
            )
            self._reference_throughputs = (
                self._throughput_estimator.get_reference_throughputs()
            )
            self._reference_job_map = {}
        if per_instance_type_prices_dir is not None:
            self._per_instance_type_spot_prices = utils.read_per_instance_type_spot_prices_json(
                per_instance_type_prices_dir
            )
            self._per_worker_type_prices = {}
            self._available_clouds = set(available_clouds)
            if assign_SLOs:
                self._SLOs = {}
            else:
                self._SLOs = None
        else:
            self._SLOs = None
            self._per_instance_type_spot_prices = None
            self._per_worker_type_prices = None
        # The per-round maximum number of steps to run for distributed jobs.
        # Indexed by single job IDs.
        self._max_steps = {}
        # All per-round lease update requests for distributed jobs.
        # Indexed by single job IDs.
        self._lease_update_requests = {}
        # List of all RPC clients.
        self._all_rpc_clients = []
        # Currently running jobs.
        self._running_jobs = set()
        # The timestamp when each worker entered the cluster.
        self._worker_start_times = {}
        # Verbose flag.
        self._verbose = False
        # Data structures for debugging.
        self._micro_tasks_per_job = {}
        self._all_jobs = []
        # In-progress updates for distributed jobs.
        self._in_progress_updates = {}
        # Set of completed job IDs.
        self._completed_jobs = set()
        # Maximum number of rounds.
        self._max_rounds = max_rounds
        # Number of completed rounds.
        self._num_completed_rounds = 0
        # Shockwave: for each job, indicate if there is a pending resource requirement update request
        # if big_bs/small_bs is True, double/half the batch size in the next round
        # key: job_id, value: {'big_bs': False, 'small_bs': False}}
        self._bs_flags = {}
        # Shockwave: maintains the mapping of job id - original batch size
        # key: job_id, value: int
        self._original_bs = {}
        # Shockwave: maintains the mapping of job id - original number of iterations
        # key: job_id, value: int
        self._original_num_steps = {}
        # Shockwave: maintains the mapping of job id - job type
        # key: job_id, value: string (e.g., "ResNet-18 (batch size 32)")
        self._job_types = {}
        # Shockwave: emergency patch for the sosp deadline, FIXME: can we remove this?
        # self._jobs_with_extended_lease_and_bs_request = []

        # shockwave: maintains a list of integer job_ids that are run in the current round
        self._scheduled_jobs_in_current_round = None
        # shockwave: in physical expr, also maintain a list of jobs scheduled in the previous round
        # that needs their epoch progress updated
        self._scheduled_jobs_in_prev_round = None

        # shockwave: if true, indicates that a recomputation is needed
        self._shockwave_job_completed_flag = False

        # shockwave: in simulator, use this to maintain the number
        # of rounds without a resolve
        self._iround_reopt = 0

        # Shockwave: for plotting/ftf calculation purposes, record the scheduling decision of each round
        self._per_round_schedule = []
        self._num_jobs_in_curr_round = (
            []
        )  # number of jobs in the cluster in a round
        self._job_start_round = {}
        self._job_end_round = {}
        self._num_jobs_in_trace = (
            0  # for calculating ftf using the static contention factor
        )

        # Shockwave: for computing envy freeness, record for each job the number
        # of rounds scheduled and queued/preempted
        self._num_scheduled_rounds = OrderedDict()
        self._num_queued_rounds = OrderedDict()

        # job_id: {round_id: throughput}. Note that round_id is the index of the round in which a lease ends
        self._throughput_timeline = {}

        # Shockwave: Keep track of the per-job cumulative training time on each worker
        # job_id: worker_id: execution_time
        self._cumulative_run_time = {}

        """
        self._tacc_preemption_overhead_distribution = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 
            4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 
            7.0, 7.0, 7.0, 7.0, 8.0, 9.0, 9.0, 9.0, 9.0, 9.0, 10.0, 10.0, 11.0, 12.0, 
            12.0, 13.0, 13.0, 13.0, 14.0, 15.0, 16.0, 16.0, 16.0, 18.0, 18.0, 19.0, 19.0, 
            21.0, 21.0, 22.559999999999945, 23.0, 24.0, 26.0, 26.0, 27.0, 29.0, 29.0, 31.0, 
            32.0, 35.0, 36.0, 38.0, 40.0, 43.0, 47.0, 60.0]
        """

        if self._policy.name == "shockwave":
            # create shockwave scheduler/solver with empty metadata
            self._shockwave_scheduler = ShockwaveScheduler(
                ngpus=shockwave_config["num_gpus"],
                gram=shockwave_config["gpu_ram"],
                init_metadata=OrderedDict(),
                future_nrounds=shockwave_config["future_rounds"],
                round_duration=shockwave_config["time_per_iteration"],
                solver_preference=["GUROBI",],
                solver_rel_gap=shockwave_config["solver_rel_gap"],
                solver_num_threads=shockwave_config["solver_num_threads"],
                solver_timeout=shockwave_config["solver_timeout"],
                n_epoch_vars_max=max(shockwave_config["future_rounds"], 30),
                logapx_bases=shockwave_config["log_approximation_bases"],
                logapx_origin={0.0: 1e-6},
                k=shockwave_config["k"],
                lam=shockwave_config["lambda"],
                rhomax=shockwave_config["rhomax"],
            )
            """
            NOTE: In physical experiments, jobs with extended leases do not update
            their steps_run_so_far by calling done_callback, as they invoke
            done_callback_extended_lease instead.
            Our workaround is that whenever a lease gets extended, 
            we keep track of for each job, the num of steps run so far under the 
            current lease.
            """
        else:
            self._shockwave_scheduler = None

        self._steps_run_in_current_lease = {}

        with open(pickle_file, "rb") as f:
            self._profiles = pickle.load(f)
        self._logger.info(f"Pickle file contains {len(self._profiles)} jobs")

        port = SCHEDULER_PORT
        callbacks = {
            "RegisterWorker": self._register_worker_callback,
            "InitJob": self._init_job_callback,
            "UpdateLease": self._update_lease_callback,
            "Done": self._done_callback,
            "UpdateResourceRequirement": self._update_resource_requirement_callback,
        }

        if not self._simulate:
            faulthandler.enable()
            f = open(".stack_trace.log", "w")
            faulthandler.dump_traceback_later(
                30, repeat=True, file=f, exit=False
            )

            if self._policy.name != "shockwave":
                self._allocation_thread = threading.Thread(
                    target=self._allocation_thread
                )
                self._allocation_thread.daemon = True
                self._allocation_thread.start()

            self.server_thread = threading.Thread(
                target=scheduler_server.serve, args=(port, callbacks)
            )
            self.server_thread.daemon = True
            self.server_thread.start()

            self._mechanism_thread = threading.Thread(
                target=self._schedule_with_rounds
            )
            self._mechanism_thread.daemon = True
            self._mechanism_thread.start()

            # if "shockwave" in policy.name:
            if False:
                self._periodic_reallocation_thread = threading.Thread(
                    target=self._periodic_reallocation_thread
                )
                self._periodic_reallocation_thread.daemon = True
                self._periodic_reallocation_thread.start()

        self.per_round_allocation = {}

    def _periodic_reallocation_thread(self):
        period_length = 740
        while True:
            # Shockwave: manually force recomputation of allocation every ? seconds
            time.sleep(period_length)
            with self._scheduler_lock:
                self._need_to_update_allocation = True
                self._logger.info(
                    f"Periodic reallocation ({period_length}s) triggered, updating allocation"
                )
                self._scheduler_cv.notifyAll()

    def _initialize_seeds(self, seed):
        np.random.seed(seed)
        random.seed(seed + 1)

        self._job_generator = random.Random()
        self._job_generator.seed(seed + 2)

        self._interarrival_time_generator = random.Random()
        self._interarrival_time_generator.seed(seed + 3)

        self._worker_type_shuffler = random.Random()
        self._worker_type_shuffler.seed(seed + 5)

        self._SLO_generator = random.Random()
        self._SLO_generator.seed(seed + 6)

    def _initialize_throughput_estimator(
        self, seed, num_reference_models, profiling_percentage
    ):
        worker_types = []
        for worker_type in self._oracle_throughputs:
            if "unconsolidated" not in worker_type:
                worker_types.append(worker_type)
        worker_types.sort()
        job_types = [(job_template.model, 1) for job_template in JobTable]
        return ThroughputEstimator(
            self._oracle_throughputs,
            worker_types,
            job_types,
            num_reference_models,
            profiling_percentage,
            seed,
        )

    def _update_per_worker_type_prices(self):
        assert self._per_worker_type_prices is not None
        current_time = self.get_current_timestamp(in_seconds=True)
        for worker_type in self._per_worker_type_prices:
            latest_price = utils.get_latest_price_for_worker_type(
                worker_type,
                current_time,
                self._per_instance_type_spot_prices,
                self._available_clouds,
            )
            if self._per_worker_type_prices[worker_type] != latest_price:
                self._per_worker_type_prices[worker_type] = latest_price
                self._scheduler_cv.acquire()
                self._need_to_update_allocation = True
                self._scheduler_cv.notifyAll()
                self._scheduler_cv.release()

    def _update_throughput(
        self, job_id, worker_type, all_num_steps, all_execution_times
    ):
        # Job might have already completed.
        if job_id not in self._throughputs:
            return

        # record the throughput over time for all jobs
        for i, single_job_id in enumerate(job_id.singletons()):
            # shockwave create throughput_timeline in add_job
            if single_job_id not in self._throughput_timeline.keys():
                self._throughput_timeline[single_job_id] = OrderedDict()

            current_round = self._num_completed_rounds
            if all_execution_times[i] <= 0:
                new_throughput = 0.0
            else:
                new_throughput = all_num_steps[i] / all_execution_times[i]
            bs_in_current_lease = self._jobs[job_id].batch_size
            self._throughput_timeline[single_job_id][current_round] = (
                new_throughput,
                bs_in_current_lease,
            )
            # self._logger.debug(f"Adding throughput measurement for job {single_job_id}: {new_throughput}")

        if self._simulate and self._estimate_throughputs:
            if not job_id.is_pair():
                # Assume single job throughputs are already populated.
                return
            else:
                oracle_throughputs = self._oracle_throughputs[worker_type]
                scale_factor = self._jobs[job_id.singletons()[0]].scale_factor
                job_types = []
                for single_job_id in job_id.singletons():
                    job_types.append(
                        (self._jobs[single_job_id].job_type, scale_factor)
                    )
                self._throughputs[job_id][worker_type] = oracle_throughputs[
                    job_types[0]
                ][job_types[1]]
        elif not self._simulate:
            # Adjust the job throughput using an exponential moving average
            # between the old value and the new measurement.
            if job_id.is_pair():
                old_throughput = copy.deepcopy(
                    self._throughputs[job_id][worker_type]
                )
            else:
                old_throughput = [self._throughputs[job_id][worker_type]]

            for i, single_job_id in enumerate(job_id.singletons()):
                if all_execution_times[i] <= 0:
                    new_throughput = 0
                else:
                    new_throughput = all_num_steps[i] / all_execution_times[i]
                if old_throughput != INFINITY:
                    new_throughput *= EMA_ALPHA
                    new_throughput += (1 - EMA_ALPHA) * old_throughput[i]
                if job_id.is_pair():
                    self._throughputs[job_id][worker_type][i] = new_throughput
                else:
                    self._throughputs[job_id][worker_type] = new_throughput
            # Manually set failed job pair throughputs to 0.
            if np.min(all_execution_times) <= 0:
                if job_id.is_pair():
                    self._throughputs[job_id][worker_type] = [0.0, 0.0]
            if job_id.is_pair():
                new_throughput = self._throughputs[job_id][worker_type]
            else:
                new_throughput = [self._throughputs[job_id][worker_type]]
            self._logger.info(
                "Job {job_id} throughput on worker type {worker_type}: "
                "{orig} -> {updated}".format(
                    job_id=job_id,
                    worker_type=worker_type,
                    orig=str(old_throughput),
                    updated=str(new_throughput),
                )
            )
            if not 0.8 < new_throughput[0] / old_throughput[0] < 1.25:
                self._logger.warning(
                    f"Job {job_id} had a big throughput change: new / old is {round(new_throughput[0] / old_throughput[0], 5)}"
                )
                if round(new_throughput[0] / old_throughput[0], 5) == 0.5:
                    self._logger.error(
                        f"Job {job_id}: Something might be wrong..."
                    )

    def _read_throughputs_for_job_type(self, job_type_key):
        """Reads oracle throughputs for passed in job type.

           Args:
             job_type_key: A tuple of (model, scale_factor).
        """
        self._job_type_throughputs[job_type_key] = {}
        other_job_type_keys = list(self._job_type_throughputs.keys())
        for worker_type in self._worker_types:
            oracle_throughputs = self._oracle_throughputs[worker_type]
            self._job_type_throughputs[job_type_key][worker_type] = {}
            self._job_type_throughputs[job_type_key][worker_type][
                None
            ] = oracle_throughputs[job_type_key]["null"]
            if self._job_packing:
                for other_job_type_key in other_job_type_keys:
                    # Don't store throughputs for jobs with different scale
                    # factors.
                    if other_job_type_key[1] != job_type_key[1]:
                        continue
                    colocated_throughputs = oracle_throughputs[job_type_key][
                        other_job_type_key
                    ]
                    self._job_type_throughputs[job_type_key][worker_type][
                        other_job_type_key
                    ] = colocated_throughputs[0]
                    self._job_type_throughputs[other_job_type_key][
                        worker_type
                    ][job_type_key] = colocated_throughputs[1]

    """
    ======================================================================
       Public-facing scheduler methods.
    ======================================================================
    """

    def add_job(self, job, timestamp=None):
        """Adds a new job to the scheduler.

        Enables users to schedule a new job. Updates the internal
        allocation of workers to jobs. An allocation is of the form
        {job: <fraction of allocations on different workers>}.

        Args:
            job: Job object to schedule. Contains information about the command
                 to run, as well as the number of steps to run the command for.
            timestamp (optional): Timestamp at which job is to be added
                                  (defaults to current_timestamp() if not
                                  specified).

        Returns:
            The job_id of the newly added job.
        """

        with self._scheduler_lock:
            current_timestamp = self.get_current_timestamp()
            job_id = job_id_pair.JobIdPair(self._job_id_counter, None)
            self._job_id_counter += 1
            job._job_id = job_id
            self._jobs[job_id] = job
            self._steps_run_so_far[job_id] = {}
            self._job_time_so_far[job_id] = {}
            self._job_cost_so_far[job_id] = 0.0
            self._job_timelines[job_id] = [[] for _ in range(job.scale_factor)]
            self._throughputs[job_id] = {}
            job_type = self._jobs[job_id].job_type
            scale_factor = job.scale_factor
            job_type_key = (job_type, scale_factor)
            self._job_id_to_job_type[job_id] = job_type_key
            # Shockwave additions
            self._original_bs[job_id] = self._jobs[job_id].batch_size
            self._original_num_steps[job_id] = job.total_steps
            self._job_types[job_id] = job.job_type
            self._num_jobs_in_trace += 1
            if self._estimate_throughputs:
                self._reference_job_map[
                    job_id
                ] = self._throughput_estimator.match_job_to_reference_job(
                    job_type_key
                )
            if job_type_key not in self._job_type_throughputs:
                self._job_type_to_job_ids[job_type_key] = set()
                self._read_throughputs_for_job_type(job_type_key)
            self._job_type_to_job_ids[job_type_key].add(job_id)
            self._num_failures_per_job[job_id] = 0
            self._total_steps_run[job_id] = 0
            self._cumulative_run_time[job_id] = {}
            if self._SLOs is not None:
                assert job.duration is not None
                assert job.SLO is not None
                self._SLOs[job_id] = (
                    job.SLO * job.duration
                    + self.get_current_timestamp(in_seconds=True)
                )
            for worker_type in self._worker_types:
                self._steps_run_so_far[job_id][worker_type] = 0
                self._set_initial_throughput(job_id, worker_type)
                if self._job_packing:
                    self._populate_job_combination_metadata(
                        job_id, worker_type
                    )
                self._job_time_so_far[job_id][worker_type] = (
                    self._time_per_iteration / 2.0
                )
            self._per_job_start_timestamps[job_id] = current_timestamp
            self._per_job_latest_timestamps[job_id] = None
            self._add_to_priorities(job_id)
            self._need_to_update_allocation = True
            self._bs_flags[job_id] = {"big_bs": False, "small_bs": False}
            self._num_scheduled_rounds[job_id] = 0
            self._num_queued_rounds[job_id] = 0
            self._job_start_round[
                job_id.integer_job_id()
            ] = self._num_completed_rounds

            # self._shockwave_scheduler.add_metadata(job_id, self._get_job_metadata(job))

            if self._policy.name == "shockwave":
                # construct JobMetaData, add to ShockwaveScheduler.metadata
                job_id = job_id.integer_job_id()
                metadata = JobMetaData(
                    job_id, self._profiles[job_id], overclock=1.0
                )

                if self._simulate:
                    metadata.register_job_submit(self.get_current_timestamp())
                else:
                    metadata.register_job_submit(
                        self.get_current_timestamp() - self._start_timestamp
                    )

                assert job_id not in self._throughput_timeline.keys()
                self._throughput_timeline[job_id] = OrderedDict()
                metadata.set_throughput_measurments(
                    measurements=self._throughput_timeline[job_id],
                    gavel_round_duration=self._shockwave_scheduler.round_duration,
                )
                self._logger.debug(
                    f"Throughput measurments OrderedDict for Job {job_id} constructed."
                )

                self._shockwave_scheduler.add_metadata(job_id, metadata)
                self._logger.debug(
                    f"Added metadata of job {job_id} to shockwave scheduler:"
                )
                # FIXME
                self._logger.debug(
                    f"Job {job_id}, model {self._profiles[job_id]['model']}, init bs {self._profiles[job_id]['bs_every_epoch'][0]}, {len(self._profiles[job_id]['bs_every_epoch'])} epochs"
                )

            # initialize data structure for bookkeeping of training progress
            self._steps_run_in_current_lease[job_id] = 0

            if timestamp is None:
                timestamp = self.get_current_timestamp()
            self._per_job_start_timestamps[job_id] = timestamp
            self._logger.info(
                "[Job dispatched]\tJob ID: {job_id}, run time duration: {duration}".format(
                    job_id=job_id, duration=job.duration
                )
            )
            self._scheduler_cv.notifyAll()

        return job_id

    def remove_job(self, job_id):
        """Public-facing interface to _remove_job."""
        with self._scheduler_lock:
            self._remove_job(job_id)
            self._scheduler_cv.notifyAll()

    def _remove_job(self, job_id):
        """Removes a job from the scheduler.

        Enables users to remove a previously scheduled job. Updates
        the internal allocation of workers to jobs.

        Args:
            job_id: The job_id of the job to remove.
        """

        if type(job_id) is int:
            job_id = job_id_pair.JobIdPair(job_id, None)
        self._completed_jobs.add(job_id)
        duration = (
            self._per_job_latest_timestamps[job_id]
            - self._per_job_start_timestamps[job_id]
        )
        self._job_priority_weights[job_id] = self._jobs[job_id].priority_weight
        job_type = self._jobs[job_id].job_type
        scale_factor = self._jobs[job_id].scale_factor
        job_type_key = (job_type, scale_factor)
        del self._jobs[job_id]
        if self._num_failures_per_job[job_id] >= MAX_FAILED_ATTEMPTS:
            # self._job_completion_times[job_id] = None
            self._job_completion_times[
                job_id
            ] = duration  # still record the completion time of a failed job
        else:
            self._job_completion_times[job_id] = duration
        job_type_key = self._job_id_to_job_type[job_id]
        self._job_type_to_job_ids[job_type_key].remove(job_id)
        del self._steps_run_so_far[job_id]
        del self._job_time_so_far[job_id]
        del self._throughputs[job_id]
        del self._job_id_to_job_type[job_id]
        del self._num_failures_per_job[job_id]
        # TODO: delete the corresponding entry in self._bs_flags
        self._job_end_round[
            job_id.integer_job_id()
        ] = self._num_completed_rounds  # FIXME
        if job_id in self._in_progress_updates:
            del self._in_progress_updates[job_id]
        if job_id in self._lease_update_requests:
            del self._lease_update_requests[job_id]
        if job_id in self._max_steps:
            del self._max_steps[job_id]
        if job_id in self._jobs_with_extended_lease:
            self._jobs_with_extended_lease.remove(job_id)
        if self._policy.name == "shockwave":
            if job_id in self._shockwave_scheduler.metadata.keys():
                num_epochs = self._shockwave_scheduler.metadata[job_id].epochs
                self._shockwave_scheduler.schedule_progress(job_id, num_epochs)
        del self._steps_run_in_current_lease[job_id]
        if self._job_packing:
            to_delete = []
            for other_job_id in self._throughputs:
                if other_job_id.is_pair() and job_id.overlaps_with(
                    other_job_id
                ):
                    to_delete.append(other_job_id)
            for other_job_id in to_delete:
                other_job_is_active = any(
                    [x in self._jobs for x in other_job_id.singletons()]
                )
                del self._throughputs[other_job_id]
                del self._job_time_so_far[other_job_id]
                if not other_job_is_active:
                    if other_job_id in self._in_progress_updates:
                        del self._in_progress_updates[other_job_id]
                    if other_job_id in self._lease_update_requests:
                        del self._lease_update_requests[other_job_id]
                    if other_job_id in self._max_steps:
                        del self._max_steps[other_job_id]
                    if other_job_id in self._jobs_with_extended_lease:
                        self._jobs_with_extended_lease.remove(other_job_id)

            if len(self._job_type_to_job_ids[job_type_key]) == 0:
                del self._job_type_to_job_ids[job_type_key]
                del self._job_type_throughputs[job_type_key]
                for other_job_type_key in self._job_type_throughputs:
                    for worker_type in self._job_type_throughputs[
                        other_job_type_key
                    ]:
                        if (
                            job_type_key
                            in self._job_type_throughputs[other_job_type_key][
                                worker_type
                            ]
                        ):
                            del self._job_type_throughputs[other_job_type_key][
                                worker_type
                            ][job_type_key]
        self._remove_from_priorities(job_id)
        # TODO: Add a flag to choose whether to update allocation here.
        # NOTE: Scheduler cv will be notified by calling function.
        self._need_to_update_allocation = True
        self._logger.info("Remaining active jobs: {0}".format(len(self._jobs)))

    def num_workers(self):
        """Returns the number of workers the scheduler is connected to."""

        n = 0
        with self._scheduler_lock:
            for worker_type in self._cluster_spec:
                n += self._cluster_spec[worker_type]
            return n

    def is_done(self, jobs_to_complete=None):
        """Returns whether the scheduler is done with all its assigned work."""
        with self._scheduler_lock:
            if (
                self._max_rounds is not None
                and self._num_completed_rounds >= self._max_rounds
            ):
                return True
            elif jobs_to_complete is not None:
                return jobs_to_complete.issubset(self._completed_jobs)
            else:
                return False

    def reset_workers(self):
        """Sends a shutdown signal to every worker and ends the scheduler."""
        with self._scheduler_lock:
            for i, rpc_client in enumerate(self._all_rpc_clients):
                rpc_client.reset()

    def shutdown(self):
        """Sends a shutdown signal to every worker and ends the scheduler."""
        if not self._simulate:
            with self._scheduler_lock:
                for rpc_client in self._all_rpc_clients:
                    rpc_client.shutdown()
        self._orig_logger.removeHandler(self._logging_handler)
        self._logging_handler.close()
        # TODO: Any other cleanup?

    def get_per_round_schedule(self):
        return self._per_round_schedule

    """
    ======================================================================
       Scheduler's main schedule() and simulate() methods.
    ======================================================================
    """

    def _get_state_snapshot(self, deepcopy=False):
        if deepcopy:
            state_snapshot = {
                "allocation": copy.deepcopy(self._allocation),
                "priorities": copy.deepcopy(self._priorities),
                "deficits": copy.deepcopy(self._deficits),
            }
        else:
            state_snapshot = {
                "allocation": self._allocation,
                "priorities": self._priorities,
                "deficits": self._deficits,
            }
        return state_snapshot

    def _print_schedule_summary(self, state_snapshot=None):
        if state_snapshot is not None:
            allocation = state_snapshot["allocation"]
            priorities = state_snapshot["priorities"]
            deficits = state_snapshot["deficits"]
        else:
            allocation = self._allocation
            priorities = self._priorities
            deficits = self._deficits

        completed_jobs = set()
        worker_types = sorted(self._cluster_spec.keys())
        for job_id, worker_ids in self._current_worker_assignments.items():
            worker_type = self._worker_id_to_worker_type_mapping[worker_ids[0]]
            if (
                job_id in self._completed_jobs_in_current_round
                or job_id not in allocation
                or job_id not in priorities[worker_type]
                or job_id not in deficits[worker_type]
            ):
                completed_jobs.add(job_id)

            if not self._simulate and job_id in completed_jobs:
                self._logger.debug(
                    "Job {job_id} has already completed on "
                    "{num_gpus} {worker_type} GPUs".format(
                        job_id=job_id,
                        num_gpus=len(worker_ids),
                        worker_type=worker_type,
                    )
                )
                continue
            allocation_str = ""
            for x in worker_types:
                allocation_str += " [%4s %.2f]" % (x, allocation[job_id][x])
            self._logger.info(
                "[Micro-task scheduled]\tJob ID: {job_id}\t"
                "Worker type: {worker_type}\tWorker ID(s): {worker_ids}\t"
                "Priority: {priority:.2f}\tDeficit: {deficit:.2f}\t"
                "Allocation: {allocation}".format(
                    job_id=job_id,
                    worker_type=worker_type,
                    worker_ids=",".join([str(x) for x in worker_ids]),
                    priority=priorities[worker_type][job_id],
                    deficit=deficits[worker_type][job_id],
                    allocation=allocation_str,
                )
            )
        num_workers_assigned = {}
        for job_id, worker_ids in self._current_worker_assignments.items():
            self._logger.debug(
                f"Job {job_id} is assigned workers {worker_ids}"
            )
            if not self._simulate and job_id in completed_jobs:
                continue
            worker_type = self._worker_id_to_worker_type_mapping[worker_ids[0]]
            if worker_type not in num_workers_assigned:
                num_workers_assigned[worker_type] = 0
            num_workers_assigned[worker_type] += len(worker_ids)
        for worker_type in worker_types:
            if worker_type not in num_workers_assigned:
                num_workers_assigned[worker_type] = 0
            if (
                num_workers_assigned[worker_type]
                < self._cluster_spec[worker_type]
            ):
                unused_workers = (
                    self._cluster_spec[worker_type]
                    - num_workers_assigned[worker_type]
                )
                # TODO: In shockwave, sometimes this warning pops up while jobs are scheduled later
                # therefore utilizing all GPUs. Try to disable the warning message in that case.
                self._logger.warn(
                    "{num_gpus} GPUs of type {worker_type} left unused. "
                    "Number of active jobs: {num_active_jobs}".format(
                        num_gpus=unused_workers,
                        worker_type=worker_type,
                        num_active_jobs=len(self._jobs),
                    )
                )

    def _assign_workers_to_job(
        self,
        job_id,
        scale_factor,
        worker_type,
        worker_state,
        worker_assignments,
    ):
        """Assign workers to jobs.

        Assigns workers in a strided fashion to minimize the number
        of servers used.

        Args:
          job_id: The job (combination) ID to schedule.
          scale_factor: The number of GPUs requested.
          worker_type: The worker type to allocate.
          worker_state: A dict comprised of the following information:
            worker_ids: Worker IDs organized into servers.
            assigned_worker_ids: The set of worker IDs assigned so far.
            server_id_ptr: The server to assign workers from.
          worker_assignments: A map from job_id to assigned worker_ids tuple.
        """

        # self._logger.debug(f"_assign_workers_to_job() is invoked using parameters {(job_id, scale_factor, worker_type, worker_state, worker_assignments)}")

        worker_ids = worker_state["worker_ids"]
        assigned_worker_ids = worker_state["assigned_worker_ids"]
        server_id_ptr = worker_state["server_id_ptr"]

        if job_id in worker_assignments:
            worker_ids_for_job = list(worker_assignments[job_id])
        else:
            worker_ids_for_job = []
        while len(worker_ids_for_job) < scale_factor and server_id_ptr < len(
            worker_ids
        ):
            if len(worker_ids[server_id_ptr]) == 0:
                server_id_ptr += 1
                continue
            worker_id_to_assign = worker_ids[server_id_ptr][0]
            if worker_id_to_assign not in assigned_worker_ids:
                worker_ids_for_job.append(worker_id_to_assign)
                assigned_worker_ids.add(worker_id_to_assign)
            worker_ids[server_id_ptr].pop(0)

        if len(worker_ids_for_job) != scale_factor:
            raise RuntimeError(
                "Could not assign workers to job %s!" % (job_id)
            )

        worker_assignments[job_id] = tuple(worker_ids_for_job)
        worker_state["server_id_ptr"] = server_id_ptr

        for single_job_id in job_id.singletons():
            if self._simulate:
                # This will be done on initialization when running on a
                # physical cluster.
                self._per_job_latest_timestamps[
                    single_job_id
                ] = self.get_current_timestamp()
                self._running_jobs.add(single_job_id)

    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
    def _schedule_jobs_on_workers_helper(self, worker_types):
        """Greedily selects the jobs to run in the next round by iterating
           through the job list in sorted priority order.

           Args:
             worker_types: An ordered list of worker types.

           Returns:
             A list of job IDs and associated scale factors to schedule for the
             upcoming round.
        """
        if self._policy.name == "shockwave":
            # # for each worker type, come up with a list of (job_id, scale_factor)
            # # example: {'v100': [(0, 1), (1, 1), (2, 1), (3, 1)]}

            scheduled_jobs = {"v100": []}

            # get the ids of the jobs that will be scheduled in the current round
            job_ids = self._shockwave_scheduler.round_schedule()
            self._scheduled_jobs_in_prev_round = (
                self._scheduled_jobs_in_current_round
            )
            self._scheduled_jobs_in_current_round = job_ids
            for job_id in job_ids:
                if job_id not in self._jobs:
                    self._logger.warning(
                        f"Job {job_id} has completed, yet it is in round_schedule {job_ids}"
                    )
                    continue
                scale_factor = self._jobs[job_id].scale_factor
                job_id = job_id_pair.JobIdPair(job_id, None)
                scheduled_jobs["v100"].append((job_id, scale_factor))
        else:  # vanilla Gavel
            already_scheduled_jobs = set()
            scheduled_jobs = {}

            num_workers_left = {}
            for worker_type in worker_types:
                scheduled_jobs[worker_type] = []
                num_workers = self._cluster_spec[worker_type]
                num_workers_left[worker_type] = num_workers

            sorted_job_queue = []
            for worker_type in worker_types:
                per_worker_type_entries = []
                for job_id in self._priorities[worker_type]:
                    allocation = 0.0
                    if (
                        self._allocation is not None
                        and job_id in self._allocation
                    ):
                        allocation = self._allocation[job_id][worker_type]
                    per_worker_type_entries.append(
                        (
                            job_id,
                            worker_type,
                            self._priorities[worker_type][job_id],
                            self._deficits[worker_type][job_id],
                            allocation,
                        )
                    )
                if not self._enable_global_queue:
                    # if (self._policy.name.startswith("MinTotal") or
                    #     self._policy.name.startswith("MaxSum")):
                    #     sorted_job_queue += sorted(per_worker_type_entries,
                    #                             key=lambda x: (x[2], x[3], x[4]),
                    #                             reverse=True)
                    # else:
                    #     sorted_job_queue += sorted(per_worker_type_entries,
                    #                             key=lambda x: (x[3], x[2], x[4]),
                    #                             reverse=True)

                    sorted_job_queue += sorted(
                        per_worker_type_entries,
                        key=lambda x: (x[2], x[3], x[4]),
                        reverse=True,
                    )

                else:
                    sorted_job_queue += per_worker_type_entries

            if self._enable_global_queue:
                sorted_job_queue.sort(
                    key=lambda x: (x[2], x[3], x[4]), reverse=True
                )

            for job_id, worker_type, *_ in sorted_job_queue:
                if num_workers_left[worker_type] == 0:
                    continue

                # Don't schedule jobs that have already been scheduled.
                if (
                    not job_id.is_pair() and job_id in already_scheduled_jobs
                ) or (
                    job_id.is_pair()
                    and (
                        job_id.singletons()[0] in already_scheduled_jobs
                        or job_id.singletons()[1] in already_scheduled_jobs
                    )
                ):
                    continue

                # Don't schedule jobs with 0 throughput.
                if (
                    job_id.is_pair()
                    and (
                        self._throughputs[job_id][worker_type][0] <= 0
                        or self._throughputs[job_id][worker_type][1] <= 0
                    )
                ) or (
                    not job_id.is_pair()
                    and self._throughputs[job_id][worker_type] <= 0
                ):
                    continue

                # For FIFO jobs, don't schedule jobs with 0 priority.
                if (
                    self._policy.name.startswith("FIFO")
                    and self._priorities[worker_type][job_id] <= 0.0
                ):
                    self._logger.error(
                        f"Job {job_id} has priority {self._priorities[worker_type][job_id]}, continuing"
                    )
                    continue

                # Make sure job fits in remaining number of workers.
                # If not, move onto next job.
                if job_id.is_pair():
                    scale_factor = self._jobs[
                        job_id.singletons()[0]
                    ].scale_factor
                    other_scale_factor = self._jobs[
                        job_id.singletons()[1]
                    ].scale_factor
                    # Only pack jobs with the same scale_factor.
                    if scale_factor != other_scale_factor:
                        continue
                else:
                    scale_factor = self._jobs[job_id].scale_factor

                if scale_factor > num_workers_left[worker_type]:
                    # use break for isolated_plus policy to strictly
                    # respect the priorities
                    if self._policy.name != "Isolated_plus":
                        continue
                    else:
                        break
                num_workers_left[worker_type] -= scale_factor

                for single_job_id in job_id.singletons():
                    already_scheduled_jobs.add(single_job_id)

                scheduled_jobs[worker_type].append((job_id, scale_factor))

        self._logger.debug(
            f"Scheduled jobs returned from _schedule_jobs_on_workers_helper is {scheduled_jobs}"
        )

        return scheduled_jobs

    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
    def _schedule_jobs_on_workers(self):
        """Attempts to schedule jobs on as many alive workers as possible.

           Returns:
             A list of job IDs and tuple of worker IDs for each scheduled job
             in the coming round.
        """
        if not self._policy.name == "shockwave":
            # Update priorities before trying to figure out applications to run
            # in the upcoming round.
            self._update_priorities()

        # self._update_priorities()

        to_remove = []
        worker_types = ["v100", "p100", "k80"]

        for i, worker_type in enumerate(worker_types):
            if worker_type not in self._worker_type_to_worker_id_mapping:
                to_remove.append(i)
        for i in reversed(to_remove):
            worker_types.pop(i)

        if (
            "Perf" not in self._policy.name
            and "Packing" not in self._policy.name
        ):
            self._worker_type_shuffler.shuffle(worker_types)

        new_worker_assignments = collections.OrderedDict()
        scheduled_jobs = self._schedule_jobs_on_workers_helper(worker_types)

        worker_state = {}
        for worker_type in worker_types:
            # Sort jobs by the scale factor: want to assign jobs from largest
            # to smallest to minimize fragmentation.
            scheduled_jobs[worker_type].sort(key=lambda x: x[1], reverse=True)
            worker_ids = copy.deepcopy(
                self._worker_type_to_worker_id_mapping[worker_type]
            )
            worker_state[worker_type] = {
                "worker_ids": worker_ids,
                "assigned_worker_ids": set(),
                "server_id_ptr": 0,
            }

        prev_worker_types = {}
        for (job_id, worker_ids) in self._current_worker_assignments.items():
            worker_type = self._worker_id_to_worker_type_mapping[worker_ids[0]]
            prev_worker_types[job_id] = worker_type

        for worker_type in worker_types:
            per_worker_state = worker_state[worker_type]
            assigned_worker_ids = per_worker_state["assigned_worker_ids"]
            current_job = 0
            scale_factors = set([x[1] for x in scheduled_jobs[worker_type]])
            scale_factors = sorted(scale_factors, reverse=True)

            # Assign workers in order of decreasing scale factor to prioritize
            # locality for multi-GPU jobs.
            for current_scale_factor in scale_factors:
                # Try to keep jobs on current workers if possible.
                for (job_id, scale_factor) in scheduled_jobs[worker_type]:
                    if scale_factor != current_scale_factor:
                        continue
                    if (
                        job_id in prev_worker_types
                        and prev_worker_types[job_id] == worker_type
                    ):
                        prev_worker_ids = self._current_worker_assignments[
                            job_id
                        ]
                        assert isinstance(prev_worker_ids, tuple)
                        extend_placement = True
                        for prev_worker_id in prev_worker_ids:
                            if prev_worker_id in assigned_worker_ids:
                                extend_placement = False
                                break
                        if extend_placement:
                            new_worker_assignments[job_id] = prev_worker_ids
                            for prev_worker_id in prev_worker_ids:
                                assigned_worker_ids.add(prev_worker_id)

                # Assign workers for remaining jobs.
                for (job_id, scale_factor) in scheduled_jobs[worker_type]:
                    if scale_factor != current_scale_factor:
                        continue
                    # in shockwave, self._allocation is an empty dict
                    elif (
                        self._policy.name != "shockwave"
                        and job_id not in self._allocation
                    ):
                        continue
                    # self._logger.debug(f"_assign_workers_to_job() is invoked using parameters {(job_id, scale_factor, worker_type, per_worker_state, new_worker_assignments)}")
                    self._assign_workers_to_job(
                        job_id,
                        scale_factor,
                        worker_type,
                        per_worker_state,
                        new_worker_assignments,
                    )
                    if self._policy.name == "shockwave":
                        # placeholder for use in _print_schedule_summary
                        self._allocation[job_id] = {}
                        self._allocation[job_id]["v100"] = -1.0

        # Verify the assignment.
        num_assignments = {}
        for job_id in new_worker_assignments:
            for worker_id in new_worker_assignments[job_id]:
                if worker_id not in num_assignments:
                    num_assignments[worker_id] = 0
                num_assignments[worker_id] += 1
        for worker_id in num_assignments:
            if num_assignments[worker_id] != 1:
                raise RuntimeError(
                    "Worker {0} was assigned {1} times!".format(
                        worker_id, num_assignments[worker_id]
                    )
                )

        self._logger.info(f"New worker assignments: {new_worker_assignments}")
        njobs_sched = 0
        ngpus_alloc = 0
        for jobid, assignment in new_worker_assignments.items():
            self._logger.debug(
                f"Job {jobid} scheduled on workers {assignment}"
            )
            njobs_sched += 1
            ngpus_alloc += len(assignment)

        # Shockwave: record the per-round schedule for plotting purposes
        assignments = deepcopy(new_worker_assignments)
        assignments = {
            job_id.integer_job_id(): worker_ids
            for job_id, worker_ids in assignments.items()
        }
        self._per_round_schedule.append(assignments)
        self._num_jobs_in_curr_round.append(len(self._jobs))

        # Shockwave: for each job, record if it is scheduled or halted
        active_jobs = [job_id.integer_job_id() for job_id in self._jobs.keys()]
        scheduled_jobs = assignments.keys()
        for job_id in active_jobs:
            if job_id in scheduled_jobs:
                self._num_scheduled_rounds[job_id] += 1
            else:
                self._num_queued_rounds[job_id] += 1

        return new_worker_assignments

    def _get_num_steps(self, job_id, worker_type, single_job_id=None):
        if self._simulate:
            oracle_throughputs = self._oracle_throughputs[worker_type]
            if job_id.is_pair():
                assert single_job_id is not None
                index = job_id.as_tuple().index(single_job_id[0])
                scale_factor = self._jobs[single_job_id].scale_factor
                job_types = []
                for x in job_id.singletons():
                    job_types.append((self._jobs[x].job_type, scale_factor))
                colocated_throughputs = oracle_throughputs[job_types[0]][
                    job_types[1]
                ]
                single_job_throughput = colocated_throughputs[index]
                num_steps = int(
                    single_job_throughput * self._time_per_iteration
                )
            else:
                # NOTE: Assumes oracle throughputs for single jobs.
                num_steps = int(
                    self._throughputs[job_id][worker_type]
                    * self._time_per_iteration
                )
        else:
            if job_id.is_pair():
                assert single_job_id is not None
                index = job_id.as_tuple().index(single_job_id[0])
                num_steps = int(
                    self._throughputs[job_id][worker_type][index]
                    * self._time_per_iteration
                )
            else:
                num_steps = int(
                    self._throughputs[job_id][worker_type]
                    * self._time_per_iteration
                )

        if single_job_id is not None:
            return min(num_steps, self._get_remaining_steps(single_job_id))
        else:
            return min(num_steps, self._get_remaining_steps(job_id))

    def _get_job_steps_and_finish_times(self, job_id, worker_type):
        """Returns the number of steps to execute and and latest finish time(s)
           for a job or job pair."""
        max_finish_time = self.get_current_timestamp()
        all_num_steps = []
        single_job_ids = job_id.singletons()
        if job_id.is_pair() and self._estimate_throughputs and self._simulate:
            oracle_throughputs = self._oracle_throughputs[worker_type]
            scale_factor = self._jobs[job_id.singletons()[0]].scale_factor
            job_types = []
            for single_job_id in single_job_ids:
                job_types.append(
                    (self._jobs[single_job_id].job_type, scale_factor)
                )
            oracle_throughput = oracle_throughputs[job_types[0]][job_types[1]]
        for i, single_job_id in enumerate(single_job_ids):
            num_steps = self._get_num_steps(job_id, worker_type, single_job_id)
            all_num_steps.append(num_steps)
            if job_id.is_pair():
                if self._estimate_throughputs and self._simulate:
                    throughput = oracle_throughput[i]
                else:
                    throughput = self._throughputs[job_id][worker_type][i]
            else:
                # NOTE: Assumes single job throughputs are accurate in
                # simulation + estimation case.
                throughput = self._throughputs[job_id][worker_type]
            if throughput <= 0:
                if self._estimate_throughputs:
                    all_num_steps.append(0)
                    finish_time = max_finish_time
                else:
                    print(single_job_id)
                    print(worker_type)
                    raise RuntimeError(
                        "Throughput for job {job_id} on "
                        "worker type {worker_type}"
                        "should not be less than 0!".format(
                            job_id=single_job_id, worker_type=worker_type
                        )
                    )
            else:
                execution_time = num_steps / throughput
                finish_time = self.get_current_timestamp() + (
                    num_steps / throughput
                )
            if finish_time > max_finish_time:
                max_finish_time = finish_time
            self._running_jobs.add(single_job_id)
        return all_num_steps, max_finish_time

    def _save_checkpoint(
        self,
        checkpoint_file,
        last_job_arrival_time,
        next_job_arrival_time,
        current_round_start_time,
        current_round_end_time,
        running_jobs,
    ):
        with open(checkpoint_file, "wb") as f:
            pickle.dump(self._completed_jobs, f)
            pickle.dump(last_job_arrival_time, f)
            pickle.dump(next_job_arrival_time, f)
            pickle.dump(current_round_start_time, f)
            pickle.dump(current_round_end_time, f)
            pickle.dump(running_jobs, f)

            pickle.dump(self._jobs, f)
            pickle.dump(self._throughputs, f)
            pickle.dump(self._allocation, f)
            pickle.dump(self._steps_run_so_far, f)
            pickle.dump(self._total_steps_run, f)
            pickle.dump(self._job_time_so_far, f)
            pickle.dump(self._worker_start_times, f)
            pickle.dump(self._worker_time_so_far, f)
            pickle.dump(self._cumulative_worker_time_so_far, f)
            pickle.dump(self._num_jobs, f)
            pickle.dump(self._priorities, f)
            pickle.dump(self._deficits, f)
            pickle.dump(self._last_reset_time, f)
            pickle.dump(self._need_to_update_allocation, f)
            pickle.dump(self._job_generator, f)
            pickle.dump(self._interarrival_time_generator, f)
            pickle.dump(self._per_job_start_timestamps, f)
            pickle.dump(self._per_job_latest_timestamps, f)
            pickle.dump(self._job_completion_times, f)
            pickle.dump(self._current_timestamp, f)
            pickle.dump(self._job_id_counter, f)

    def _load_checkpoint(self, checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            self._completed_jobs = pickle.load(f)
            last_job_arrival_time = pickle.load(f)
            next_job_arrival_time = pickle.load(f)
            current_round_start_time = pickle.load(f)
            current_round_end_time = pickle.load(f)
            running_jobs = pickle.load(f)

            self._jobs = pickle.load(f)
            self._throughputs = pickle.load(f)
            self._allocation = pickle.load(f)
            self._steps_run_so_far = pickle.load(f)
            self._total_steps_run = pickle.load(f)
            self._job_time_so_far = pickle.load(f)
            self._worker_start_times = pickle.load(f)
            self._worker_time_so_far = pickle.load(f)
            self._cumulative_worker_time_so_far = pickle.load(f)
            self._num_jobs = pickle.load(f)
            self._priorities = pickle.load(f)
            self._deficits = pickle.load(f)
            self._last_reset_time = pickle.load(f)
            self._need_to_update_allocation = pickle.load(f)
            self._job_generator = pickle.load(f)
            self._interarrival_time_generator = pickle.load(f)
            self._per_job_start_timestamps = pickle.load(f)
            self._per_job_latest_timestamps = pickle.load(f)
            self._job_completion_times = pickle.load(f)
            self._current_timestamp = pickle.load(f)
            self._job_id_counter = pickle.load(f)

            return (
                last_job_arrival_time,
                next_job_arrival_time,
                current_round_start_time,
                current_round_end_time,
                running_jobs,
            )

    def _sample_arrival_time_delta(self, rate_parameter):
        """Samples job interarrival rate from a Poisson distribution according
           to the specified rate parameter."""
        return (
            -math.log(1.0 - self._interarrival_time_generator.random())
            / rate_parameter
        )

    def _simulate_gns(self, job_id):
        with self._scheduler_lock:
            model = self._jobs[job_id].model
            job_type = self._jobs[job_id].job_type
            batch_size = self._jobs[job_id].batch_size
            original_batch_size = self._original_bs[job_id]
            # original_batch_size = self._jobs[job_id].original_bs
            total_steps_run = self._total_steps_run[job_id]
            current_epoch = self._get_num_epochs(
                job_type, batch_size, total_steps_run
            )
            scale_factor = self._jobs[job_id].scale_factor
            bs_gns = utils.get_gns_bs_pattern(
                job_type,
                original_batch_size,
                max(760, current_epoch + 2),
                scale_factor,
            )
            # self._logger.debug(f"[Job {job_id}] current epoch: {current_epoch} in gns")
            # print("batch size in gns is", batch_size)
            epoch_duration = (
                math.ceil(dataset_size_dict[model] / batch_size)
                / self._throughputs[job_id]["v100"]
            )
            # print("epoch_duration for job id , model ,batchsize and throughput is ", job_id, model, batch_size, self._throughputs[job_id]["v100"], epoch_duration)

            if (
                bs_gns[current_epoch + 1] > batch_size
                or bs_gns[current_epoch] > batch_size
            ):
                if not (
                    model == "ResNet-18"
                    and batch_size == 256
                    or model == "ResNet-50"
                    and batch_size == 128
                    or model == "Transformer"
                    and batch_size == 128
                    or model == "LM"
                    and batch_size == 80
                    or model == "Recommendation"
                    and batch_size == 8192
                ):
                    self._logger.debug(
                        f"[Job {job_id}] current epoch: {current_epoch}, doubling batch size in gns"
                    )
                    self._bs_flags[job_id]["big_bs"] = True
                    epoch_duration = (
                        math.ceil(dataset_size_dict[model] / batch_size)
                        / self._throughputs[job_id]["v100"]
                    )
                    # print("changed epoch_duration for job id , model ,batchsize and throughput is ", job_id, model, batch_size, self._throughputs[job_id]["v100"], epoch_duration)

            self._scheduler_cv.notifyAll()

    def _simulate_accordion(self, job_id):
        with self._scheduler_lock:
            model = self._jobs[job_id].model
            job_type = self._jobs[job_id].job_type
            batch_size = self._jobs[job_id].batch_size
            original_batch_size = self._original_bs[job_id]
            # original_batch_size = self._jobs[job_id].original_bs
            total_steps_run = self._total_steps_run[job_id]
            current_epoch = self._get_num_epochs(
                job_type, batch_size, total_steps_run
            )

            if model == "Transformer":
                # Accordion cannot be applied to transformer jobs for now
                return
            elif model == "LM":
                in_critical_regime = current_epoch < 10
            elif model == "Recommendation":
                if original_batch_size in [512, 1024]:
                    in_critical_regime = current_epoch < 30
                elif original_batch_size == 2048:
                    in_critical_regime = current_epoch < 40
                elif original_batch_size in [4096, 8192]:
                    in_critical_regime = current_epoch < 10
            elif model == "ResNet-50":
                in_critical_regime = (current_epoch % 30) < 10
            elif model == "ResNet-18":
                head_cr_len = 20 if original_batch_size == 256 else 10
                in_critical_regime = (
                    (0 <= current_epoch and current_epoch < head_cr_len)
                    or (150 <= current_epoch and current_epoch < 160)
                    or (250 <= current_epoch and current_epoch < 260)
                )

            if batch_size == original_batch_size and not in_critical_regime:
                if not (
                    model == "ResNet-18"
                    and batch_size == 256
                    or model == "ResNet-50"
                    and batch_size == 128
                    or model == "Transformer"
                    and batch_size == 128
                    or model == "LM"
                    and batch_size == 80
                    or model == "Recommendation"
                    and batch_size == 8192
                ):
                    self._logger.debug(
                        f"[Job {job_id}] current epoch: {current_epoch}, scaling up batch size"
                    )
                    self._bs_flags[job_id]["big_bs"] = True
            elif batch_size != original_batch_size and in_critical_regime:
                if not (
                    model == "ResNet-18"
                    and batch_size == 16
                    or model == "ResNet-50"
                    and batch_size == 16
                    or model == "Transformer"
                    and batch_size == 16
                    or model == "LM"
                    and batch_size == 5
                    or model == "Recommendation"
                    and batch_size == 512
                ):
                    self._logger.debug(
                        f"[Job {job_id}] current epoch: {current_epoch}, scaling down batch size"
                    )
                    self._bs_flags[job_id]["small_bs"] = True
            self._scheduler_cv.notifyAll()

    def simulate(
        self,
        cluster_spec,
        arrival_times=None,
        jobs=None,
        measure_steady_state_jobs=False,
        lam=None,
        jobs_to_complete=None,
        fixed_job_duration=None,
        num_total_jobs=None,
        generate_multi_gpu_jobs=False,
        generate_multi_priority_jobs=False,
        simulate_steady_state=False,
        debug=False,
        checkpoint_threshold=None,
        checkpoint_file=None,
        num_gpus_per_server=None,
        ideal=False,
        output_trace_file_name=None,
    ):
        """Simulates the scheduler execution.
           Simulation can be performed using a trace or with continuously
           generated synthetic data. Simulation is terminated when either
               1) All jobs in the specified trace complete.
               2) A specific subset of jobs complete.
               3) All jobs in a specific time window complete.
           Currently, the cluster specification must be statically
           specified from the beginning of execution.
           Args:
            cluster_spec: A dictionary of worker type to worker count.
            arrival_times: The arrival times of a set of pre-generated jobs.
            jobs: A set of pre-generated jobs.
            lam: 1 / the rate parameter to be passed in to the Poisson process
                 used to generate arrival times.
            jobs_to_complete: A set of `JobIdPair`s that must be completed
                              before terminating the simulation.
            fixed_job_duration: If set, all generated jobs will have this
                                duration if run exclusively on a v100.
            num_total_jobs: If set, only `num_total_jobs` jobs will
                            be generated.
            generate_multi_gpu_jobs: If set, some jobs will have `scale_factor`
                                     greater than 1, according to a pre-defined
                                     distribution.
            generate_multi_priority_jobs: If set, 20% of jobs will have a
                                          priority of 5.0.
            simulate_steady_state: If set, adds as many jobs as there are
                                   workers before beginning the simulation.
            debug: If set, pauses the simulation at the start of every loop.
        """
        from_trace = arrival_times is not None and jobs is not None
        if num_total_jobs is not None:
            remaining_jobs = num_total_jobs
        if from_trace:
            remaining_jobs = len(jobs)
            queued_jobs = []
        else:
            if self._oracle_throughputs is None:
                raise ValueError(
                    "Scheduler must be initialized with a " "throughputs file."
                )
            elif lam is None:
                raise ValueError(
                    "'lam' must be specified when running " "without trace."
                )
        if (
            not from_trace
            and jobs_to_complete is None
            and num_total_jobs is None
        ):
            raise ValueError(
                "One of 'jobs_to_complete' " "or 'num_total_jobs' must be set."
            )
        if checkpoint_file is not None and (
            from_trace or simulate_steady_state
        ):
            raise ValueError(
                "Checkpointing only intended to be used "
                "when generating trace on-the-fly."
            )

        if not from_trace and output_trace_file_name is not None:
            output_trace_file = open(output_trace_file_name, "w")
        else:
            output_trace_file = None

        running_jobs = []
        num_jobs_generated = 0
        last_job_arrival_time = None
        next_job_arrival_time = 0
        if arrival_times is not None and len(arrival_times) > 0:
            next_job_arrival_time = arrival_times[0]
        no_dispatched_or_running_jobs = False
        current_round_start_time = 0
        current_round_end_time = None
        window_start_time = None
        SLO_generator = self._SLO_generator if self._SLOs is not None else None

        # Set up the cluster according to the provided spec.
        worker_types = sorted([worker_type for worker_type in cluster_spec])
        for worker_type in worker_types:
            num_gpus = 1
            if num_gpus_per_server is not None:
                num_gpus = num_gpus_per_server[worker_type]
            for i in range(cluster_spec[worker_type] // num_gpus):
                self._register_worker_callback(worker_type, num_gpus=num_gpus)

        if checkpoint_file is not None and checkpoint_threshold is None:
            (
                last_job_arrival_time,
                next_job_arrival_time,
                current_round_start_time,
                current_round_end_time,
                running_jobs,
            ) = self._load_checkpoint(checkpoint_file)

        if from_trace:
            # Add all jobs to the queue.
            for i in range(1, len(arrival_times)):
                assert arrival_times[i] >= arrival_times[i - 1]

            for (arrival_time, job) in zip(arrival_times, jobs):
                queued_jobs.append((arrival_time, job))
            self._current_timestamp = arrival_times[0]
            # self._logger.info(f"self._current_timestamp is set to {self._current_timestamp}")
        elif simulate_steady_state:
            for worker_type in worker_types:
                num_remaining_workers = cluster_spec[worker_type]
                while num_remaining_workers > 0:
                    job = utils.generate_job(
                        throughputs=self._oracle_throughputs,
                        reference_worker_type="v100",
                        rng=self._job_generator,
                        job_id=None,
                        fixed_job_duration=fixed_job_duration,
                        generate_multi_gpu_jobs=generate_multi_gpu_jobs,
                        generate_multi_priority_jobs=generate_multi_priority_jobs,
                        SLO_rng=SLO_generator,
                    )
                    if (
                        jobs_to_complete is None
                        or window_start_time is not None
                    ) and output_trace_file is not None:
                        output_trace_file.write("%s\t%f\n" % (str(job), 0))
                    num_remaining_workers -= job.scale_factor
                    num_jobs_generated += 1
                    self._all_jobs.append((0, job))
                    job_id = self.add_job(job, timestamp=0)

        current_round = 0

        while True:
            self._logger.info("-" * 50)
            self._logger.info("*** START ROUND {0} ***".format(current_round))
            if debug:
                input("Press Enter to continue...")
            if jobs_to_complete is not None:
                num_completed_jobs = len(
                    jobs_to_complete.intersection(self._completed_jobs)
                )
                self._logger.info(
                    "Number of completed jobs: {0}".format(num_completed_jobs)
                )
                if self.is_done(jobs_to_complete):
                    break
            elif num_total_jobs is not None and remaining_jobs <= 0:
                break
            elif from_trace:
                if remaining_jobs == 0:
                    break
                elif len(queued_jobs) > 0:
                    next_job_arrival_time = queued_jobs[0][0]
                else:
                    next_job_arrival_time = None

            # Jump to the next event's timestamp.
            # Find the time when the latest job completes, which signals
            # the finishing of the round.
            max_timestamp = 0
            if len(running_jobs) > 0 and -running_jobs[0][0] > max_timestamp:
                max_timestamp = -running_jobs[0][0]
                if current_round_end_time is not None:
                    current_round_start_time = current_round_end_time
                current_round_end_time = max_timestamp
            if max_timestamp > 0:
                self._current_timestamp = max_timestamp
            else:
                self._current_timestamp = next_job_arrival_time

            # Update per-instance type prices.
            if self._per_worker_type_prices is not None:
                self._update_per_worker_type_prices()

            # Check if any jobs have completed.
            while len(running_jobs) > 0:
                (
                    finish_time,
                    job_id,
                    worker_ids,
                    all_num_steps,
                ) = running_jobs[0]
                finish_time = -finish_time
                if finish_time <= self._current_timestamp:
                    all_execution_times = []
                    for single_job_id in job_id.singletons():
                        start_time = current_round_start_time
                        execution_time = finish_time - start_time
                        nfs_slowdown_factor = 1

                        if current_round != 1:
                            # inject overhead of preemption/reallocation for simulation fidelity
                            self._logger.debug(
                                f"Jobs run in the previous round include {self._per_round_schedule[current_round - 2].keys()}"
                            )
                            if (
                                single_job_id
                                not in self._per_round_schedule[
                                    current_round - 2
                                ].keys()
                            ):
                                # The job is not scheduled in the previous round.
                                # The actual execution time needs to account for the checkpoint/resume overhead.
                                # Here, to simplify things, we combine the overhead from checkpoint and resume.
                                preemption_overhead = (
                                    20  # use an approach similar to pollux
                                )
                                self._logger.debug(
                                    f"[Job {single_job_id}] is not run in the previous round, manually adding NFS overhead of {round(preemption_overhead, 1)}s"
                                )
                                if (
                                    execution_time != 0
                                    and self._time_per_iteration - 5
                                    < execution_time
                                ):
                                    # if execution time is smaller than round length, it indicates that the
                                    # last round is finishing up, and if we continue to inject the overhead,
                                    # there might be rounding issues that cause jobs to progress for only a
                                    # few steps in a round, creating a long tail
                                    nfs_slowdown_factor = (
                                        execution_time - preemption_overhead
                                    ) / execution_time
                                    execution_time -= preemption_overhead

                        all_execution_times.append(execution_time)
                        self._per_job_latest_timestamps[
                            single_job_id
                        ] = finish_time
                    self._in_progress_updates[job_id] = []
                    scale_factor = self._jobs[
                        job_id.singletons()[0]
                    ].scale_factor
                    total_steps = [0] * len(job_id.singletons())
                    all_num_steps = [
                        int(x * nfs_slowdown_factor) for x in all_num_steps
                    ]
                    for i, worker_id in enumerate(worker_ids):
                        if i == len(worker_ids) - 1:
                            # For the last worker, assign all remaining
                            # steps to account for any rounding error.
                            all_num_steps_ = []
                            for j in range(len(all_num_steps)):
                                remaining_steps = (
                                    all_num_steps[j] - total_steps[j]
                                )
                                all_num_steps_.append(remaining_steps)
                        else:
                            # Each worker gets an equal fraction of the total
                            # number of steps.
                            all_num_steps_ = [
                                x // scale_factor for x in all_num_steps
                            ]

                        for j in range(len(all_num_steps_)):
                            total_steps[j] += all_num_steps_[j]
                        # self._logger.info(f"{job_id}, {worker_id}, {all_num_steps_}, {all_execution_times}")
                        self._done_callback(
                            job_id,
                            worker_id,
                            all_num_steps_,
                            all_execution_times,
                        )
                    for single_job_id in job_id.singletons():
                        if single_job_id not in self._jobs:
                            if from_trace or num_total_jobs is not None:
                                remaining_jobs -= 1
                    heapq.heappop(running_jobs)
                else:
                    break

            #########################################################################
            # Simulation of resource requirements change for Accordion & GNS
            #########################################################################
            for single_job_id in self._jobs.keys():
                if self._jobs[single_job_id].mode == "accordion":
                    self._simulate_accordion(single_job_id)
                elif self._jobs[single_job_id].mode == "gns":
                    self._simulate_gns(single_job_id)

            # shockwave: end of current round, update epoch progress for all jobs,
            # check if any job has completed
            if (
                self._policy.name == "shockwave"
                and self._current_timestamp != 0.0
            ):
                # don't do these calculations if in the first round, before populating self._jobs
                self._update_shockwave_scheduler()

            # Since we're scheduling in rounds, no jobs should be
            # running when scheduling the next round of jobs.
            assert len(running_jobs) == 0

            # Dispatch any newly arrived jobs.
            last_added_job_id = None
            if from_trace:
                while len(queued_jobs) > 0:
                    (arrival_time, job) = queued_jobs[0]
                    if arrival_time <= self._current_timestamp:
                        job_id = self.add_job(job, timestamp=arrival_time)
                        if jobs_to_complete is not None and job_id == min(
                            jobs_to_complete
                        ):
                            window_start_time = self._current_timestamp
                        last_added_job_id = job_id
                        queued_jobs.pop(0)
                    else:
                        break
            else:
                while next_job_arrival_time <= self._current_timestamp:
                    if num_total_jobs is not None:
                        if num_jobs_generated >= num_total_jobs:
                            break
                    job = utils.generate_job(
                        throughputs=self._oracle_throughputs,
                        reference_worker_type="v100",
                        rng=self._job_generator,
                        job_id=None,
                        fixed_job_duration=fixed_job_duration,
                        generate_multi_gpu_jobs=generate_multi_gpu_jobs,
                        generate_multi_priority_jobs=generate_multi_priority_jobs,
                        SLO_rng=SLO_generator,
                    )
                    num_jobs_generated += 1
                    self._all_jobs.append((next_job_arrival_time, job))
                    job_id = self.add_job(job, timestamp=next_job_arrival_time)
                    if jobs_to_complete is not None and job_id == min(
                        jobs_to_complete
                    ):
                        window_start_time = next_job_arrival_time
                        if output_trace_file is not None:
                            self._logger.info(
                                "{0} running jobs at window start".format(
                                    len(self._jobs) - 1
                                )
                            )
                            # Dump already running jobs.
                            for running_job_id in sorted(self._jobs.keys()):
                                remaining_steps = self._get_remaining_steps(
                                    running_job_id
                                )
                                total_steps = self._jobs[
                                    running_job_id
                                ].total_steps
                                self._jobs[
                                    running_job_id
                                ]._total_steps = remaining_steps
                                output_trace_file.write(
                                    "%s\t0\n"
                                    % (str(self._jobs[running_job_id]))
                                )
                                self._jobs[
                                    running_job_id
                                ]._total_steps = total_steps
                    if (
                        jobs_to_complete is None
                        or window_start_time is not None
                    ) and output_trace_file is not None:
                        output_arrival_time = next_job_arrival_time
                        if window_start_time is not None:
                            output_arrival_time -= window_start_time
                        output_trace_file.write(
                            "%s\t%f\n" % (str(job), output_arrival_time)
                        )
                    last_added_job_id = job_id

                    last_job_arrival_time = next_job_arrival_time
                    if lam == 0.0:
                        arrival_time_delta = 0.0
                    else:
                        arrival_time_delta = self._sample_arrival_time_delta(
                            1.0 / lam
                        )
                    next_job_arrival_time = (
                        arrival_time_delta + last_job_arrival_time
                    )

            # Schedule jobs until there are no available workers or no jobs
            # with non-zero allocations on available workers.
            if ideal:
                time_to_next_event = (
                    next_job_arrival_time - self._current_timestamp
                )
                all_num_steps = {}
                self._update_priorities()
                for job_id in self._allocation:
                    for worker_type in self._allocation[job_id]:
                        time_spent_on_worker_type = (
                            self._allocation[job_id][worker_type]
                            * time_to_next_event
                        )
                        if job_id.is_pair():
                            for i, single_job_id in enumerate(
                                job_id.singletons()
                            ):
                                if job_id not in self._throughputs:
                                    continue
                                num_steps = (
                                    time_spent_on_worker_type
                                    * self._throughputs[job_id][worker_type][i]
                                )
                                if single_job_id not in all_num_steps:
                                    all_num_steps[single_job_id] = 0
                                all_num_steps[single_job_id] += int(num_steps)
                        else:
                            if job_id in self._throughputs:
                                num_steps = (
                                    time_spent_on_worker_type
                                    * self._throughputs[job_id][worker_type]
                                )
                                if job_id not in all_num_steps:
                                    all_num_steps[job_id] = 0
                                all_num_steps[job_id] += int(num_steps)
                for job_id in all_num_steps:
                    allocation_str = ""
                    for x in worker_types:
                        allocation_str += " [%4s %f]" % (
                            x,
                            self._allocation[job_id][x],
                        )
                    self._logger.info(
                        "[Micro-task scheduled]\tJob ID: {job_id}\t"
                        "Allocation: {allocation}".format(
                            job_id=job_id, allocation=allocation_str
                        )
                    )
                    heapq.heappush(
                        running_jobs,
                        (
                            -next_job_arrival_time,
                            job_id,
                            (0,),
                            [all_num_steps[job_id]],
                        ),
                    )
                    self._running_jobs.add(job_id)
            else:
                if len(self._jobs) == 0:
                    # completed simulating the trace
                    self._logger.warning(
                        f"Simulation has completed, no jobs left"
                    )
                    break
                with self._scheduler_lock:
                    #######################################################################
                    # _schedule_jobs_on_workers() calls _update_priorities(), which in turn calls _compute_allocation()
                    # shockwave integration is done in _schedule_jobs_on_workers_helper
                    scheduled_jobs = self._schedule_jobs_on_workers()
                    #######################################################################

                    for job_id in self._current_worker_assignments:
                        is_active = any(
                            [x in self._jobs for x in job_id.singletons()]
                        )
                        if is_active:
                            self._num_lease_extension_opportunities += 1
                    for job_id in scheduled_jobs:
                        if job_id in self._current_worker_assignments:
                            current_worker_ids = set(
                                self._current_worker_assignments[job_id]
                            )
                            next_worker_ids = set(scheduled_jobs[job_id])
                            if current_worker_ids == next_worker_ids:
                                self._num_lease_extensions += 1
                    self._current_worker_assignments = scheduled_jobs
                    # self._print_schedule_summary()
                for (job_id, worker_ids) in scheduled_jobs.items():
                    worker_type = self._worker_id_to_worker_type_mapping[
                        worker_ids[0]
                    ]
                    for worker_id in worker_ids:
                        self._remove_available_worker_id(worker_id)
                    (
                        all_num_steps,
                        max_finish_time,
                    ) = self._get_job_steps_and_finish_times(
                        job_id, worker_type
                    )
                    heapq.heappush(
                        running_jobs,
                        (-max_finish_time, job_id, worker_ids, all_num_steps),
                    )

            if (
                checkpoint_threshold is not None
                and last_added_job_id is not None
                and last_added_job_id[0] >= checkpoint_threshold
                and not checkpoint_complete
            ):
                # Create checkpoint.
                assert checkpoint_file is not None
                self._save_checkpoint(
                    checkpoint_file,
                    last_job_arrival_time,
                    next_job_arrival_time,
                    current_round_start_time,
                    current_round_end_time,
                    running_jobs,
                )
                checkpoint_complete = True

            self._logger.info(
                "*** END ROUND {0} ***\n\n\n".format(current_round)
            )
            current_round += 1
            self._num_completed_rounds += 1

        if window_start_time is not None:
            print("Window start time: %f" % (window_start_time))
            window_duration = self._current_timestamp - window_start_time
            print(
                "Window duration: "
                "%.3f seconds (%.2f hours)"
                % (window_duration, window_duration / 3600.0)
            )
        if output_trace_file is not None:
            output_trace_file.close()
        self._logger.info(
            "Total duration/makespan: %.3f seconds "
            "(%.2f hours)"
            % (self._current_timestamp, self._current_timestamp / 3600.0)
        )

        return self._current_timestamp

    def _update_shockwave_scheduler(self, jobs_with_extended_lease=None):
        """TODO: fill in function docstring
        """
        # NOTE: in physical expr, shockwave scheduler is updated after the schedule of the next round is computed
        for job_id in (
            self._scheduled_jobs_in_current_round
            if self._simulate
            else self._scheduled_jobs_in_prev_round
        ):
            # for job_id in self._scheduled_jobs_in_current_round:
            if job_id in self._completed_jobs:
                # job has completely finished
                self._logger.debug(
                    f"Make sure job {job_id} has completed in the previous round!"
                )
                # set the epoch progress of jobs that have already completed
                if job_id in self._shockwave_scheduler.metadata.keys():
                    num_epochs = self._shockwave_scheduler.metadata[
                        job_id
                    ].epochs
                    self._shockwave_scheduler.schedule_progress(
                        job_id, num_epochs
                    )
                continue
            if self._simulate:
                if job_id not in self._steps_run_so_far.keys():
                    # job hasn't been run
                    steps_run_so_far = 0
                else:
                    # TODO: use total_steps_run in the future if we use heterogeneous hardware
                    steps_run_so_far = self._steps_run_so_far[job_id]["v100"]
            else:  # physical experiments
                if job_id not in self._steps_run_so_far.keys():
                    # job hasn't been run
                    steps_run_so_far = 0
                else:
                    self._logger.debug(
                        f"jobs_with_extended_lease is {jobs_with_extended_lease}"
                    )
                    if job_id in jobs_with_extended_lease:
                        """
                        NOTE: In physical experiments, when jobs run for consecutive
                        rounds using an extended lease, steps_run_so_far will not get
                        updated in time as steps_run_so_far is only updated in done_callback, 
                        but those jobs call _done_callback_extended_lease.
                        In this case, job has an extended lease, and steps_run_so_far did not include 
                        the iterations executed in the current lease 
                        """
                        steps_run_in_current_lease = self._steps_run_in_current_lease[
                            job_id
                        ]
                    else:
                        # job does not have an extended lease, no possibilities of shenanigans
                        steps_run_in_current_lease = 0
                    self._logger.debug(
                        f"Setting epoch progress for job {job_id} which do{'' if job_id in jobs_with_extended_lease else ' not'} have an extended lease"
                    )
                    self._logger.debug(
                        f"steps_run_in_current_lease is {steps_run_in_current_lease}, steps_run_so_far is {self._steps_run_so_far[job_id]['v100']}"
                    )
                    steps_run_so_far = self._steps_run_so_far[job_id]["v100"]
                    steps_run_so_far += steps_run_in_current_lease
            current_bs = self._jobs[job_id].batch_size
            len_dataset = dataset_size_dict[self._jobs[job_id].model]
            current_epoch = math.floor(
                steps_run_so_far / math.ceil(len_dataset / current_bs)
            )
            assert job_id in self._shockwave_scheduler.metadata.keys()
            num_epochs = self._shockwave_scheduler.metadata[job_id].epochs
            self._logger.debug(
                f"Setting epoch progress for job {job_id} to {current_epoch}/{num_epochs} (steps_run_so_far={steps_run_so_far})"
            )
            self._shockwave_scheduler.schedule_progress(job_id, current_epoch)

        all_jobs = set(self._jobs.keys())
        scheduled_jobs_in_current_round = set(
            self._scheduled_jobs_in_current_round
            if self._simulate
            else self._scheduled_jobs_in_prev_round
        )
        # for active jobs that are not scheduled in the current round, add waiting delay
        jobs_not_scheduled_in_current_round = (
            all_jobs - scheduled_jobs_in_current_round
        )

        for job_id in jobs_not_scheduled_in_current_round:
            self._logger.debug(f"Increment waiting delay for job {job_id}")
            self._shockwave_scheduler.deschedule_waiting_delay(
                job_id, self._time_per_iteration
            )

        self._shockwave_scheduler.increment_round_ptr()

        # check if any job completed its whole progress
        self._iround_reopt += 1
        if (
            self._shockwave_job_completed_flag
            or self._iround_reopt >= REOPT_ROUNDS
        ):
            self._logger.debug(
                f"_shockwave_job_completed_flag is {self._shockwave_job_completed_flag}, resolving a new schedule for future rounds"
            )
            self._shockwave_job_completed_flag = False
            self._iround_reopt = 0
            self._shockwave_scheduler.set_resolve()

    def _is_final_round(self):
        return (
            self._max_rounds is not None
            and self._num_completed_rounds + 1 == self._max_rounds
        )

    def _begin_round(self, state_snapshot=None):
        """Executes beginning stage of a scheduling round."""

        self._current_round_start_time = self.get_current_timestamp()
        current_round = self._num_completed_rounds

        # Reset lease update requests.
        for job_id in self._current_worker_assignments:
            for single_job_id in job_id.singletons():
                self._lease_update_requests[single_job_id] = []
                self._max_steps[single_job_id] = None

        # Re-dispatch jobs that had extended leases but completed early.
        for job_id in self._redispatched_worker_assignments:
            is_active = any([x in self._jobs for x in job_id.singletons()])
            if is_active:
                if job_id not in self._current_worker_assignments:
                    raise RuntimeError(
                        "Trying to re-dispatch job {0} but it has not "
                        "been scheduled for round {1}!".format(
                            job_id, current_round
                        )
                    )
                worker_ids = self._redispatched_worker_assignments[job_id]
                self._logger.info(
                    "Re-dispatching job {0} as it completed "
                    "early but had an extended lease".format(job_id)
                )
                self._try_dispatch_job(job_id, worker_ids)
                self._logger.debug("Re-dispatched job {0}".format(job_id))
        self._redispatched_worker_assignments = collections.OrderedDict()

        self._logger.debug("Finished re-dispatching jobs")

        self._logger.info("*** START ROUND {0} ***".format(current_round))
        self._print_schedule_summary(state_snapshot)

    def _mid_round(self, pool):
        """Executes intermediate stage of a scheduling round.

        Computes the schedule for the upcoming round partway through the
        current round and extends leases if necessary. Then dispatches jobs
        for the upcoming round and schedules callbacks for jobs with extended
        leases.

        Note that this updates self._next_worker_assignments. We update
        self._current_worker_assignments when we end the round.
        """

        if self._is_final_round():
            self._logger.debug("In final round, not dispatching any more jobs")
            self._jobs_with_extended_leases = set()
            return

        round_end_time = (
            self._current_round_start_time + self._time_per_iteration
        )

        # Recompute the schedule for the upcoming round.
        # NOTE: if jobs complete after mid_round is called, they will still
        # be scheduled in the next round
        self._next_worker_assignments = self._schedule_jobs_on_workers()

        # Count how many jobs could be eligible for lease extensions.
        for job_id in self._current_worker_assignments:
            is_active = any([x in self._jobs for x in job_id.singletons()])
            if is_active:
                self._num_lease_extension_opportunities += 1

        # Check whether we should update the lease for any jobs.
        for job_id in self._current_worker_assignments:
            current_worker_ids = set(self._current_worker_assignments[job_id])
            if (
                job_id in self._next_worker_assignments
                and job_id not in self._completed_jobs_in_current_round
            ):
                next_worker_ids = set(self._next_worker_assignments[job_id])
                if current_worker_ids == next_worker_ids:
                    # Job will be scheduled on the same workers in
                    # upcoming round; extend its lease.
                    self._jobs_with_extended_lease.add(job_id)
                    self._logger.info(
                        "Extending lease for job {0}".format(job_id)
                    )
                    self._num_lease_extensions += 1
                elif job_id in self._jobs_with_extended_lease:
                    # Job will not be scheduled on the same workers
                    # in upcoming round; remove it from the
                    # extended lease set if it had previously
                    # received an extended lease.
                    self._jobs_with_extended_lease.remove(job_id)
            elif job_id in self._jobs_with_extended_lease:
                # Job will not be scheduled in upcoming round;
                # remove it from the extended lease set if it
                # had previously received an extended lease.
                self._jobs_with_extended_lease.remove(job_id)

        # Dispatch jobs for upcoming round.
        for (job_id, worker_ids) in self._next_worker_assignments.items():
            is_active = any([x in self._jobs for x in job_id.singletons()])
            if not is_active:
                continue
            elif job_id not in self._jobs_with_extended_lease or (
                job_id in self._jobs_with_extended_lease
                and job_id in self._completed_jobs_in_current_round
            ):
                self._logger.info("Dispatching job {0}".format(job_id))
                self._try_dispatch_job(job_id, worker_ids, next_round=True)

        # Schedule completion events.
        self._schedule_completion_events(round_end_time, pool)

    def _try_dispatch_job(self, job_id, worker_ids, next_round=False):
        """Attempts to dispatch the specified job combination.

           Updates relevant metadata and returns if job has already been
           dispatched.
        """
        # self._logger.debug(f"*******_try_dispatch_job is invoked")

        # Initialize metadata.
        if not next_round or job_id not in self._current_worker_assignments:
            self._in_progress_updates[job_id] = []
            for single_job_id in job_id.singletons():
                self._lease_update_requests[single_job_id] = []
                self._max_steps[single_job_id] = None

        scale_factor = len(worker_ids)
        worker_type = self._worker_id_to_worker_type_mapping[worker_ids[0]]
        if scale_factor > 1:
            master_addr = self._worker_connections[worker_ids[0]].addr
            master_server_port = self._worker_connections[worker_ids[0]].port
            master_job_ports = []
            for i in range(len(job_id.singletons())):
                master_job_ports.append(BASE_JOB_PORT + self._port_offset)
                self._logger.info(
                    f"Job {job_id} port is {BASE_JOB_PORT + self._port_offset}"
                )
                self._port_offset += 1
                self._port_offset %= MAX_PORT - BASE_JOB_PORT

        # Dispatch the job.
        current_round = self._num_completed_rounds
        if next_round:
            current_round += 1
        for i, worker_id in enumerate(worker_ids):
            job_descriptions = []
            for j, single_job_id in enumerate(job_id.singletons()):
                self._logger.info(
                    f"Job {single_job_id}: {self._jobs[single_job_id]}"
                )
                num_steps = self._jobs[single_job_id].total_steps
                command = self._jobs[single_job_id].command
                mps_thread_percentage = self._jobs[
                    single_job_id
                ].mps_thread_percentage
                # Add distributed args if necessary.
                if scale_factor > 1:
                    command = (
                        "%s --master_addr %s "
                        "--master_port %d "
                        "--world_size %d "
                        "--rank %d"
                        % (
                            command,
                            master_addr,
                            master_job_ports[j],
                            scale_factor,
                            i,
                        )
                    )
                # self._logger.info(f"Dispatching job {job_id} with mps percentage {mps_thread_percentage}%")
                job_descriptions.append(
                    (
                        single_job_id,
                        command,
                        self._jobs[single_job_id].working_directory,
                        self._jobs[single_job_id].needs_data_dir,
                        self._jobs[single_job_id].num_steps_arg,
                        num_steps,
                        self._jobs[single_job_id].mode,
                        mps_thread_percentage,
                    )
                )
            self._logger.info(
                f"Round {current_round}, running job {job_id[0]} on worker {worker_id}, job_descriptions: {job_descriptions}"
            )
            self._worker_connections[worker_id].run_job(
                job_descriptions, worker_id, current_round
            )
            if not next_round:
                self._remove_available_worker_id(worker_id)

    def _schedule_completion_events(self, round_end_time, pool):
        """Schedules completion events for every dispatched job.

        A completion event in this setting is a callback that will be
        triggered at the conclusion of the current round to indicate that the
        specified job has completed the round. This is necessary for two
        reasons: 1) jobs with extended leases will not trigger the standard
        done_callback at the end of the round, and 2) jobs might freeze.
        """
        current_time = self.get_current_timestamp()
        for job_id in self._current_worker_assignments:
            is_active = any([x in self._jobs for x in job_id.singletons()])
            if (
                not is_active
                or job_id in self._completed_jobs_in_current_round
            ):
                continue
            delay = round_end_time - current_time
            if job_id not in self._jobs_with_extended_lease:
                delay += JOB_COMPLETION_BUFFER_TIME
                action = self._kill_job
            else:
                action = self._done_callback_extended_lease
            event = self._completion_event_scheduler.enter(
                delay=delay, priority=1, action=action, argument=(job_id,)
            )
            self._completion_events[job_id] = event
        self._logger.debug(f"Completion events: {self._completion_events}")
        self._logger.debug(
            f"_completion_event_scheduler.queue: {self._completion_event_scheduler.queue}"
        )
        pool.submit(self._completion_event_scheduler.run)

    def _end_round(self):
        """Executes final stage of a scheduling round.

        Waits for all currently dispatched jobs to complete, then resets
        the set of jobs with extended leases as well as relevant metadata.
        """

        current_round = self._num_completed_rounds

        # Wait for jobs in current round to complete.
        jobs_to_complete = set()
        for job_id in self._current_worker_assignments:
            is_active = any([x in self._jobs for x in job_id.singletons()])
            if is_active:
                jobs_to_complete.add(job_id)
        self._logger.debug(
            "Waiting for following jobs "
            "to complete: {0}".format(sorted(jobs_to_complete))
        )

        while not jobs_to_complete.issubset(
            self._completed_jobs_in_current_round
        ):
            self._scheduler_cv.wait()
            remaining_jobs = jobs_to_complete.difference(
                self._completed_jobs_in_current_round
            )
            self._logger.debug(
                "Remaining jobs in round: {0}".format(sorted(remaining_jobs))
            )

        self._logger.debug(
            "All jobs in round {0} have completed!".format(current_round)
        )

        if len(self._completion_events) > 0:
            raise RuntimeError(
                "Remaining completion events: {0}".format(
                    self._completion_events.keys()
                )
            )

        # self._jobs_with_extended_lease_and_bs_request = []

        # Reset extended leases.
        jobs_with_extended_lease = list(self._jobs_with_extended_lease)
        for job_id in jobs_with_extended_lease:
            if job_id in self._jobs:
                current_worker_ids = self._current_worker_assignments[job_id]
                for worker_id in current_worker_ids:
                    self._add_available_worker_id(worker_id)
            self._jobs_with_extended_lease.remove(job_id)
        self._logger.debug("Reset extended leases")

        if not self._is_final_round():
            # The next worker assignments must have been computed here as
            # _end_round is called sequentially after _mid_round.
            if self._next_worker_assignments is None:
                raise RuntimeError(
                    "Next worker assignments have not been computed!"
                )

            for (job_id, worker_ids) in self._next_worker_assignments.items():
                is_active = any([x in self._jobs for x in job_id.singletons()])
                if is_active:
                    # If the job needs to be dispatched again, defer removing
                    # its worker ID.
                    if job_id in self._redispatched_worker_assignments:
                        continue
                    self._logger.info(
                        f"Trying to remove worker_ids {worker_ids}"
                    )
                    for worker_id in worker_ids:
                        self._remove_available_worker_id(worker_id)

            # Ensure that rounds do not finish earlier than the specified
            # round duration.
            current_time = self.get_current_timestamp()
            round_end_time = (
                self._current_round_start_time + self._time_per_iteration
            )
            remaining_time_in_round = round_end_time - current_time
            if remaining_time_in_round > 0:
                self._logger.debug(
                    "Waiting {0:.2f} seconds before starting "
                    "round {1}...".format(
                        remaining_time_in_round, current_round + 1
                    )
                )
                time.sleep(remaining_time_in_round)

        self._num_completed_rounds += 1

        # Reset metadata.
        self._completed_jobs_in_current_round = set()
        self._current_worker_assignments = self._next_worker_assignments
        self._next_worker_assignments = None

        self._scheduler_cv.notifyAll()

        self._logger.info("*** END ROUND {0} ***".format(current_round))

    def _schedule_with_rounds(self):
        """Schedules jobs on workers using rounds.

        In a loop, schedules in rounds the applications most in need of
        being run (that is, the applications with the highest
        fraction_allocated/fraction_run ratio) using a DP algorithm.
        """

        self._logger.info("_schedule_with_rounds is invoked")

        self._scheduler_cv.acquire()
        # Wait for jobs to arrive and all workers to register with scheduler.
        while len(self._jobs) == 0 or (
            self._expected_num_workers is not None
            and len(self._worker_ids) < self._expected_num_workers
        ):
            self._scheduler_cv.wait()

        # Add all workers to the queue.
        for worker_id in self._worker_ids:
            self._available_worker_ids.put(worker_id)

        # Wait for initial allocation to be computed.
        if self._policy.name != "shockwave":
            while self._need_to_update_allocation:
                self._scheduler_cv.wait()

        # Compute initial schedule and dispatch initial set of jobs.
        self._current_worker_assignments = self._schedule_jobs_on_workers()
        if self._policy.name == "shockwave":
            # fetch a new round of schedule in the next mid_round
            self._shockwave_scheduler.increment_round_ptr()
        state_snapshot = self._get_state_snapshot()
        for (job_id, worker_ids) in self._current_worker_assignments.items():
            self._try_dispatch_job(job_id, worker_ids)
        self._scheduler_cv.release()

        with ThreadPoolExecutor(max_workers=1) as pool:
            while True:
                is_final_round = self._is_final_round()

                round_start_time = self.get_current_timestamp()

                with self._scheduler_cv:
                    self._begin_round(state_snapshot)

                # Wait for partway through round to recompute schedule.
                delay = self._time_per_iteration * SCHEDULE_RECOMPUTE_FRACTION
                time.sleep(delay)

                with self._scheduler_cv:
                    self._logger.info("Starting _mid_round")
                    self._mid_round(pool)
                    state_snapshot = self._get_state_snapshot(deepcopy=True)
                    if self._policy.name == "shockwave":
                        jobs_with_extended_lease = copy.deepcopy(
                            self._jobs_with_extended_lease
                        )
                    self._logger.info("Starting _end_round")
                    self._end_round()
                    if self._policy.name == "shockwave":
                        # TODO: should this happen before end_round?
                        # probably not if we are only moving mid_round and lease update backwards in time
                        self._update_shockwave_scheduler(
                            jobs_with_extended_lease=jobs_with_extended_lease
                        )
                if is_final_round:
                    break

    def get_average_jct(self, job_ids=None, verbose=True):
        """Computes the average job completion time.

           Args:
               job_ids: A list of `JobIdPair` objects. If specified, computes
                        the average JCT using only these jobs.

           Returns: The average JCT.
        """
        with self._scheduler_lock:
            if len(self._job_completion_times) == 0:
                return
            if job_ids is None:
                job_ids = sorted(list(self._job_completion_times.keys()))
            else:
                job_ids = sorted(job_ids)
            self._logger.info("-" * 50)
            self._logger.info("Job completion times:")
            all_job_completion_times = []
            low_priority_job_completion_times = []
            high_priority_job_completion_times = []
            for job_id in job_ids:
                completion_time = self._job_completion_times[job_id]
                if completion_time is None:
                    continue
                else:
                    all_job_completion_times.append(completion_time)
                if self._job_priority_weights[job_id] == 1.0:
                    self._logger.info(
                        "Job %s: %.3f" % (job_id, completion_time)
                    )
                    low_priority_job_completion_times.append(completion_time)
                else:
                    self._logger.info(
                        "Job %s (high priority): %.3f"
                        % (job_id, completion_time)
                    )
                    high_priority_job_completion_times.append(completion_time)
                # self._logger.info(f"Job {job_id}, start {self._per_job_start_timestamps[job_id]}, end {self._per_job_latest_timestamps[job_id]}")
            avg_jct = np.mean(all_job_completion_times)
            geometric_mean_jct = scipy.stats.mstats.gmean(
                all_job_completion_times
            )
            harmonic_mean_jct = scipy.stats.hmean(all_job_completion_times)
            if verbose:
                self._logger.info(
                    "Average job completion time: %.3f seconds "
                    "(%.2f hours)" % (avg_jct, avg_jct / 3600.0)
                )
                self._logger.info(
                    "Geometric avg jct: %.3f seconds "
                    "(%.2f hours)"
                    % (geometric_mean_jct, geometric_mean_jct / 3600.0)
                )
                self._logger.info(
                    "Harmonic avg jct: %.3f seconds "
                    "(%.2f hours)"
                    % (harmonic_mean_jct, harmonic_mean_jct / 3600.0)
                )
                if len(low_priority_job_completion_times) > 0:
                    average_low_pri_jct = np.mean(
                        low_priority_job_completion_times
                    )
                    self._logger.info(
                        "Average job completion time (low priority): "
                        "%.3f seconds "
                        "(%.2f hours)"
                        % (average_low_pri_jct, average_low_pri_jct / 3600.0)
                    )
                if len(high_priority_job_completion_times) > 0:
                    average_high_pri_jct = np.mean(
                        high_priority_job_completion_times
                    )
                    self._logger.info(
                        "Average job completion time (high priority): "
                        "%.3f seconds "
                        "(%.2f hours)"
                        % (average_high_pri_jct, average_high_pri_jct / 3600.0)
                    )
            return (
                avg_jct,
                geometric_mean_jct,
                harmonic_mean_jct,
                all_job_completion_times,
            )

    def get_finish_time_fairness(self, job_ids=None, pickle_file_name=None):
        """Computes the finish time fairness for each job.
           Args:
               job_ids: A list of `JobIdPair` objects. If specified, computes
                        the normalized utilities using only these jobs.
        
           Returns;
                utilities: A list of finish time fairness for all jobs in job_ids
        """
        self._logger.info("-" * 50)
        self._logger.info("Finish time fairness:")
        num_gpus = len(self._worker_ids)
        with self._scheduler_lock:
            if len(self._job_completion_times) == 0:
                return
            if job_ids is None:
                job_ids = sorted(list(self._job_completion_times.keys()))
            else:
                job_ids = sorted(job_ids)

            finish_time_fairness_list = []
            finish_time_fairness_themis_list = []
            for job_id in job_ids:
                completion_time = self._job_completion_times[job_id]
                if completion_time is None:
                    continue
                # get the original job type
                job_type = self._job_types[job_id]
                # get the original num iterations
                num_steps = self._original_num_steps[job_id]
                # get the original batch size
                original_batch_size = self._original_bs[job_id]
                # get the original number of epochs
                original_num_epochs = self._get_num_epochs(
                    job_type, original_batch_size, num_steps
                )
                # calculate the total time of a job running individually
                total_time_running_exclusively = sum(
                    self._profiles[job_id.integer_job_id()][
                        "duration_every_epoch"
                    ]
                )

                avg_contention_factor = (
                    self._num_jobs_in_trace / num_gpus
                )  # N jobs in trace, each job gets 1/N of the cluster
                avg_contention_factor = max(
                    1.0, avg_contention_factor
                )  # otherwise, for job 0 in a continuous trace, the contention factor will < 0 and the fairness will > 1 even if it doesn't queue/hang
                finish_time_fairness = round(
                    completion_time
                    / (total_time_running_exclusively * avg_contention_factor),
                    5,
                )
                finish_time_fairness_list.append(finish_time_fairness)
                self._logger.info(
                    f"Job {job_id}: {finish_time_fairness} = {round(completion_time, 2)} / {round(total_time_running_exclusively * avg_contention_factor, 2)}"
                )

                start_round = self._job_start_round[job_id.integer_job_id()]
                end_round = self._job_end_round[job_id.integer_job_id()]
                avg_contention_factor = (
                    np.mean(
                        self._num_jobs_in_curr_round[start_round:end_round]
                    )
                    / num_gpus
                )  # TODO: fix this for job that start & finish in the same round
                avg_contention_factor = max(
                    1.0, avg_contention_factor
                )  # otherwise, for job 0 in a continuous trace, the contention factor will < 0 and the fairness will > 1 even if it doesn't queue/hang
                finish_time_fairness = round(
                    completion_time
                    / (total_time_running_exclusively * avg_contention_factor),
                    5,
                )
                finish_time_fairness_themis_list.append(finish_time_fairness)

            self._logger.info(f"Finish time fairness sorted from high to low:")
            jobs_rhos_static = list(zip(job_ids, finish_time_fairness_list))
            jobs_rhos_static = sorted(
                jobs_rhos_static, key=lambda kv: kv[1], reverse=True
            )
            for job, ftfrho in jobs_rhos_static:
                self._logger.info(f"Job_ID: {job}, Rho(Static): {ftfrho}")
            self._logger.info(
                f"Finish time fairness (static contention factor) rho percentiles: {np.percentile(np.array(finish_time_fairness_list), [0, 25, 50, 75, 100])}"
            )

            jobs_rhos_themis = list(
                zip(job_ids, finish_time_fairness_themis_list)
            )
            jobs_rhos_themis = sorted(
                jobs_rhos_themis, key=lambda kv: kv[1], reverse=True
            )
            for job, ftfrho in jobs_rhos_themis:
                self._logger.debug(f"Job_ID: {job}, Rho(Themis): {ftfrho}")
            self._logger.debug(
                f"Finish time fairness (Themis) rho percentiles: {np.percentile(np.array(finish_time_fairness_themis_list), [0, 25, 50, 75, 100])}"
            )
            return finish_time_fairness_list, finish_time_fairness_themis_list

    def get_envy_list(self):
        """Computes the envy ratio & absolute envy list.
        TODO: This can be worded better...

        Returns;
            envy_ratios: A list of envy-ness for all jobs
            vals_absdiff: A list of absolute difference for all envy-ness of jobs
        """
        self._logger.info("-" * 50)
        self._logger.info("Envy ratio sorted from low to high:")
        envy_ratios = OrderedDict()

        for job_id in range(self._job_id_counter):
            num_scheduled_rounds = self._num_scheduled_rounds[job_id]
            num_queued_rounds = self._num_queued_rounds[job_id]
            """
            num_scheduled_rounds is the exclusive running time,
            num_scheduled_rounds + num_queued_rounds is the shared running time.
            This ratio is the reciprocal of the slowdown due to sharing, 
            and a job will envy another job if this other job has a smaller slowdown.
            """
            ratio = num_scheduled_rounds / (
                num_scheduled_rounds + num_queued_rounds
            )
            envy_ratios[job_id] = ratio

        sorted_envy_ratios = sorted(envy_ratios, key=envy_ratios.get)
        for job_id in sorted_envy_ratios:
            self._logger.info(
                f"JobID: {job_id}, envy ratio: {round(envy_ratios[job_id], 2)}"
            )

        ratio_values = envy_ratios.values()
        vals_absdiff = [
            np.abs(vali - valj)
            for j, valj in enumerate(ratio_values)
            for i, vali in enumerate(ratio_values)
            if i > j
        ]

        if len(envy_ratios) >= 2:
            self._logger.info(
                "Pairwise envy freeness ratios at percentiles {0, 25, 50, 75, 90, 100}%:"
            )
            self._logger.info(
                f"{[round(x, 2) for x in np.percentile(np.array(vals_absdiff), [0, 25, 50, 75, 90, 100])]}"
            )

        return envy_ratios, vals_absdiff

    def get_throughput_timeline(self):
        return self._throughput_timeline

    def get_job_run_time(self):
        """Return the per-job training time on each worker
        """
        self._logger.info("Cumulative run time for each job:")
        self._logger.info(self._cumulative_run_time)
        return self._cumulative_run_time

    def get_completed_steps(self, job_ids=None):
        print("Completed steps:")
        if job_ids is None:
            job_ids = sorted(list(self._total_steps_run.keys()))
        else:
            job_ids = sorted(job_ids)
        for job_id in job_ids:
            if job_id in self._total_steps_run:
                completed_steps = self._total_steps_run[job_id]
                print("Job %s: %d steps" % (job_id, completed_steps))

    def get_cluster_utilization(self):
        """Computes the utilization of the cluster."""
        with self._scheduler_lock:
            utilizations = []
            current_timestamp = self.get_current_timestamp()
            for worker_id in self._cumulative_worker_time_so_far:
                total_runtime = (
                    current_timestamp - self._worker_start_times[worker_id]
                )
                worker_time = self._cumulative_worker_time_so_far[worker_id]
                utilization = worker_time / total_runtime
                if utilization > 1.0 and not self._job_packing:
                    print("Error: invalid utilization %.3f" % (utilization))
                    print("Worker ID: %d" % (worker_id))
                    print("Worker time: %.3f" % (worker_time))
                    print("Total time: %.3f." % (total_runtime))
                    return None
                utilizations.append(round(utilization, 5))
            print("utilization is", utilizations)
            cluster_utilization = np.mean(utilizations)
            print("Cluster utilization: %.3f" % (cluster_utilization))
            return cluster_utilization, utilizations

    def get_total_cost(self, verbose=True):
        total_cost = 0.0
        for job_id in self._job_cost_so_far:
            total_cost += self._job_cost_so_far[job_id]
        if verbose:
            print("Total cost: $%.2f" % (total_cost))
        return total_cost

    def get_num_SLO_violations(self, verbose=True):
        num_SLO_violations = 0
        if self._SLOs is not None:
            for job_id in self._SLOs:
                SLO = self._SLOs[job_id]
                completion_time = self._job_completion_times[job_id]
                if verbose:
                    print(
                        "%s: completion_time=%f, SLO=%f, "
                        "completion_time / SLO = %f"
                        % (job_id, completion_time, SLO, completion_time / SLO)
                    )
                if completion_time > SLO:
                    num_SLO_violations += 1
        if verbose:
            print("Number of SLO violations: %d" % (num_SLO_violations))
        return num_SLO_violations

    def get_num_lease_extensions(self, verbose=True):
        if self._num_lease_extension_opportunities > 0:
            percentage = (
                100.0 * self._num_lease_extensions
            ) / self._num_lease_extension_opportunities
            if verbose:
                print(
                    "Extended leases {0:.2f}% of the time ({1}/{2})".format(
                        percentage,
                        self._num_lease_extensions,
                        self._num_lease_extension_opportunities,
                    )
                )
        elif verbose:
            percentage = 0
            print("No lease extension opportunities")

        return (
            percentage,
            self._num_lease_extensions,
            self._num_lease_extension_opportunities,
        )

    def save_job_timelines(self, timeline_dir):
        if not os.path.isdir(timeline_dir):
            try:
                os.mkdir(timeline_dir)
            except Exception as e:
                self._logger.error("Could not save timelines!")
                traceback.print_exc()
                return

        for job_id in sorted(self._job_timelines.keys()):
            job_dir = os.path.join(timeline_dir, "job_id={0}".format(job_id))
            if not os.path.isdir(job_dir):
                os.mkdir(job_dir)
            for i in range(len(self._job_timelines[job_id])):
                timeline_file = os.path.join(
                    job_dir, "worker={0}.log".format(i)
                )
                with open(timeline_file, "w") as f:
                    for event in self._job_timelines[job_id][i]:
                        f.write("{0}\n".format(event))

    def get_micro_tasks(self):
        """Prints all micro-tasks run for each job.

           Debug function used print all micro-tasks run for each job.
        """
        job_ids = sorted(self._micro_tasks_per_job.keys())
        for job_id in job_ids:
            print(
                "Job %s: %d" % (job_id, len(self._micro_tasks_per_job[job_id]))
            )
            for i, (start, end) in enumerate(
                self._micro_tasks_per_job[job_id]
            ):
                print("\t%d%f - %f" % (i, start, end))
            print("")

    def get_job_start_and_end_times(self):
        """Returns the start and end times of each job.

           Debug function for returning the start and end times of each job.
        """
        with self._scheduler_lock:
            job_ids = sorted(
                [job_id for job_id in self._per_job_latest_timestamps]
            )
            start_times = [
                self._per_job_start_timestamps[job_id] for job_id in job_ids
            ]
            end_times = [
                self._per_job_latest_timestamps[job_id] for job_id in job_ids
            ]
        return start_times, end_times

    def get_all_simulated_jobs(self, job_range):
        """Returns all the jobs run during simulation.

           Debug function used to print all jobs generated during
           simulation within a specified range.

           Args:
               job_range: A tuple specifying which jobs to be printed.
        """
        print("All simulated jobs:")
        for arrival_time, job in self._all_jobs[job_range[0] : job_range[1]]:
            print(
                "%s\t%s\t%d\t%f"
                % (job.job_id, job.job_type, job.total_steps, arrival_time)
            )

    """
    ======================================================================
       Helper methods to get and mutate state needed for scheduling.
    ======================================================================
    """

    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
    def _print_deficits(self):
        """Prints the deficit.

           Debug method used for printing the deficit of each job on each
           worker type.
        """
        print("")
        print("=" * 80)
        print("Deficits\t(Current_time: %f)" % (self.get_current_timestamp()))
        print("-" * 80)
        for job_id in sorted(list(self._jobs.keys())):
            deficit_str = "Job ID %s:" % (job_id)
            for worker_type in sorted(self._worker_types):
                deficit = self._deficits[worker_type][job_id]
                deficit_str += " [%s: %f]" % (worker_type, deficit)
            print(deficit_str)
        print("=" * 80)
        print("")

    def _get_allocation_state(self):
        """Prepare all relevant scheduler state for computing the allocation."""
        state = {}
        state["scale_factors"] = {
            job_id: self._jobs[job_id].scale_factor for job_id in self._jobs
        }
        state["priority_weights"] = {
            job_id: self._jobs[job_id].priority_weight for job_id in self._jobs
        }

        # state['num_steps_remaining'] = {
        #     job_id: self._get_remaining_steps(job_id)
        #     for job_id in self._jobs
        # }

        state["num_steps_remaining"] = {}
        for job_id in self._jobs:
            remaining_steps = self._get_remaining_steps(job_id)
            # account for in-lease job progress
            # _steps_run_in_current_lease is aggregated over all workers for multi-GPU jobs
            steps_run_in_current_lease = self._steps_run_in_current_lease[
                job_id
            ]
            remaining_steps -= steps_run_in_current_lease
            state["num_steps_remaining"][job_id] = remaining_steps

        state["times_since_start"] = {
            job_id: self.get_current_timestamp()
            - self._per_job_start_timestamps[job_id]
            for job_id in self._jobs
        }
        state["throughputs"] = copy.deepcopy(self._throughputs)
        state["per_round_schedule"] = copy.deepcopy(self._per_round_schedule)

        # # inject different levels of throughput fluctuation during allocation computation
        # # TODO: remove me after the robustness test
        # for job_id in state['throughputs'].keys():
        # state['throughputs'][job_id]['v100'] *= np.random.normal(loc=1.0, scale=0.1)
        # state['throughputs'][job_id]['v100'] *= np.random.normal(loc=1.0, scale=0.25)
        # state['throughputs'][job_id]['v100'] /= abs(np.random.normal(loc=1.0, scale=1))

        state["cluster_spec"] = copy.deepcopy(self._cluster_spec)

        if self._policy.name.startswith("ThroughputNormalizedByCostSum"):
            state["instance_costs"] = copy.deepcopy(
                self._per_worker_type_prices
            )
            if "SLO" in self._policy.name:
                SLOs = {}
                if self._SLOs is not None:
                    for job_id in self._jobs:
                        SLOs[job_id] = self._SLOs[
                            job_id
                        ] - self.get_current_timestamp(in_seconds=True)
                    state["SLOs"] = SLOs
                else:
                    state["num_steps_remaining"] = {}
        return state

    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
    def _compute_allocation(self, state=None):
        """Computes the allocation.

        Uses the specified policy to compute an allocation of jobs to
        compute resources. Requires self._scheduler_lock to be held
        when calling this function.

        Returns:
            A 2-level dict indexed by job_id and then worker_type (the
            unflattened allocation). For example,

            {0: {"v100": 0.25, "p100": 0.95}, 1: {"v100": 0.75, "p100": 0.05}}

            indicates that for 25% of the time, worker type 'v100' should run,
            job 0 and for 95% of the time, worker type 'p100' should run job 0.
        """
        if state is None:
            state = self._get_allocation_state()
        throughputs = state["throughputs"]
        scale_factors = state["scale_factors"]
        times_since_start = state["times_since_start"]
        num_steps_remaining = state["num_steps_remaining"]
        priority_weights = state["priority_weights"]
        cluster_spec = state["cluster_spec"]
        per_round_schedule = state["per_round_schedule"]

        # Compute the allocation.
        if self._policy.name == "AlloX_Perf":
            allocation = self._policy.get_allocation(
                throughputs,
                scale_factors,
                times_since_start,
                num_steps_remaining,
                per_round_schedule,
                cluster_spec,
            )
        elif self._policy.name.startswith("FinishTimeFairness"):
            allocation = self._policy.get_allocation(
                throughputs,
                scale_factors,
                priority_weights,
                times_since_start,
                num_steps_remaining,
                cluster_spec,
            )
        elif self._policy.name.startswith(
            "Isolated"
        ):  # Isolated or Isolated_plus
            allocation = self._policy.get_allocation(
                throughputs, scale_factors, cluster_spec
            )
        elif self._policy.name.startswith("MaxMinFairness"):
            allocation = self._policy.get_allocation(
                throughputs, scale_factors, priority_weights, cluster_spec
            )
        elif self._policy.name.startswith("MinTotalDuration"):
            allocation = self._policy.get_allocation(
                throughputs, scale_factors, num_steps_remaining, cluster_spec
            )
        elif self._policy.name.startswith("ThroughputNormalizedByCostSum"):
            instance_costs = state["instance_costs"]
            if "SLO" in self._policy.name:
                SLOs = state["SLOs"]
                allocation = self._policy.get_allocation(
                    throughputs,
                    scale_factors,
                    cluster_spec,
                    instance_costs=instance_costs,
                    SLOs=SLOs,
                    num_steps_remaining=num_steps_remaining,
                )
            else:
                allocation = self._policy.get_allocation(
                    throughputs,
                    scale_factors,
                    self._cluster_spec,
                    instance_costs=instance_costs,
                )
        elif self._policy.name == "shockwave":
            """
            Before May 25, the Gavel time fraction matrix was used as the solver output,
            and Gavel's internal mechanism was used for turning (fractional matrix + priorities) 
            into a scheduling decision.
            Now, our solver directly outputs the jobs to be scheduled in each round, 
            and we modify _schedule_jobs_on_workers_helper to achieve that.
            """
            allocation = None
        else:
            allocation = self._policy.get_allocation(
                throughputs, scale_factors, self._cluster_spec
            )
        if allocation is None:
            allocation = {}

        self._logger.info(f"Allocation: {allocation}")

        return allocation

    def _allocation_thread(self):
        """Computes the allocation asynchronously."""
        while True:
            # Check whether allocation needs to be re-computed.
            self._scheduler_cv.acquire()
            while not self._need_to_update_allocation:
                self._scheduler_cv.wait()
            state = self._get_allocation_state()
            self._scheduler_cv.release()
            allocation = self._compute_allocation(state)

            # Update allocation and clean up.
            self._scheduler_cv.acquire()
            for job_id in allocation:
                still_active = []
                for single_job_id in job_id.singletons():
                    if single_job_id in self._jobs:
                        still_active.append(True)
                    else:
                        still_active.append(False)
                if not all(still_active):
                    worker_types = allocation[job_id].keys()
                    for i, single_job_id in enumerate(job_id.singletons()):
                        if still_active[i]:
                            # If only one job in a job combination is still
                            # active, re-distribute the job combination's
                            # allocation to the still-active job's isolated
                            # allocation.
                            for worker_type in worker_types:
                                allocation[single_job_id][
                                    worker_type
                                ] += allocation[job_id][worker_type]
                                del allocation[job_id][worker_type]
                            del allocation[job_id]
            self._allocation = allocation
            self._need_to_update_allocation = False
            self._allocation_changed_since_last_time_reset = True
            self._scheduler_cv.notifyAll()
            self._scheduler_cv.release()

    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
    def _populate_job_combination_metadata(self, job_id, worker_type):
        """Populate metadata for job combinations involving passed-in job_id."""
        job = self._jobs[job_id]
        job_type_key = (job.job_type, job.scale_factor)
        if self._estimate_throughputs:
            assert job.scale_factor == 1
            reference_throughputs = self._reference_throughputs[worker_type]
        for other_job_id in self._jobs:
            if other_job_id != job_id:
                other_job = self._jobs[other_job_id]
                if job.scale_factor != other_job.scale_factor:
                    continue
                other_job_type_key = (
                    other_job.job_type,
                    other_job.scale_factor,
                )
                job_type_keys = [job_type_key, other_job_type_key]
                merged_job_id = job_id_pair.JobIdPair(
                    job_id[0], other_job_id[0]
                )
                if merged_job_id not in self._throughputs:
                    self._throughputs[merged_job_id] = {}
                    self._job_time_so_far[merged_job_id] = {}
                    self._priorities[worker_type][job_id] = 0.0
                    self._deficits[worker_type][job_id] = 0.0
                self._job_time_so_far[merged_job_id][worker_type] = 0.0
                if self._estimate_throughputs:
                    reference_job_types = [
                        self._reference_job_map[job_id],
                        self._reference_job_map[other_job_id],
                    ]
                    isolated_throughputs = [
                        self._oracle_throughputs[worker_type][job_type_key][
                            "null"
                        ],
                        self._oracle_throughputs[worker_type][
                            other_job_type_key
                        ]["null"],
                    ]
                    if job_id < other_job_id:
                        self._throughputs[merged_job_id][
                            worker_type
                        ] = np.multiply(
                            reference_throughputs[reference_job_types[0]][
                                reference_job_types[1]
                            ],
                            isolated_throughputs,
                        )
                    else:
                        self._throughputs[merged_job_id][
                            worker_type
                        ] = np.multiply(
                            reference_throughputs[reference_job_types[1]][
                                reference_job_types[0]
                            ],
                            isolated_throughputs[::-1],
                        )
                elif (
                    self._oracle_throughputs is None
                    or job.scale_factor != other_job.scale_factor
                ):
                    self._throughputs[merged_job_id][worker_type] = [0.0, 0.0]
                else:
                    oracle_throughputs = self._oracle_throughputs[worker_type]
                    # The single-job IDs for job pairs are stored in sorted
                    # order so make sure the co-located throughputs match this
                    # order.
                    scale_factor = job.scale_factor
                    if job_id < other_job_id:
                        self._throughputs[merged_job_id][
                            worker_type
                        ] = oracle_throughputs[job_type_keys[0]][
                            job_type_keys[1]
                        ]
                    else:
                        self._throughputs[merged_job_id][
                            worker_type
                        ] = oracle_throughputs[job_type_keys[1]][
                            job_type_keys[0]
                        ]

    def _set_initial_throughput(self, job_id, worker_type):
        assert not job_id.is_pair()
        if self._oracle_throughputs is not None:
            job_type = self._jobs[job_id].job_type
            scale_factor = self._jobs[job_id].scale_factor
            key = (job_type, scale_factor)
            self._throughputs[job_id][worker_type] = self._oracle_throughputs[
                worker_type
            ][key]["null"]
        else:
            self._throughputs[job_id][worker_type] = DEFAULT_THROUGHPUT

    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
    def _reset_time_run_so_far(self):
        """Reset _time_run_so_far so that all jobs receive new fair allocation
        from here on out.

        Requires self._scheduler_lock to be held when calling this function.
        """
        self._logger.debug("Resetting time run so far")
        current_time = self.get_current_timestamp()
        if self._last_reset_time == 0 and not self._simulate:
            elapsed_time_since_last_reset = current_time - self._init_timestamp
        else:
            elapsed_time_since_last_reset = (
                current_time - self._last_reset_time
            )
        for worker_type in self._worker_types:
            self._worker_time_so_far[worker_type] = 0.0
            for job_id in self._job_time_so_far:
                # _job_time_so_far keeps track of how long job_id has run on
                # worker_type since the last reset event.
                if worker_type not in self._job_time_so_far[job_id]:
                    time_received = 0.0
                else:
                    # Ignore the initial time recorded for the job.
                    time_received = self._job_time_so_far[job_id][
                        worker_type
                    ] - (self._time_per_iteration / 2.0)

                # Compute the time this job_id should have received since the
                # last reset event.
                if job_id not in self._allocation:
                    time_should_have_received = 0
                else:
                    time_should_have_received = (
                        self._allocation[job_id][worker_type]
                        * elapsed_time_since_last_reset
                    )

                # deficit is now just the difference between the time job_id
                # should have received, and how much it actually received.
                deficit = time_should_have_received - time_received
                if job_id not in self._deficits[worker_type]:
                    self._deficits[worker_type][job_id] = 0.0
                self._deficits[worker_type][job_id] += deficit

                self._job_time_so_far[job_id][worker_type] = (
                    self._time_per_iteration / 2.0
                )
                self._worker_time_so_far[worker_type] += self._job_time_so_far[
                    job_id
                ][worker_type]
        # Prints deficits every time allocation is reset.
        # self._print_deficits()
        self._last_reset_time = current_time
        self._allocation_changed_since_last_time_reset = False

    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
    def _add_to_priorities(self, job_id, worker_type=None):
        """Adds a job_id to each worker type's priority list.
        NOTE: Used when scheduling is performed in rounds.

        Requires self._scheduler_lock to be held when calling this function.

        Args:
            job_id: The job_id to add to the workers' priority data structures.
        """

        worker_types = self._worker_types
        if worker_type is not None:
            worker_types = [worker_type]
        for worker_type in worker_types:
            self._priorities[worker_type][job_id] = 0.0
            self._deficits[worker_type][job_id] = 0.0
            for other_job_id in self._throughputs:
                if other_job_id.is_pair() and job_id.overlaps_with(
                    other_job_id
                ):
                    self._priorities[worker_type][other_job_id] = 0.0
                    self._deficits[worker_type][other_job_id] = 0.0

    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
    def _remove_from_priorities(self, job_id):
        """Removes a job_id from each worker type's priority list.
        NOTE: Used when scheduling is performed in rounds.

        Requires self._scheduler_lock to be held when calling this function.

        Args:
           job_id: The job_id to remove from the workers' priority data structures.
        """
        for worker_type in self._worker_types:
            while True:
                found = False
                for other_job_id in self._priorities[worker_type]:
                    if job_id.overlaps_with(other_job_id):
                        del self._priorities[worker_type][other_job_id]
                        del self._deficits[worker_type][other_job_id]
                        found = True
                        break
                if not found:
                    break

    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
    def _update_priorities(self):
        """Updates each per-worker priority data structure.

        Re-sorts the data structure of each worker to compute the next job to run.
        For a given worker w_i, the next job to be scheduled will be the job
        that has so far received the smallest fraction of its computed
        fair allocation.
        Requires self._scheduler_lock to be held when calling this function.

        NOTE: Used when scheduling is performed in rounds.
        """
        current_time = self.get_current_timestamp()
        time_since_last_reset = current_time - self._last_reset_time
        reset_interval_elapsed = (
            time_since_last_reset
            >= self._minimum_time_between_allocation_resets
        )
        need_to_reset_time_run_so_far = (
            reset_interval_elapsed or self._last_reset_time == 0
        )
        if self._simulate:
            need_to_reset_time_run_so_far = (
                self._need_to_update_allocation
                and need_to_reset_time_run_so_far
            )
        else:
            need_to_reset_time_run_so_far = (
                self._allocation_changed_since_last_time_reset
                and need_to_reset_time_run_so_far
            )
        if need_to_reset_time_run_so_far:
            self._reset_time_run_so_far()
            # In simulation mode, wait for allocation computation to complete
            # before proceeding.
            if self._simulate:
                self._allocation = self._compute_allocation()
                self._need_to_update_allocation = False

        # Account for time elapsed since job was dispatched if running on a
        # physical cluster. Note that the total time for each job is the
        # sum of a) the time for all microtasks that have finished
        # (accounted for by self._job_time_so_far), and b) the unaccounted time
        # for all microtasks that are currently running (elapsed_job_time).
        if not self._simulate:
            elapsed_job_time = {}
            elapsed_worker_time = {}
            for job_id in self._current_worker_assignments:
                single_job_id = job_id.singletons()[0]
                if single_job_id not in self._per_job_latest_timestamps:
                    continue
                dispatch_time = self._per_job_latest_timestamps[single_job_id]
                if dispatch_time is None:
                    continue
                dispatch_time = max(dispatch_time, self._last_reset_time)
                elapsed_time = current_time - dispatch_time
                elapsed_job_time[job_id] = {}
                worker_ids = self._current_worker_assignments[job_id]
                worker_type = self._worker_id_to_worker_type_mapping[
                    worker_ids[0]
                ]
                if worker_type not in elapsed_job_time[job_id]:
                    elapsed_job_time[job_id][worker_type] = 0.0
                if worker_type not in elapsed_worker_time:
                    elapsed_worker_time[worker_type] = 0.0
                elapsed_job_time[job_id][worker_type] += elapsed_time
                elapsed_worker_time[worker_type] += elapsed_time

        # Stores the fraction of time spent running a job for each worker.
        fractions = {}

        # Compute priorities.
        for worker_type in self._worker_types:
            fractions[worker_type] = {}
            worker_time_so_far = self._worker_time_so_far[worker_type]
            for job_id in self._job_time_so_far:
                worker_time_so_far = self._worker_time_so_far[worker_type]
                if not self._simulate and worker_type in elapsed_worker_time:
                    worker_time_so_far += elapsed_worker_time[worker_type]
                if (
                    worker_time_so_far == 0.0
                    or worker_type not in self._job_time_so_far[job_id]
                ):
                    fraction = 0.0
                else:
                    job_time_so_far = self._job_time_so_far[job_id][
                        worker_type
                    ]
                    if not self._simulate:
                        if (
                            job_id in elapsed_job_time
                            and worker_type in elapsed_job_time[job_id]
                        ):
                            job_time_so_far += elapsed_job_time[job_id][
                                worker_type
                            ]
                    fraction = job_time_so_far / worker_time_so_far
                fractions[worker_type][job_id] = fraction
            for job_id in self._priorities[worker_type]:
                # Don't use inf so 2*new_priority > new_priority.
                #
                # Scale the default value by the allocation so that newly
                # added jobs run according to their respective allocations.
                if job_id not in self._allocation:
                    self._priorities[worker_type][job_id] = 0.0
                else:
                    new_priority = self._allocation[job_id][worker_type] * 1e9
                    if self._allocation[job_id][worker_type] == 0.0:
                        assert new_priority == 0
                    elif (
                        job_id.is_pair()
                        and (
                            self._throughputs[job_id][worker_type][0] == 0
                            or self._throughputs[job_id][worker_type][1] == 0
                        )
                    ) or (
                        not job_id.is_pair()
                        and self._throughputs[job_id][worker_type] == 0
                    ):
                        new_priority = 0
                    elif fractions[worker_type][job_id] > 0.0:
                        new_priority = (
                            self._allocation[job_id][worker_type]
                            / fractions[worker_type][job_id]
                        )
                    self._priorities[worker_type][job_id] = new_priority

    def _add_available_worker_id(self, worker_id):
        """Adds a worker_id to the list of available workers."""

        if not self._simulate:
            self._logger.debug(
                "Adding worker {0} back to queue...".format(worker_id)
            )
        self._available_worker_ids.put(worker_id)
        if not self._simulate:
            self._logger.debug(
                "Added worker {0} back to queue".format(worker_id)
            )

    def _remove_available_worker_id(self, worker_id=None):
        """Returns the worker_id of the next available worker."""

        if self._simulate:
            try:
                return self._available_worker_ids.get_nowait(item=worker_id)
            except queue.Empty as e:
                return None
        else:
            self._logger.debug(
                "Removing worker {0} from the queue...".format(worker_id)
            )
            ret = self._available_worker_ids.get(item=worker_id)
            if ret != worker_id:
                self._logger.warning(
                    "Worker {0} does not match requested worker {1}!".format(
                        ret, worker_id
                    )
                )
            self._logger.debug("Removed worker {0} from the queue".format(ret))
            return ret

    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
    def _get_remaining_steps(self, job_id):
        steps_run_so_far = self._total_steps_run[job_id]
        return self._jobs[job_id].total_steps - steps_run_so_far

    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
    def get_current_timestamp(self, in_seconds=False):
        if self._simulate:
            return self._current_timestamp
        else:
            if in_seconds:
                return time.time() - self._start_timestamp
            else:
                return time.time()

    """
    ======================================================================
       Callback methods called by workers.
    ======================================================================
    """

    def _register_worker_callback(
        self, worker_type, num_gpus=1, ip_addr=None, port=None
    ):
        """Registers a worker with the scheduler.

        Initializes state for a new worker and assigns it an id.
        The worker provides an IP address and port for its RPC server
        so that the scheduler can establish an RPC client for
        scheduler-to-worker communication. The worker also
        enumerates its available devices so that the scheduler
        can make fine-grained scheduling decisions.

        Args:
            worker_type: The type of GPU available on the worker.
            num_gpus: The number of GPUs available on the worker.
            ip_addr: IP address of the worker's RPC server.
            port: Port number for the worker's RPC server.
            devices: List of available devices on the worker.

        Returns:
            The worker_id of the newly registered worker.
        """

        self._logger.info(
            f"Trying to register worker with ip_addr {ip_addr}:{port}"
        )

        # Share a single RPC client for each GPU on the worker.
        if not self._simulate:
            rpc_client = scheduler_client.SchedulerRpcClient(ip_addr, port)
            self._all_rpc_clients.append(rpc_client)

        with self._scheduler_lock:
            # Update relevant data structures if worker type was
            # previously unseen.
            found = True
            if worker_type not in self._worker_type_to_worker_id_mapping:
                found = False
                self._worker_type_to_worker_id_mapping[worker_type] = []

            if not found:
                self._priorities[worker_type] = {}
                self._deficits[worker_type] = {}
                if self._per_worker_type_prices is not None:
                    self._per_worker_type_prices[
                        worker_type
                    ] = utils.get_latest_price_for_worker_type(
                        worker_type,
                        self.get_current_timestamp(in_seconds=True),
                        self._per_instance_type_spot_prices,
                        self._available_clouds,
                    )
                for job_id in self._jobs:
                    self._steps_run_so_far[job_id][worker_type] = 0
                    self._job_time_so_far[job_id][worker_type] = (
                        self._time_per_iteration / 2.0
                    )
                    self._set_initial_throughput(job_id, worker_type)
                    if self._job_packing:
                        self._populate_job_combination_metadata(
                            job_id, worker_type
                        )
                    # Add to relevant priority data structure.
                    self._add_to_priorities(job_id, worker_type=worker_type)
                if worker_type not in self._worker_time_so_far:
                    self._worker_time_so_far[worker_type] = 0.0

            # Update relevant data structures for each GPU available
            # on the worker.
            per_worker_ids = []
            for i in range(num_gpus):
                worker_id = self._worker_id_counter
                per_worker_ids.append(worker_id)
                self._worker_ids.append(worker_id)
                self._worker_id_counter += 1
                self._worker_types.add(worker_type)
                self._cumulative_worker_time_so_far[worker_id] = 0.0

                self._worker_id_to_worker_type_mapping[worker_id] = worker_type
                self._add_available_worker_id(worker_id)

                if worker_type not in self._cluster_spec:
                    self._cluster_spec[worker_type] = 0
                self._cluster_spec[worker_type] += 1
                if not self._simulate:
                    self._worker_connections[worker_id] = rpc_client

                self._worker_start_times[
                    worker_id
                ] = self.get_current_timestamp()
            self._worker_type_to_worker_id_mapping[worker_type].append(
                per_worker_ids
            )
            self._need_to_update_allocation = True
            self._scheduler_cv.notifyAll()

        return (per_worker_ids, self._time_per_iteration)

    def _init_job_callback(self, job_id):
        """Initializes a job.

           Args:
             job_id: The ID for the (single) job to initialize.
        """
        with self._scheduler_cv:
            # Job could have completed in previous round.
            if job_id not in self._jobs:
                return (0, 0, 0)

            # Wait if this job has been scheduled for the next round
            # but is still running in the previous round (possibly on
            # a different worker).
            while True:
                currently_active = False
                next_job_combination = None

                if self._next_worker_assignments is not None:
                    for job_combination in self._next_worker_assignments:
                        if job_id.overlaps_with(job_combination):
                            next_job_combination = job_combination
                            break

                if next_job_combination is not None:
                    # Check whether this job is blocked by a currently active
                    # job - this could be a job (combination) involving this
                    # job itself or this job's colocation partner in the
                    # upcoming round. For example, consider the following
                    # scenario:
                    #
                    # Round r: Job <0, 1> is scheduled
                    # Round r+1 : Job <1, 2> is scheduled
                    # Job 2 requests initialization partway through round r
                    #
                    # In this case, we would wait to intialize job 2 until
                    # jobs 0 and 1 complete so that job 2 can execute together
                    # with job 1.
                    for job_combination in self._current_worker_assignments:
                        for single_job_id in next_job_combination.singletons():
                            if single_job_id.overlaps_with(job_combination):
                                if (
                                    job_combination
                                    not in self._completed_jobs_in_current_round
                                ):
                                    currently_active = True
                                    break
                        if currently_active:
                            break

                if currently_active and next_job_combination is not None:
                    self._scheduler_cv.wait()
                else:
                    break

            # Record initializiation as latest job event.
            self._per_job_latest_timestamps[
                job_id
            ] = self.get_current_timestamp()

            for single_job_id in job_id.singletons():
                self._running_jobs.add(single_job_id)

            # Determine initial lease.
            scale_factor = self._jobs[job_id].scale_factor
            remaining_steps = self._get_remaining_steps(job_id)
            remaining_steps = int(math.ceil(remaining_steps / scale_factor))

            current_time = self.get_current_timestamp()
            current_round_end_time = (
                self._current_round_start_time + self._time_per_iteration
            )
            remaining_time_in_current_round = max(
                current_round_end_time - current_time, 0
            )

            self._logger.debug(
                f"Job {job_id}: remaining_time_in_current_round ({remaining_time_in_current_round}) is the max of {current_round_end_time} - {current_time} = {current_round_end_time - current_time} and 0"
            )
            self._logger.info(
                f"self._next_worker_assignments is {self._next_worker_assignments}"
            )
            self._logger.info(
                f"next_job_combination is {next_job_combination}"
            )

            # # Return a tuple of (steps, duration, extra time) as the initial
            # # lease. Extra time is granted if the job was scheduled for the
            # # upcoming round but is being initialized in the current round.
            # if (self._next_worker_assignments is not None and
            #     next_job_combination is not None):
            #     if self._jobs[job_id].scale_factor == 1:
            #         # Job was dispatched early, so add additional time.
            #         self._logger.debug(f"Sending lease 1 to job {job_id} with (steps, duration, extra time): ({remaining_steps}, {self._time_per_iteration}, {remaining_time_in_current_round})")
            #         return (remaining_steps, self._time_per_iteration,
            #                 remaining_time_in_current_round)
            #     else:  # multi-GPU job
            #         """
            #         Say we have four workers waiting to be run on GPU 0,1,2,3 in round r+1.
            #         In round r, if another single-GPU job finished completely on GPU 0, it will
            #         relinquish GPU 0, and then worker 0 will start running on GPU 0, asking for an init lease.
            #         As it's dispatched early, it will have additional extra_time.
            #         However, at this point, GPU 1/2/3 are still busy, so worker 1/2/3 will wait for round r+1
            #         to initialize, getting leases without extra_time.
            #         So, going back to when worker 0 got the lease -- even though it was dispatched early, it
            #         could not start the training, and self._prev_time in gavel_iterator is only set when
            #         all four workers arrive. As a result, it will ask for a lease later than the other workers,
            #         creating the problem of "3 / 4 workers have ...".
            #         To prevent this, we do not grant extra_time for multi-GPU jobs that initialized early,
            #         and give the duration 10 seconds off to account for the training initialization overhead.
            #         A small price to pay for salvation...
            #         """
            #         self._logger.debug(f"Sending lease 2 to job {job_id} with (steps, duration, extra time): ({remaining_steps}, {self._time_per_iteration - 10}, {0})")
            #         return (remaining_steps, self._time_per_iteration - 10, 0)
            # else:
            #     if remaining_time_in_current_round > 0:
            #         self._logger.debug(f"Sending lease 3 to job {job_id} with (steps, duration, extra time): ({remaining_steps}, {remaining_time_in_current_round}, {0})")
            #         return (remaining_steps, remaining_time_in_current_round, 0)
            #     else:
            #         """
            #         NOTE: The time difference between the start of round r and the start
            #         of round r+1 may not equal the round duration.
            #         In this case, if a job is dispatched in round r (and will be
            #         run in round r+1), it may ask for an initial lease after
            #         round r ends and before round r+1 starts. At this time,
            #         end_round() just finished executing, so remaining_time_in_current_round
            #         would be the max of for example -1.7470808029174805 and 0.
            #         This would result in a lease with valid num_steps but 0 duration,
            #         leading to a micro-task failure during job initialization in round r+1.
            #         """
            #         self._logger.warning(f"Job {job_id} asking for lease between round r ends and round r+1 starts")
            #         self._logger.debug(f"Sending lease 4 to job {job_id} with (steps, duration, extra time): ({remaining_steps}, {self._time_per_iteration - EARLY_INIT_THRESHOLD}, {remaining_time_in_current_round})")
            #         return (remaining_steps, self._time_per_iteration - EARLY_INIT_THRESHOLD, remaining_time_in_current_round)

            # Return a tuple of (steps, duration, extra time) as the initial
            # lease. Extra time is granted if the job was scheduled for the
            # upcoming round but is being initialized in the current round.
            if (
                self._next_worker_assignments is not None
                and next_job_combination is not None
            ):
                # Job was dispatched early, so add additional time.
                self._logger.debug(
                    f"Sending init lease to job {job_id} with (steps, duration, extra time): ({remaining_steps}, {self._time_per_iteration}, {remaining_time_in_current_round}) (type 1)"
                )
                return (
                    remaining_steps,
                    self._time_per_iteration,
                    remaining_time_in_current_round,
                )
            else:
                if remaining_time_in_current_round > 0:
                    self._logger.debug(
                        f"Sending init lease to job {job_id} with (steps, duration, extra time): ({remaining_steps}, {remaining_time_in_current_round}, {0}) (type 2)"
                    )
                    return (
                        remaining_steps,
                        remaining_time_in_current_round,
                        0,
                    )
                else:
                    self._logger.debug(
                        f"Sending init lease to job {job_id} with (steps, duration, extra time): ({remaining_steps}, {self._time_per_iteration - EARLY_INIT_THRESHOLD}, {remaining_time_in_current_round}) (type 3)"
                    )
                    return (
                        remaining_steps,
                        self._time_per_iteration - EARLY_INIT_THRESHOLD,
                        remaining_time_in_current_round,
                    )

    def _update_lease_callback(
        self, job_id, worker_id, steps, duration, max_steps, max_duration
    ):
        # NOTE: when a worker requests a new lease, the steps are passed in
        # with the variable steps. We can use this to update the epoch progress
        # for shockwave, but in done_callback, we should only add to steps_run_so_far
        # the number of steps ran in the last round before the lease expiration
        with self._scheduler_lock:
            # include run_time_so_far and deadline in the lease to prevent jobs from running overtime
            run_time_so_far = int(
                sum(self._cumulative_run_time[job_id].values())
                / self._jobs[job_id].scale_factor
            )
            deadline = int(self._jobs[job_id].duration * 1.5)

            if job_id not in self._lease_update_requests:
                self._lease_update_requests[job_id] = []
            update_id = len(self._lease_update_requests[job_id])
            self._lease_update_requests[job_id].append(
                (steps, duration, max_steps, max_duration)
            )

            # Round the remaining steps to the nearest multiple of scale_factor.
            scale_factor = self._jobs[job_id].scale_factor
            remaining_steps = self._get_remaining_steps(job_id)
            remaining_steps = int(math.ceil(remaining_steps / scale_factor))
            current_time = self.get_current_timestamp()
            current_round_end_time = (
                self._current_round_start_time + self._time_per_iteration
            )
            remaining_time_in_current_round = (
                current_round_end_time - current_time
            )
            remaining_time_in_current_round = max(
                0, remaining_time_in_current_round
            )

        self._logger.info(
            f"Job {job_id}: sending updated lease, run_time_so_far: {run_time_so_far}, deadline: {deadline}"
        )

        # if shockwave, add to _steps_run_in_current_lease to maintain epoch progress
        # if self._policy.name == "shockwave":
        integer_job_id = job_id.integer_job_id()
        self._steps_run_in_current_lease[integer_job_id] = (
            steps * self._jobs[job_id].scale_factor
        )  # aggregate the #iterations reported on each worker

        if steps == 0 or duration == 0:
            self._logger.debug(
                f"Sending updated lease to job {job_id} with (steps, duration): ({remaining_steps}, {remaining_time_in_current_round}) (type 1)"
            )
            return (
                remaining_steps,
                remaining_time_in_current_round,
                run_time_so_far,
                deadline,
            )

        # Extend the lease if the job has been placed on the same workers
        # for the upcoming round.
        with self._scheduler_lock:
            # TODO: Remove scan of self._jobs_with_extended_lease.
            for job_id_combination in self._jobs_with_extended_lease:
                if job_id.overlaps_with(job_id_combination):
                    updated_lease_duration = duration
                    updated_lease_duration += remaining_time_in_current_round
                    updated_lease_duration += self._time_per_iteration
                    self._logger.debug(
                        f"Sending updated lease to job {job_id} with (steps, duration): ({max_steps}, {updated_lease_duration}) (type 2)"
                    )
                    return (
                        max_steps,
                        updated_lease_duration,
                        run_time_so_far,
                        deadline,
                    )

        if scale_factor == 1:
            self._logger.debug(
                f"Sending updated lease to job {job_id} with (steps, duration): ({max_steps}, {duration + remaining_time_in_current_round}) (type 3)"
            )
            return (
                max_steps,
                duration + remaining_time_in_current_round,
                run_time_so_far,
                deadline,
            )
        else:
            if update_id == 0:
                assert self._max_steps[job_id] is None

            # The first worker to request a lease update computes the new
            # lease for all workers.
            if update_id == 0:
                with self._scheduler_lock:
                    throughput = steps / duration
                    self._max_steps[job_id] = min(
                        remaining_steps,
                        steps
                        + int(remaining_time_in_current_round * throughput),
                    )
                    self._logger.debug(
                        f"Sending updated lease to job {job_id} with (steps, duration): ({self._max_steps[job_id]}, INFINITY) (update_id {update_id}) (type 4)"
                    )
                    return (
                        self._max_steps[job_id],
                        INFINITY,
                        run_time_so_far,
                        deadline,
                    )
                    # return (self._max_steps[job_id], duration + remaining_time_in_current_round, run_time_so_far, deadline)
            else:
                # Wait for the first update to complete.
                while True:
                    with self._scheduler_lock:
                        max_steps = self._max_steps[job_id]
                        if max_steps is not None:
                            break
                    # TODO: Sleep for less time?
                    self._logger.debug(
                        "Job {0} (worker {1}) waiting for "
                        "lease...".format(job_id, worker_id)
                    )
                    time.sleep(1)
                assert max_steps is not None
                self._logger.debug(
                    f"Sending updated lease to job {job_id} with (steps, duration): ({max_steps}, INFINITY) (update_id {update_id}) (type 5)"
                )
                return (max_steps, INFINITY, run_time_so_far, deadline)
                # return (max_steps, duration + remaining_time_in_current_round, run_time_so_far, deadline)

    def _update_resource_requirement_callback(
        self, job_id, worker_id, big_bs, small_bs
    ):
        with self._scheduler_lock:
            self._logger.info(
                "Scheduler received resource requirement update "
                f"request from job_id {job_id} on worker_id {worker_id} "
                f"changing to {'big' if big_bs else 'small'} batch size"
            )
            assert big_bs != small_bs
            # mark the flag so that during the job done callback,
            # all metadata are updated
            if big_bs:
                self._bs_flags[job_id]["big_bs"] = True
            else:
                self._bs_flags[job_id]["small_bs"] = True

            self._scheduler_cv.notifyAll()

    def _kill_job(self, job_id):
        with self._scheduler_cv:
            if job_id not in self._current_worker_assignments:
                raise RuntimeError(
                    "Trying to kill job ({0}) that is not active "
                    "in this round!".format(job_id)
                )
            elif job_id not in self._completion_events:
                if job_id not in self._completed_jobs_in_current_round:
                    raise RuntimeError(
                        "Completion event for job {0} is not active "
                        "even though job has not completed!".format(job_id)
                    )
                elif job_id not in self._jobs_with_extended_lease:
                    # Job has already completed normally.
                    return
            self._logger.info("Killing job {0}".format(job_id))
            worker_ids = self._current_worker_assignments[job_id]
            servers = set()
            for worker_id in worker_ids:
                rpc_client = self._worker_connections[worker_id]
                server = (rpc_client.addr, rpc_client.port)
                if server not in servers:
                    for single_job_id in job_id.singletons():
                        self._logger.debug(
                            "Killing job {0} on server {1}:{2}".format(
                                single_job_id, rpc_client.addr, rpc_client.port
                            )
                        )
                        rpc_client.kill_job(single_job_id)
                    servers.add(server)
            del self._completion_events[job_id]

            # Wait for the killed job to send a completion notification and
            # proceed if no notification is sent.
            prev_round = self._num_completed_rounds
            self._logger.debug(
                "Waiting for job {0} to be killed...".format(job_id)
            )
            self._scheduler_cv.wait(timeout=30)
            self._logger.debug(
                "Checking if job {0} was killed...".format(job_id)
            )
            new_round = self._num_completed_rounds

            successful_kill = (
                new_round != prev_round
                or job_id in self._completed_jobs_in_current_round
            )
            if successful_kill:
                self._logger.debug(
                    "Job {0} was successfully killed in round {1}".format(
                        job_id, prev_round
                    )
                )
            else:
                self._logger.debug(
                    "Job {0} was killed but did not complete!".format(job_id)
                )
                all_worker_ids = set(self._current_worker_assignments[job_id])
                completed_worker_ids = set()
                for update in self._in_progress_updates[job_id]:
                    worker_id = update[0]
                    completed_worker_ids.add(worker_id)
                worker_ids_to_complete = all_worker_ids.difference(
                    completed_worker_ids
                )
                self._logger.debug(
                    "Need to send done callbacks for the following "
                    "workers for job {0}: {1}".format(
                        job_id, worker_ids_to_complete
                    )
                )
        if not successful_kill:
            x = [0 for _ in range(len(job_id.singletons()))]
            for worker_id in worker_ids_to_complete:
                self._logger.debug(
                    "Sending done callback for worker {0} "
                    "for job {1}".format(worker_id, job_id)
                )
                self._done_callback(job_id, worker_id, x, x)

    def _done_callback_extended_lease(self, job_id):
        kill_job = False
        self._logger.debug(f"[Job {job_id}] _done_callback_extended_lease")

        with self._scheduler_cv:
            is_active = any([x in self._jobs for x in job_id.singletons()])
            if not is_active:
                return

            self._logger.debug(
                "Trying to complete job {0} which had an "
                "extended lease...".format(job_id)
            )

            scale_factor = self._jobs[job_id.singletons()[0]].scale_factor
            num_updates = []
            for single_job_id in job_id.singletons():
                num_updates.append(
                    len(self._lease_update_requests[single_job_id])
                )
            # TODO: figure out why there are sometimes multiple lease requests in 1 round for a single-GPU job
            updated_lease = (
                min(num_updates) >= scale_factor
            )  # use >= instead of == to handle multiple lease requests in 1 round by 1 worker
            for i, single_job_id in enumerate(job_id.singletons()):
                self._logger.debug(
                    "{0} / {1} worker(s) for job {2} have "
                    "requested a lease update this "
                    "round".format(num_updates[i], scale_factor, single_job_id)
                )
            if not updated_lease:
                # Job has not requested lease updates so assume it has failed.
                self._logger.error(
                    "Job {0} had an extended lease but has "
                    "been unresponsive".format(job_id)
                )
                kill_job = True
            elif job_id in self._completion_events:
                self._logger.info("Completing job {0}".format(job_id))

                # Mark job as completed.
                self._completed_jobs_in_current_round.add(job_id)
                del self._completion_events[job_id]

                # Reset metadata.
                # NOTE: We do not reset self._in_progress_updates here as
                # multi-GPU jobs might have partially completed updates.
                for single_job_id in job_id.singletons():
                    self._lease_update_requests[single_job_id] = []
                    self._max_steps[single_job_id] = None

            if not kill_job:
                self._scheduler_cv.notifyAll()

        if kill_job:
            self._logger.error("!" * 100)
            self._kill_job(job_id)

    def _done_callback(
        self,
        job_id,
        worker_id,
        all_num_steps,
        all_execution_times,
        all_iterator_logs=None,
    ):
        """Handles completion of a scheduled job.

        Updates the running total of completed steps and time spent on each
        worker, for every currently active application. Removes the job from
        the scheduler if the job has finished all its requested steps. Adds
        the worker back to the list of available workers.

        Args:
            job_id: The id of the completed job(s).
            worker_id: The id of the worker where the job(s) were completed.
            all_num_steps: List of the number of steps each job ran for.
            all_execution_times: List of the duration each job ran for.
            all_iterator_logs: List of the GavelIterator logs for each job.
        """
        # NOTE: the resource requirement update request will be processed here
        # 1. update the batch size used in the job type and training command
        # 2. update the throughput of the new batch size according to the pre-profiled
        # oracle throughput file
        # 3. update the MPS percentage in the next round of scheduling
        # 4. scale up/down the number of steps (iterations)
        # 5. ping the allocation thread to have it update the allocation

        # self._logger.info(f"done_callback invoked: job {job_id}, worker {worker_id}, all_num_steps {all_num_steps}, all_execution_times {all_execution_times}")
        to_remove = []
        with self._scheduler_lock:
            # update the cumulative run time for current job
            if worker_id not in self._cumulative_run_time[job_id]:
                self._cumulative_run_time[job_id][worker_id] = 0
            self._cumulative_run_time[job_id][worker_id] += np.max(
                all_execution_times
            )

            if job_id in self._jobs.keys():
                run_time_so_far = (
                    sum(self._cumulative_run_time[job_id].values())
                    / self._jobs[job_id].scale_factor
                )
                # self._logger.info(f"[Job {job_id}] self._cumulative_run_time is {self._cumulative_run_time[job_id]}")
                # NOTE: run_time_so_far may be under-reported, as other workers in a distributed job might haven't called done_callback
                self._logger.debug(
                    f"[Job {job_id}] After worker {worker_id} called done_callback, cumulative run time so far is {run_time_so_far}; Deadline is {self._jobs[job_id].duration}"
                )

                # TODO: double check if this deprecated utility is useless, and if so, clean up
                is_over_deadline = run_time_so_far > int(
                    self._jobs[job_id].duration * 1.5
                )
                if is_over_deadline:
                    self._logger.warning(
                        f"[Job {job_id}] Over the deadline, manually removing from Gavel"
                    )
            else:  # job has already completed
                is_over_deadline = True

            # If current round is r, job might have been dispatched for
            # round r+1 and completed before round r is done. If so,
            # wait for round r to finish before proceeding.
            if not self._simulate:
                while (
                    job_id not in self._current_worker_assignments
                    or job_id in self._completed_jobs_in_current_round
                ):
                    if job_id not in self._current_worker_assignments and (
                        self._next_worker_assignments is not None
                        and job_id not in self._next_worker_assignments
                    ):
                        self._logger.warning(
                            "Discarding completion notification for job {0} "
                            "as it is not currently scheduled".format(job_id)
                        )
                        return
                    self._logger.debug(
                        "Waiting to complete job {0}...".format(job_id)
                    )
                    self._scheduler_cv.wait()

            # Check whether jobs are still active as jobs might have
            # completed after being dispatched for the subsequent round.
            is_active = {}
            for single_job_id in job_id.singletons():
                is_active[single_job_id] = single_job_id in self._jobs
            if not any(is_active.values()):
                self._logger.info(
                    "Job {job_id} (worker {worker_id}) has "
                    "already completed!".format(
                        job_id=job_id, worker_id=worker_id
                    )
                )
                return

            current_timestamp = self.get_current_timestamp()
            worker_type = self._worker_id_to_worker_type_mapping[worker_id]
            self._add_available_worker_id(worker_id)

            scale_factor = len(self._current_worker_assignments[job_id])
            self._in_progress_updates[job_id].append(
                (
                    worker_id,
                    all_num_steps,
                    all_execution_times,
                    all_iterator_logs,
                )
            )
            if len(self._in_progress_updates[job_id]) < scale_factor:
                return
            else:
                # Sort updates in order of increasing worker ID.
                self._in_progress_updates[job_id].sort(key=lambda x: x[0])

                # If a job completes before the end of the round, cancel the
                # job's completion event.
                """
                self._logger.debug(
                    'Current active completion events: {0}'.format(
                        self._completion_events.keys()))
                """

                if job_id in self._completion_events:
                    event = self._completion_events[job_id]
                    try:
                        self._completion_event_scheduler.cancel(event)
                    except ValueError as e:
                        # Completion event might have been triggered after
                        # entering done_callback.
                        self._logger.error(
                            f"ValueError when removing the completion event of job {job_id}: {e}"
                        )
                        self._logger.error(
                            "Completion event might have been triggered after entering done_callback"
                        )
                        pass
                    self._logger.debug(
                        "Removing completion event for job {0}".format(job_id)
                    )
                    # if self._bs_flags[job_id]["big_bs"] or self._bs_flags[job_id]["small_bs"]:
                    #     self._jobs_with_extended_lease_and_bs_request.append(job_id)
                    del self._completion_events[job_id]
                self._completed_jobs_in_current_round.add(job_id)
                micro_task_succeeded = True
                all_worker_ids = [
                    x[0] for x in self._in_progress_updates[job_id]
                ]
                all_worker_ids.sort()
                all_num_steps = [0] * len(job_id.singletons())
                all_execution_times = [0] * len(job_id.singletons())
                for i, update in enumerate(self._in_progress_updates[job_id]):
                    all_num_steps_ = update[1]
                    all_execution_times_ = update[2]
                    all_iterator_logs_ = update[3]
                    for j, single_job_id in enumerate(job_id.singletons()):
                        if not is_active[single_job_id]:
                            continue
                        # elif (all_num_steps_[j] <= 0 or   # use and instead of or to prevent microtask failures due to rounding errors in simulation
                        elif (
                            all_num_steps_[j] <= 0
                            and all_execution_times_[j] <= 0
                        ):
                            micro_task_succeeded = False
                            self._logger.debug(
                                f"Micro-task failed. all_num_steps_[j]: {all_num_steps_[j]}, all_execution_times_[j]: {all_execution_times_[j]}"
                            )
                            break
                    for j, single_job_id in enumerate(job_id.singletons()):
                        all_num_steps[j] += all_num_steps_[j]
                        all_execution_times[j] = max(
                            all_execution_times[j], all_execution_times_[j]
                        )
                        if all_iterator_logs_ is not None:
                            self._job_timelines[single_job_id][i].extend(
                                all_iterator_logs_[j].split("\n")
                            )

            # Reset metadata.
            self._in_progress_updates[job_id] = []
            for single_job_id in job_id.singletons():
                self._lease_update_requests[single_job_id] = []
                self._max_steps[single_job_id] = None

            if not self._simulate:
                # NOTE: We update the timestamp before calling this
                # function in simulation.
                for single_job_id in job_id.singletons():
                    if is_active[single_job_id]:
                        self._per_job_latest_timestamps[
                            single_job_id
                        ] = self.get_current_timestamp()

            if not micro_task_succeeded:
                # Micro-task failed.
                self._logger.info(
                    "[Micro-task failed]\tJob ID: {job_id}".format(
                        job_id=job_id
                    )
                )
                if not job_id.is_pair() and is_active[job_id]:
                    self._num_failures_per_job[
                        job_id
                    ] += 1  # TODO: did this consider multi-GPU jobs where each worker calls done_callback?
                    self._logger.debug(
                        f"[Job {job_id}] _num_failures_per_job is now {self._num_failures_per_job[job_id]} after worker {worker_id}'s report"
                    )
                    if (
                        self._num_failures_per_job[job_id]
                        >= MAX_FAILED_ATTEMPTS
                    ):
                        start_time = self._per_job_start_timestamps[job_id]
                        finish_time = self._per_job_latest_timestamps[job_id]
                        duration = finish_time - start_time
                        self._logger.info(
                            "[Job failed]\tJob ID: {job_id}\t"
                            "Start timestamp: {start_timestamp:.2f}\t"
                            "End timestamp: {end_timestamp:.2f}\t"
                            "Duration: {duration:.2f}".format(
                                job_id=job_id,
                                start_timestamp=start_time,
                                end_timestamp=finish_time,
                                duration=duration,
                            )
                        )
                        to_remove.append(job_id)
                self._need_to_update_allocation = True

            else:

                """
                self._logger.info(
                    '[Micro-task succeeded]\t'
                    'Job ID: {job_id}\tWorker type: {worker_type}\t'
                    'Worker ID(s): {worker_ids}'.format(
                        job_id=job_id, worker_type=worker_type,
                        worker_ids=str(all_worker_ids)))
                """

                self._num_failures_per_job[job_id] = 0
                for single_job_id, num_steps, execution_time in zip(
                    job_id.singletons(), all_num_steps, all_execution_times
                ):
                    if not is_active[single_job_id]:
                        self._logger.debug(
                            "Job {0} is not active, not "
                            "updating metadata".format(single_job_id)
                        )
                        continue
                    if self._per_worker_type_prices is not None:
                        self._job_cost_so_far[single_job_id] += (
                            self._per_worker_type_prices[worker_type]
                            * execution_time
                            / 3600.0
                            * scale_factor
                        )
                        job_cost_so_far = self._job_cost_so_far[single_job_id]
                        self._logger.info(
                            "Job {job_id} cost so far: ${cost:.2f}".format(
                                job_id=single_job_id, cost=job_cost_so_far
                            )
                        )
                    # Job may be multi-GPU, and have already been removed from
                    # running_jobs by another worker.
                    if single_job_id in self._running_jobs:
                        self._running_jobs.remove(single_job_id)
                        self._steps_run_so_far[single_job_id][
                            worker_type
                        ] += num_steps
                        self._total_steps_run[single_job_id] += num_steps
                        # shockwave: clear self._steps_run_in_current_lease if lease gets terminated
                        # if self._policy.name == "shockwave":
                        self._steps_run_in_current_lease[single_job_id] = 0
                        remaining_steps = self._get_remaining_steps(
                            single_job_id
                        )

                        # if a job is running over time, force it to complete
                        if remaining_steps <= 0 or is_over_deadline:
                            # if remaining_steps <= 0 or (is_over_deadline and not self._simulate):
                            start_time = self._per_job_start_timestamps[
                                single_job_id
                            ]
                            finish_time = self._per_job_latest_timestamps[
                                single_job_id
                            ]
                            duration = finish_time - start_time
                            self._logger.info(
                                "[Job succeeded]\tJob ID: {job_id}\t"
                                "Start timestamp: {start_timestamp:.2f}\t"
                                "End timestamp: {end_timestamp:.2f}\t"
                                "Duration: {duration:.2f}".format(
                                    job_id=single_job_id,
                                    start_timestamp=start_time,
                                    end_timestamp=finish_time,
                                    duration=duration,
                                )
                            )
                            to_remove.append(single_job_id)
                        else:
                            if not self._simulate:
                                self._logger.debug(
                                    "Job {job_id} has {steps} "
                                    "remaining steps".format(
                                        job_id=single_job_id,
                                        steps=remaining_steps,
                                    )
                                )

                # If we just ran co-located jobs, use the maximum of the
                # individual execution times.
                max_execution_time = np.max(all_execution_times)
                # Job may be multi-GPU, and have already been marked complete
                # by another worker.
                if job_id in self._job_time_so_far:
                    self._job_time_so_far[job_id][
                        worker_type
                    ] += max_execution_time
                    self._worker_time_so_far[worker_type] += max_execution_time
                for worker_id in all_worker_ids:
                    self._cumulative_worker_time_so_far[
                        worker_id
                    ] += max_execution_time

            self._update_throughput(
                job_id, worker_type, all_num_steps, all_execution_times
            )

            if type(job_id) is int:
                job_ids = [job_id]
            else:
                job_ids = [job_id[0], job_id[1]]

            # accordion/gns: scale the batch size and num of iterations for current job(s)
            for jobid in job_ids:
                self._scale_bs_and_iters(jobid)

            for single_job_id in to_remove:
                self._logger.debug(
                    f"***************job {single_job_id} finished, removing from scheduler"
                )
                self._remove_job(single_job_id)
                # if self._policy.name == "shockwave" and not self._simulate:
                #     self._shockwave_job_completed_flag = True
                #     self._shockwave_scheduler.remove_metadata(single_job_id)

                if self._policy.name == "shockwave":
                    self._shockwave_scheduler.remove_metadata(single_job_id)

            # Schedule the job for re-dispatching if necessary.
            is_active = any([x in self._jobs for x in job_id.singletons()])
            if is_active and job_id in self._jobs_with_extended_lease:
                self._redispatched_worker_assignments[
                    job_id
                ] = self._next_worker_assignments[job_id]

            for job_id in job_ids:
                if job_id is None:
                    continue
                if (
                    self._bs_flags[job_id]["big_bs"]
                    or self._bs_flags[job_id]["small_bs"]
                ):
                    # ping the allocation thread to have it re-compute the allocation
                    # to be used in the next round of scheduling
                    # TOOD: Is this unfair for shockwave?
                    self._need_to_update_allocation = True

                # reset the flags that indicate resource requirement change
                self._bs_flags[job_id]["big_bs"] = False
                self._bs_flags[job_id]["small_bs"] = False

            self._scheduler_cv.notifyAll()

    """
    ======================================================================
       Shockwave additions
    ======================================================================
    """

    def _get_num_epochs(self, job_type, batch_size, num_steps):
        """Takes in the specifications of a job, calculate and 
        return the number of total epochs
        """
        model = job_type[: job_type.find(" ")]
        dataset_size = dataset_size_dict[model]
        return math.ceil(num_steps / math.ceil(dataset_size / batch_size))

    def _scale_bs_and_iters(self, job_id):
        """Scales up/down the batch size and number of iterations
        for a job

        Args:
            job_id (int): ID of a single job
        """
        if job_id is None:
            return
        if (
            self._bs_flags[job_id]["big_bs"]
            or self._bs_flags[job_id]["small_bs"]
        ):
            # update the batch size used in the job type
            # and training command
            assert self._oracle_throughputs is not None
            assert type(job_id) is int or not job_id.is_pair()

            # excerpt the old batch size
            old_command = self._jobs[job_id].command
            old_bs = self._jobs[job_id].batch_size
            model = self._jobs[job_id].model
            mode = self._jobs[job_id].mode
            original_batch_size = self._original_bs[job_id]

            max_bs_dict = {  # max possible bs of jobs that support bs scaling
                "LM": 80,
                "ResNet-18": 256,
                "ResNet-50": 128,
                "Recommendation": 8192,
            }

            # if (model == "LM" and original_batch_size == 80) \
            #     or (model == "ResNet-18" and original_batch_size == 256) \
            #     or (model == "ResNet-50" and original_batch_size == 128) \
            #     or (model == "Recommendation" and original_batch_size == 8192):
            if (
                model in max_bs_dict.keys()
                and original_batch_size == max_bs_dict[model]
            ):
                # oom/going beyond the gavel pre-profiled throughputs, no space for speedups due to batch size scaling
                # reject the batch size update
                self._bs_flags[job_id]["big_bs"] = False
                self._bs_flags[job_id]["small_bs"] = False
                return
            else:
                if mode == "gns":
                    # batch size strictly doubles up
                    assert self._bs_flags[job_id]["big_bs"]
                    new_bs = 2 * old_bs
                elif mode == "accordion":
                    if self._bs_flags[job_id]["big_bs"]:
                        # scale bs to the maximum possible
                        new_bs = max_bs_dict[model]
                    else:  # self._bs_flags[job_id]["small_bs"]
                        # scale down to original/initial bs
                        new_bs = original_batch_size
                elif mode == "static":
                    # no bs scaling, this is a failsafe mechanism
                    new_bs = old_bs

            self._logger.debug(f"old_command is {old_command}")
            self._logger.debug(f"old_bs before conversion is {old_bs}")
            bs_scaling_factor = new_bs / old_bs
            # new_bs = int(old_bs * (bs_scaling_factor if self._bs_flags[job_id]["big_bs"] else 1 / bs_scaling_factor))
            self._logger.info(
                f"Updating job {job_id} bs: {old_bs} -> {new_bs}, bs scaling factor {bs_scaling_factor}"
            )
            self._jobs[job_id].update_bs(new_bs)

            # update the throughput to make it reflect that of the new batch size
            # the new throughput is read from the pre-profiled throughput file
            for worker_type in self._worker_types:
                scale_factor = self._jobs[job_id].scale_factor
                new_job_type = self._jobs[job_id].job_type
                key = (new_job_type, scale_factor)
                old_throughput = self._throughputs[job_id][worker_type]

                if key not in self._oracle_throughputs[worker_type].keys():
                    # fault tolerance: prevent jobs from asking for new batch sizes
                    # that are not recorded in the throughput file
                    self._logger.error(
                        f"Job {job_id} asking for unreasonable bs rescale ({key}), resuming with old bs"
                    )
                    self._bs_flags[job_id]["big_bs"] = False
                    self._bs_flags[job_id]["small_bs"] = False
                    self._logger.error(
                        f"Reverting job {job_id} bs: {new_bs} -> {old_bs}"
                    )
                    self._jobs[job_id].update_bs(
                        old_bs
                    )  # this reverts update_bs(new_bs)
                    return

                new_throughput = self._oracle_throughputs[worker_type][key][
                    "null"
                ]
                self._logger.debug(
                    "Updating throughput once again to reflect the new batch size"
                )
                self._logger.debug(
                    "Job {job_id} throughput on worker type {worker_type}: "
                    "[{orig}] -> [{updated}]".format(
                        job_id=job_id,
                        worker_type=worker_type,
                        orig=str(old_throughput),
                        updated=str(new_throughput),
                    )
                )
                self._throughputs[job_id][worker_type] = new_throughput

            # update the MPS thread percentage in the next round of scheduling
            # for now, big_bs jobs use 100% while small_bs jobs use 50%.
            # NOTE: we are not messing with the MPS thread percentages for now
            percentage = 100 if self._bs_flags[job_id]["big_bs"] else 100
            # percentage = 100 if self._bs_flags[job_id]["big_bs"] else 50
            self._logger.debug(
                f"Updating job {job_id} percentage: {self._jobs[job_id].mps_thread_percentage}% -> {percentage}%"
            )
            self._jobs[job_id].set_mps_thread_percentage(percentage)

            # scale down/up the number of steps
            iterations_scaling_factor = 1 / bs_scaling_factor

            total_steps = self._jobs[job_id].total_steps
            total_steps_run = self._total_steps_run[job_id]
            steps_run_so_far = self._steps_run_so_far[job_id]["v100"]

            """
            NOTE: Preserve the number of epochs between batch size rescale. For example,
            before scaling, a Recommendation job (len(dataset) = 117907)
            had bs = 4096 & num_iterations = 3500 -> 
            num_epochs = math.ceil(num_iterations / num_iterations_in_one_epoch) ->
            = math.ceil(3500 / math.ceil(117907 / 4096)) = 121.
            If we directly scale num_iterations by scale_factor, new_total_steps would become 1750.
            num_iterations_in_one_epoch = math.ceil(117907 / 8192) = 15 ->
            num_epochs becomes math.ceil(1750 / 15) = 117, losing 4 epochs of training.
            Instead, we preserve num_epochs = 121 by using new_total_steps = 
            121 epochs * 15 iterations/epoch = 1815 iterations.
            """

            old_total_epochs = math.ceil(
                total_steps / math.ceil(dataset_size_dict[model] / old_bs)
            )
            new_total_steps = math.ceil(
                total_steps * iterations_scaling_factor
            )
            new_total_epochs = math.ceil(
                new_total_steps / math.ceil(dataset_size_dict[model] / new_bs)
            )
            if new_total_epochs == old_total_epochs:
                self._logger.debug(
                    f"Job {job_id}: After scaling, num_epochs {new_total_epochs} is preserved"
                )
                adjusted_scaling_factor = 1
            else:
                self._logger.debug(
                    f"Job {job_id}: After scaling, new_total_epochs={new_total_epochs}, old_total_epochs={old_total_epochs}, manually rescaling"
                )
                new_total_steps = (
                    math.ceil(dataset_size_dict[model] / new_bs)
                    * old_total_epochs
                )
                adjusted_scaling_factor = old_total_epochs / new_total_epochs
                self._logger.debug(
                    f"Job {job_id}: Adjusted scaling factor is {adjusted_scaling_factor}"
                )

            self._logger.debug(
                f"Job {job_id}: scaling total_steps from {total_steps} to {new_total_steps}, preserving num_epochs = {old_total_epochs}"
            )
            self._jobs[job_id].total_steps = new_total_steps

            # also preserve the finished progress during rescaling
            num_completed_epochs = math.ceil(
                total_steps_run / math.ceil(dataset_size_dict[model] / old_bs)
            )
            new_total_steps_run = num_completed_epochs * math.ceil(
                dataset_size_dict[model] / new_bs
            )
            progress_scaling_factor = new_total_steps_run / math.ceil(
                total_steps_run * iterations_scaling_factor
            )

            self._logger.debug(
                f"Job {job_id}: scaling total_steps_run by {iterations_scaling_factor}x from {total_steps_run} to {new_total_steps_run} (adjustment: {round(progress_scaling_factor, 5)}x)"
            )
            # total_steps_run = math.ceil(total_steps_run * iterations_scaling_factor)
            # self._total_steps_run[job_id] = total_steps_run
            self._total_steps_run[job_id] = new_total_steps_run

            self._logger.debug(
                f"Job {job_id}: scaling steps_run_so_far by {iterations_scaling_factor}x from {steps_run_so_far} to {new_total_steps_run} (adjustment: {round(progress_scaling_factor, 5)}x)"
            )
            # steps_run_so_far = math.ceil(steps_run_so_far * iterations_scaling_factor)
            # self._steps_run_so_far[job_id]["v100"] = steps_run_so_far
            self._steps_run_so_far[job_id]["v100"] = new_total_steps_run

            # reset the flags
            self._bs_flags[job_id]["big_bs"] = False
            self._bs_flags[job_id]["small_bs"] = False
