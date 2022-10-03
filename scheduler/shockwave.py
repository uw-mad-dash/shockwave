import logging
import math
import random
import sys
from collections import OrderedDict
from cvxpy.constraints.nonpos import NonNeg
from cvxpy.reductions.solvers.defines import SOLVER_MAP_CONIC

import numpy as np
import cvxpy
import gurobipy
import mosek
import copy
import weakref

from shockwave_helper import MinMaxSumKSubarrays
from JobMetaData import JobMetaData


class ShockwaveScheduler(object):
    def __init__(
        self,
        ngpus: int,
        gram: int,
        init_metadata: OrderedDict,
        future_nrounds: int,
        round_duration: int,
        solver_preference: list,
        solver_rel_gap: float,
        solver_num_threads: int,
        solver_timeout: float,
        n_epoch_vars_max: int,
        logapx_bases: list,
        logapx_origin: dict,
        k: float,
        lam: float,
        rhomax: float,
    ):

        self.ngpus = ngpus
        self.gram = gram
        assert self.ngpus > 0
        assert self.gram > 0

        self.future_nrounds = future_nrounds
        assert self.future_nrounds > 0

        self.round_duration = round_duration
        assert self.round_duration > 0

        self.solver_preference = solver_preference
        self.solver_rel_gap = solver_rel_gap
        self.solver_num_threads = solver_num_threads
        self.solver_timeout = solver_timeout
        assert self.solver_timeout > 0

        self.n_epoch_vars_max = n_epoch_vars_max
        assert type(self.n_epoch_vars_max) == int and self.n_epoch_vars_max > 0

        self.logapx_bases = logapx_bases
        assert type(self.logapx_bases) == list

        self.logapx_origin = logapx_origin
        assert type(self.logapx_origin) == dict

        self.k = k
        assert self.k > 0

        assert type(init_metadata) == OrderedDict
        self.metadata = OrderedDict()
        for jobid, jobobj in init_metadata.items():
            self.add_metadata(jobid, jobobj)

        self.schedules = OrderedDict()

        self.round_ptr = 0

        self.resolve = True

        self.completed_jobs = OrderedDict()

        self.reestimate_share = True
        self.share_series = {}

        self.lam = lam
        self.rhomax = rhomax

    def finish_time_uniform_share(self):
        ngpus = self.ngpus
        njobs = len(self.metadata)

        if self.reestimate_share:

            for jobid in self.metadata.keys():
                job = self.metadata[jobid]

                uniform_share = min(1.0, ngpus / njobs)

                assert uniform_share > 0.0

                job.calibrate_profiled_epoch_duration()

                finish_time_estimate = (
                    job.timestamp_submit
                    + (
                        sum(job.epoch_duration[: job.epoch_progress])
                        + job.dirichlet_posterior_remaining_runtime(
                            job.epoch_progress
                        )
                    )
                    / uniform_share
                )

                if jobid not in self.share_series.keys():
                    self.share_series[jobid] = []
                self.share_series[jobid].append(
                    (self.round_ptr, finish_time_estimate)
                )

        self.reestimate_share = False

    def round_schedule(self):

        if not self.resolve:
            if len(self.schedules) > 0:
                if self.round_ptr in self.schedules.keys():
                    return self.schedules[self.round_ptr]

        jobids = list(self.metadata.keys())
        jobobjs = list(self.metadata.values())

        self.finish_time_uniform_share()
        share_series = [self.share_series[jobid] for jobid in jobids]

        solution = dynamic_eisenberg_gale_scheduling(
            self.ngpus,
            jobobjs,
            self.round_ptr,
            self.future_nrounds,
            self.round_duration,
            share_series,
            self.solver_preference,
            self.solver_rel_gap,
            self.solver_num_threads,
            self.solver_timeout,
            self.n_epoch_vars_max,
            self.logapx_bases,
            self.logapx_origin,
            self.k,
            self.lam,
            self.rhomax,
        )

        schedules = construct_schedules(
            solution,
            jobids,
            jobobjs,
            self.round_ptr,
            self.future_nrounds,
            self.ngpus,
        )

        self.schedules = schedules
        self.clear_resolve()

        return self.schedules[self.round_ptr]

    def increment_round_ptr(self):
        self.round_ptr += 1

    def set_resolve(self):
        self.resolve = True

    def clear_resolve(self):
        self.resolve = False

    def schedule_progress(self, jobid, epoch_progress, share_update=True):
        assert jobid in self.metadata.keys()
        job = self.metadata[jobid]
        job.set_epoch_progress(epoch_progress)
        job.reset_waiting_delay()
        # if(job.epoch_progress >= job.epochs):
        #     self.remove_metadata(jobid)
        #     assert(jobid not in self.completed_jobs.keys())
        #     self.completed_jobs[jobid] = job
        #     if(share_update):
        #         self.reestimate_share = True

    def deschedule_waiting_delay(self, jobid, delay):
        if jobid in self.metadata.keys():
            job = self.metadata[jobid]
            job.add_waiting_delay(delay)
        else:
            return

    def add_metadata(self, jobid, jobobj: JobMetaData, share_update=True):
        assert jobid not in self.metadata.keys()
        self.metadata[jobid] = jobobj
        self.set_resolve()
        if share_update:
            self.reestimate_share = True

    def remove_metadata(self, jobid, share_update=True):
        assert jobid not in self.completed_jobs.keys()
        self.completed_jobs[jobid] = self.metadata[jobid]
        if share_update:
            self.reestimate_share = True
        assert jobid in self.metadata.keys()
        self.metadata.pop(jobid)
        self.set_resolve()


def construct_schedules(
    schedule_solution: list,
    jobids: list,
    jobobjs: list,
    round_ptr: int,
    future_rounds: int,
    ngpus: int,
):

    log = logging.getLogger("scheduler")

    njobs = len(jobids)
    assert njobs == len(jobobjs)
    assert njobs == len(schedule_solution)

    nrounds = len(schedule_solution[0])
    assert nrounds == future_rounds

    rounds_sched = OrderedDict()

    for iround in range(nrounds):
        jobids_cur_round = []
        jobobjs_cur_round = []
        for ijob in range(njobs):
            round_index = round_ptr + iround
            job = jobobjs[ijob]
            jobid = jobids[ijob]
            assert jobid == job.jobid
            sched = schedule_solution[ijob][iround].value[()]
            if round(sched) == 1.0:
                jobids_cur_round.append(jobid)
                jobobjs_cur_round.append(job)
        if len(jobids_cur_round) <= 0:
            log.warning(
                "Invalid solution: none jobs are scheduled in round {}".format(
                    round_index
                )
            )

        nworkers = sum([job.nworkers for job in jobobjs_cur_round])
        n_idle_workers = ngpus - nworkers
        if n_idle_workers > 0:
            non_sched_indice = [
                idx
                for idx in range(njobs)
                if jobids[idx] not in jobids_cur_round
            ]

            non_sched_indice_sorted = sorted(
                non_sched_indice,
                key=lambda idx: jobobjs[
                    idx
                ].dirichlet_posterior_remaining_runtime(),
                reverse=True,
            )

            for ijob in non_sched_indice_sorted:
                jobid = jobids[ijob]
                job = jobobjs[ijob]
                if job.nworkers <= n_idle_workers:
                    n_idle_workers -= job.nworkers
                    jobids_cur_round.append(jobid)
                    log.info(
                        "Work conserving scheduling for job {}".format(
                            job.jobname
                        )
                    )
                if n_idle_workers <= 0:
                    break

        rounds_sched[round_index] = jobids_cur_round

    return rounds_sched


def construct_round_sched_vars(njobs: int, future_nrounds: int):
    jobs_round_sched_vars = []
    for _ in range(njobs):
        jobs_round_sched_vars.append(
            [cvxpy.Variable(boolean=True) for _ in range(future_nrounds)]
        )
    return jobs_round_sched_vars


def construct_round_sched_constraints(
    jobs_metadata: list,
    jobs_round_sched_vars: list,
    ngpus: int,
    future_nrounds: int,
):
    consts = []
    njobs = len(jobs_round_sched_vars)
    assert njobs == len(jobs_metadata)

    for iround in range(future_nrounds):
        round_scheds = []
        for ijob in range(njobs):
            job = jobs_metadata[ijob]
            nworkers = job.nworkers
            round_sched_vars = jobs_round_sched_vars[ijob]
            assert len(round_sched_vars) == future_nrounds
            round_scheds.append(
                cvxpy.multiply(nworkers, round_sched_vars[iround])
            )
        consts.append(cvxpy.sum(cvxpy.hstack(round_scheds)) <= ngpus)

    return consts


def interpolate_epoch_duration(job):
    job.calibrate_profiled_epoch_duration()
    return np.mean(job.epoch_duration[: job.epoch_progress + 1])


def nash_social_welfare_first_order_apx(
    jobs_metadata: list,
    jobs_round_sched_vars: list,
    round_duration: int,
    logapx_bases,
    logapx_origin,
):
    njobs = len(jobs_metadata)
    assert njobs == len(jobs_round_sched_vars)
    job_log_progresses = []
    jobs_remained_epochs = []

    assert logapx_bases[0] == 0.0
    logapx_base_values = []
    for base in logapx_bases:
        assert base >= 0.0 and base <= 1.0
        if base == 0.0:
            assert 0.0 in logapx_origin.keys()
            logapx_base_values.append(math.log(logapx_origin[0.0]))
        else:
            logapx_base_values.append(math.log(base))
    assert len(logapx_bases) == len(logapx_base_values)
    assert all(
        prev < next
        for prev, next in zip(logapx_base_values, logapx_base_values[1:])
    )

    planned_runtime_list = []
    planned_progress_consts = []
    log_apx_consts = []

    min_progress_rate = 1.0
    for ijob in range(njobs):
        job = jobs_metadata[ijob]

        job_nepochs = job.epochs
        assert job_nepochs > 0
        if 1.0 / job_nepochs < min_progress_rate:
            min_progress_rate = 1.0 / job_nepochs

        cur_progress = job.epoch_progress

        planned_progress = cvxpy.Variable(nonneg=True)
        epoch_duration_interpolated = interpolate_epoch_duration(job)
        planned_runtime = planned_progress * epoch_duration_interpolated
        planned_runtime_list.append(planned_runtime)
        planned_progress_consts += [
            planned_runtime
            <= cvxpy.sum(cvxpy.hstack(jobs_round_sched_vars[ijob]))
            * round_duration
        ]

        objective_progress = cur_progress + planned_progress
        objective_progress_normalized = cvxpy.multiply(
            objective_progress, 1.0 / float(job_nepochs)
        )

        vars_cursor = [
            cvxpy.Variable(nonneg=True) for _ in range(len(logapx_bases))
        ]
        var_log_progress_normalized = cvxpy.sum(
            cvxpy.multiply(
                cvxpy.hstack((vars_cursor)), np.array(logapx_base_values)
            )
        )

        cursor_consts = []
        cursor_consts += [
            cvxpy.sum(
                cvxpy.multiply(
                    cvxpy.hstack(vars_cursor), np.array(logapx_bases)
                )
            )
            == objective_progress_normalized
        ]
        cursor_consts += [cvxpy.sum(cvxpy.hstack(vars_cursor)) == 1.0]
        vars_boundary = [
            cvxpy.Variable(boolean=True) for _ in range(len(logapx_bases))
        ]

        boundary_consts = []
        boundary_consts += [cvxpy.sum(cvxpy.hstack(vars_boundary)) <= 2]

        for varcursor, varboundary in zip(vars_cursor, vars_boundary):
            boundary_consts += [varcursor <= varboundary]

        if len(vars_boundary) > 2:
            for lboundary in range(0, len(vars_boundary) - 2):
                for rboundary in range(lboundary + 2, len(vars_boundary)):
                    boundary_consts += [
                        vars_boundary[lboundary] + vars_boundary[rboundary]
                        <= 1.0
                    ]

        log_apx_consts += cursor_consts
        log_apx_consts += boundary_consts
        job_log_progresses.append(var_log_progress_normalized)

        remained_epochs = job_nepochs - cur_progress - planned_progress
        jobs_remained_epochs.append(remained_epochs)

    return (
        job_log_progresses,
        planned_runtime_list,
        log_apx_consts,
        planned_progress_consts,
    )


def call_cvxpy_solver(
    solver,
    objective,
    constraints,
    solver_rel_gap,
    solver_num_threads,
    solver_timeout,
    seed=0,
):

    log = logging.getLogger("scheduler")

    result = None
    problem = cvxpy.Problem(objective=objective, constraints=constraints)

    random.seed(seed)
    np.random.seed(seed)

    if SOLVER_MAP_CONIC == cvxpy.MOSEK:
        mosek_options = {
            mosek.dparam.mio_tol_rel_gap: solver_rel_gap,
            mosek.iparam.num_threads: solver_num_threads,
            mosek.dparam.optimizer_max_time: solver_timeout,
        }
        result = problem.solve(
            solver=cvxpy.MOSEK, verbose=True, mosek_params=mosek_options
        )
    elif solver == cvxpy.GUROBI:
        time_limit = (
            solver_timeout if solver_timeout > 0 else gurobipy.GRB.INFINITY
        )
        result = problem.solve(
            solver=cvxpy.GUROBI,
            verbose=True,
            MIPGap=solver_rel_gap,
            Threads=solver_num_threads,
            TimeLimit=time_limit,
        )
    else:
        log.error("CVXPY solver is not supported.")

    return problem, result


def finish_time_momentumed_average(series, round, momentum=0.9):
    assert len(series) > 0
    irounds = [ir for ir, _ in series]
    assert max(irounds) <= round
    irounds += [round]
    ftwindows = np.diff(irounds)
    if max(ftwindows) == 0:
        ftprobs = [1.0]
    else:
        ftprobs = ftwindows / np.sum(ftwindows)
        ftprobs = ftprobs.tolist()
    ftvals = [val for _, val in series]
    assert len(ftprobs) == len(ftvals)
    running_average = 0.0
    for prob, val in zip(ftprobs, ftvals):
        running_average += prob * val

    running_average = (
        momentum * running_average + (1.0 - momentum) * ftvals[-1]
    )

    return running_average


def dynamic_eisenberg_gale_scheduling(
    ngpus: int,
    jobobjs: list,
    round_index: int,
    future_nrounds: int,
    round_duration: int,
    share_series: list,
    solver_preference: list,
    solver_rel_gap: float,
    solver_num_threads: int,
    solver_timeout: float,
    n_epoch_vars_max: int,
    logapx_bases: list,
    logapx_origin: dict,
    k: float,
    lam: float,
    rhomax: float,
):

    for solver in solver_preference:
        cvxpy_solver = getattr(cvxpy, solver, None)
        if (cvxpy_solver is not None) and (
            solver in cvxpy.installed_solvers()
        ):
            break
        cvxpy_solver = None

    assert cvxpy_solver is not None

    log = logging.getLogger("scheduler")

    njobs = len(jobobjs)
    constraints = []
    jobs_round_sched_vars = construct_round_sched_vars(njobs, future_nrounds)
    constraints += construct_round_sched_constraints(
        jobobjs, jobs_round_sched_vars, ngpus, future_nrounds
    )

    (
        jobs_log_utilities,
        jobs_planned_runtime,
        swconsts,
        progress_consts,
    ) = nash_social_welfare_first_order_apx(
        jobobjs,
        jobs_round_sched_vars,
        round_duration,
        logapx_bases,
        logapx_origin,
    )

    jobs_remaining_time_sched = []
    for ijob in range(len(jobs_planned_runtime)):
        job = jobobjs[ijob]
        remaining_runtime_shed = cvxpy.maximum(
            0,
            job.dirichlet_posterior_remaining_runtime()
            - jobs_planned_runtime[ijob],
        )
        jobs_remaining_time_sched.append(remaining_runtime_shed)

    objective = cvxpy.Maximize(
        cvxpy.sum(cvxpy.hstack(jobs_log_utilities) / (njobs * future_nrounds))
        - k * cvxpy.max(cvxpy.hstack(jobs_remaining_time_sched))
    )

    constraints += swconsts
    constraints += progress_consts

    finish_time_consts = []
    next_sched_time = round_duration * (round_index + future_nrounds)

    finish_time_sched_vars = []
    finish_time_objectives = []
    for ijob in range(len(share_series)):
        job = jobobjs[ijob]

        future_share = min(1.0, ngpus / njobs)
        # future_share = min(1.0, ngpus/(njobs*job.nworkers))

        remaining_runtime_shed = jobs_remaining_time_sched[ijob]
        finish_time_sched = next_sched_time + cvxpy.multiply(
            remaining_runtime_shed, 1.0 / future_share
        )
        finish_time_objective = finish_time_momentumed_average(
            share_series[ijob], round_index
        )

        finish_time_consts.append(
            finish_time_sched <= finish_time_objective * rhomax
        )

        finish_time_sched_vars.append(finish_time_sched)
        finish_time_objectives.append(finish_time_objective)

    ENABLE_FTF_CONSTS = True
    # ENABLE_FTF_CONSTS=False

    if ENABLE_FTF_CONSTS:

        log.info(f"Call CVXPY solver in Round {round_index}...")

        problem, result = call_cvxpy_solver(
            cvxpy_solver,
            objective,
            constraints + finish_time_consts,
            solver_rel_gap,
            solver_num_threads,
            solver_timeout,
        )

        log.info(f"CVXPY solution status:{problem.status}")
        log.info(f"CVXPY objective value:{result}")

    if ENABLE_FTF_CONSTS and problem.status in cvxpy.settings.SOLUTION_PRESENT:
        solver_solution_info(
            ngpus,
            jobobjs,
            jobs_round_sched_vars,
            round_index,
            future_nrounds,
            round_duration,
            finish_time_sched_vars,
            finish_time_objectives,
            share_series,
        )
    else:
        if (
            not ENABLE_FTF_CONSTS
        ) or problem.status in cvxpy.settings.INF_OR_UNB:
            log.warning(f"Nullify finish time constraints...")

            (
                jobs_log_utilities_upgrade,
                jobs_priorities,
            ) = relax_finish_time_constraints(
                ngpus,
                jobobjs,
                jobs_log_utilities,
                round_index,
                future_nrounds,
                round_duration,
                share_series,
                rhomax,
                lam,
            )

            objective = cvxpy.Maximize(
                cvxpy.sum(
                    cvxpy.hstack(jobs_log_utilities_upgrade)
                    / (njobs * future_nrounds)
                )
                - k * cvxpy.max(cvxpy.hstack(jobs_remaining_time_sched))
            )

            log.info(
                f"Re-call CVXPY solver in Round {round_index} with finish time constraints relaxed..."
            )

            problem, result = call_cvxpy_solver(
                cvxpy_solver,
                objective,
                constraints,
                solver_rel_gap,
                solver_num_threads,
                solver_timeout,
            )
            assert problem.status in cvxpy.settings.SOLUTION_PRESENT
            solver_solution_info(
                ngpus,
                jobobjs,
                jobs_round_sched_vars,
                round_index,
                future_nrounds,
                round_duration,
                finish_time_sched_vars,
                finish_time_objectives,
                share_series,
            )

            log.info(f"Optimize job ranks in schedule...")
            jobs_round_sched_vars = rank_in_schedule_jobs(
                jobs_round_sched_vars,
                jobs_priorities,
                jobobjs,
                ngpus,
                cvxpy_solver,
                solver_rel_gap,
                solver_num_threads,
                solver_timeout,
            )
            log.info(f"Adjusted job ranks in schedule:")
            solver_solution_info(
                ngpus,
                jobobjs,
                jobs_round_sched_vars,
                round_index,
                future_nrounds,
                round_duration,
                finish_time_sched_vars,
                finish_time_objectives,
                share_series,
            )

        else:
            log.error(f"Solver internal error.")

    return jobs_round_sched_vars


def rank_in_schedule_jobs(
    jobs_round_sched_vars: list,
    jobs_priorities: list,
    jobobjs: list,
    ngpus: int,
    cvxpy_solver,
    solver_rel_gap,
    solver_num_threads,
    solver_timeout,
):

    njobs = len(jobs_round_sched_vars)
    nrounds = len(jobs_round_sched_vars[0])
    jobs_nworkers = [job.nworkers for job in jobobjs]

    jobs_round_sched_ranked = []
    for _ in range(njobs):
        jobs_round_sched_ranked.append(
            [cvxpy.Variable(boolean=True) for _ in range(nrounds)]
        )

    jobs_sched_nrounds = []
    for ijob in range(njobs):
        jobs_sched_nrounds.append(
            cvxpy.sum(cvxpy.hstack(jobs_round_sched_vars[ijob])).value
        )

    consts = []
    for ijob in range(njobs):
        consts += [
            jobs_sched_nrounds[ijob]
            == cvxpy.sum(cvxpy.hstack(jobs_round_sched_ranked[ijob]))
        ]

    for iround in range(nrounds):
        scheds = [
            jobs_round_sched_ranked[ijob][iround] for ijob in range(njobs)
        ]
        consts += [
            cvxpy.sum(
                cvxpy.multiply(
                    cvxpy.hstack(scheds), cvxpy.hstack(jobs_nworkers)
                )
            )
            <= ngpus
        ]

    obj_components = []
    for ijob in range(njobs):
        priority = jobs_priorities[ijob]
        if jobs_sched_nrounds[ijob] > 0:
            avg_shed_idx = (
                cvxpy.sum(
                    cvxpy.multiply(
                        cvxpy.hstack([t for t in range(nrounds)]),
                        cvxpy.hstack(jobs_round_sched_ranked[ijob]),
                    )
                )
                / jobs_sched_nrounds[ijob]
            )
            obj_components.append(avg_shed_idx * priority)

    if len(obj_components) <= 0:
        return jobs_round_sched_vars

    obj = cvxpy.Minimize(cvxpy.sum(cvxpy.hstack(obj_components)))

    problem, result = call_cvxpy_solver(
        cvxpy_solver,
        obj,
        consts,
        solver_rel_gap,
        solver_num_threads,
        solver_timeout,
    )

    if problem.status not in cvxpy.settings.SOLUTION_PRESENT:
        return jobs_round_sched_vars

    return jobs_round_sched_ranked


def solver_solution_info(
    ngpus,
    jobobjs,
    jobs_round_sched_vars,
    round_index,
    future_nrounds,
    round_duration,
    finish_time_sched_vars,
    finish_time_objectives,
    share_series,
):

    return  # NOTE: reduce output log complexity of shockwave

    log = logging.getLogger("scheduler")

    for ijob in range(len(jobobjs)):
        job = jobobjs[ijob]
        shares = share_series[ijob]
        round_sched = [var.value for var in jobs_round_sched_vars[ijob]]
        sched_ftf = float(finish_time_sched_vars[ijob].value)
        runavg_ftf = float(finish_time_objectives[ijob])

        log.info(f"Job {job.jobid}")
        log.info(f"Current Round:{round_index}")

        log.info(f"Round schedule:{round_sched}")

        log.info(f"Finish Time (VAR.VALUE): {sched_ftf}")
        log.info(f"Finish Time Objective (RunAvg): {runavg_ftf}")
        log.info(f"Ratio: {sched_ftf/runavg_ftf}")
        log.info(f"-" * 40)


def relax_finish_time_constraints(
    ngpus,
    jobobjs,
    jobutils,
    round_index,
    future_nrounds,
    round_duration,
    share_series,
    rhomax,
    lam,
):
    log = logging.getLogger("scheduler")

    priority_threshold = rhomax
    priority_power = lam
    priority_M = 1e2

    njobs = len(jobobjs)
    assert ngpus > 0
    assert njobs > 0

    jobs_rho_ratios = []
    jobs_remaining_runtime_projected = []

    for ijob in range(len(jobobjs)):
        job = jobobjs[ijob]
        shares = share_series[ijob]

        round_time = round_duration * round_index

        future_share = min(1.0, ngpus / njobs)
        # future_share = min(1.0, ngpus/(njobs*job.nworkers))

        job.calibrate_profiled_epoch_duration()
        remaining_runtime_projected = (
            job.dirichlet_posterior_remaining_runtime()
        )
        jobs_remaining_runtime_projected.append(remaining_runtime_projected)
        finish_time_projected = (
            round_time + remaining_runtime_projected / future_share
        )
        finish_time_runavg = finish_time_momentumed_average(
            shares, round_index
        )
        rho_ratio_projected = finish_time_projected / finish_time_runavg

        # log.info(
        #     f"Job {job.jobid}"
        # )
        # log.info(
        #     f"Finish Time (Pre-solving Projection): {finish_time_projected}"
        # )
        # log.info(
        #     f"Finish Time Objective (RunAvg): {finish_time_runavg}")
        # log.info(
        #     f"Ratio: {rho_ratio_projected}"
        # )
        # log.info(f"-"*40)

        jobs_rho_ratios.append(rho_ratio_projected)

    jobs_log_utilities_upgrade = []
    jobs_priorities = []

    for ijob, ratio in enumerate(jobs_rho_ratios):
        log_utility = jobutils[ijob]
        job = jobobjs[ijob]

        remaining_runtime_projected = jobs_remaining_runtime_projected[ijob]
        if ratio > priority_threshold:
            priority = ratio ** priority_power
            if remaining_runtime_projected < round_duration:
                priority = ratio ** priority_M
            utility_upgrade = log_utility * priority
            # log.info(f"Relax finish time constraints for Job {job.jobid}, Priority: {priority}")
        else:
            priority = 1.0
            utility_upgrade = log_utility
        jobs_log_utilities_upgrade.append(utility_upgrade)
        jobs_priorities.append(priority)

    return jobs_log_utilities_upgrade, jobs_priorities
