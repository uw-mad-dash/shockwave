import math
import cvxpy
import mosek
import copy
import numpy as np


def unpack_sched_alloc_solution(
    jobs_epochs_sched, jobs_steps_mig_alloc, jobs_steps_mem_alloc, future_steps
):

    assert (
        len(jobs_epochs_sched)
        == len(jobs_steps_mig_alloc)
        == len(jobs_steps_mem_alloc)
    )

    njobs = len(jobs_epochs_sched)

    jobs_sched_alloc_steps = []
    for istep in range(future_steps):
        jobs_sched_alloc = []
        for ijob in range(njobs):
            job_sched = int(sum(jobs_epochs_sched[ijob][:, istep]) > 0)
            job_mig_step = jobs_steps_mig_alloc[ijob, istep] * job_sched
            job_mem_step = jobs_steps_mem_alloc[ijob, istep] * job_sched
            jobs_sched_alloc.append((job_sched, job_mig_step, job_mem_step))
        jobs_sched_alloc_steps.append(jobs_sched_alloc)
    return jobs_sched_alloc_steps


def shockwave_allocator(
    ngpus,
    gpu_mig,
    gpu_mem,
    input_jobs,
    future_steps,
    step_duration,
    solver_reltol=1e-4,
    solver_maxtime=60,
):

    print(
        f"shockwave_allocator got a list of jobs with length {len(input_jobs)}"
    )

    (
        jobs_epochs_sched,
        jobs_steps_mig_alloc,
        jobs_steps_mem_alloc,
    ) = social_welfare_optimization(
        ngpus=ngpus,
        gpu_mig=gpu_mig,
        gpu_mem=gpu_mem,
        job_list=input_jobs,
        future_steps=future_steps,
        step_duration=step_duration,
        solver_reltol=solver_reltol,
        solver_maxtime=solver_maxtime,
    )

    jobs_sched_alloc_steps = unpack_sched_alloc_solution(
        jobs_epochs_sched=jobs_epochs_sched,
        jobs_steps_mig_alloc=jobs_steps_mig_alloc,
        jobs_steps_mem_alloc=jobs_steps_mem_alloc,
        future_steps=future_steps,
    )

    return jobs_sched_alloc_steps


def construct_epoch_sched_vars(
    job_list, future_steps, step_duration, binary=True
):
    print(f"construct_epoch_sched_vars got a list of length {len(job_list)}")
    jobs_epochs_sched_vars = []
    for job in job_list:
        job_nepochs_unprogressed = job.epochs - job.epoch_progress
        print(
            f"Job {job.jobid}, job_nepochs_unprogressed {job_nepochs_unprogressed} = job.epochs-job.epoch_progress ({job.epochs} - {job.epoch_progress})"
        )
        duration_modes = job.epoch_duration_modes()
        min_epoch_duration = max(1, min(duration_modes))
        max_epoch_duration = max(1, max(duration_modes))
        assert max_epoch_duration < step_duration
        nepochs = int(
            math.ceil((future_steps * step_duration) / min_epoch_duration)
        )
        print(
            f"Job {job.jobid}, nepochs is min(job_nepochs_unprogressed, nepochs) = min({job_nepochs_unprogressed}, {nepochs})"
        )
        nepochs = min(job_nepochs_unprogressed, nepochs)
        # The last fake step to dump unscheduled epochs
        jobs_epochs_sched_vars.append(
            cvxpy.Variable(shape=(nepochs, future_steps + 1), boolean=binary)
        )
        # print("Job: {}, Future Epochs: {}, Future Rounds: {}".format (
        #    job.jobid, nepochs, future_steps))
    return jobs_epochs_sched_vars


def construct_steps_alloc_vars(job_list, future_steps):
    njobs = len(job_list)
    jobs_epochs_mig_vars = cvxpy.Variable(shape=(njobs, future_steps))
    jobs_epochs_mem_vars = cvxpy.Variable(shape=(njobs, future_steps))
    return jobs_epochs_mig_vars, jobs_epochs_mem_vars


def disclose_future_epochs_info(job_list, epoch_sched_vars, future_steps):
    assert len(job_list) == len(epoch_sched_vars)
    njobs = len(job_list)
    disclosed_epochs_duration = []
    disclosed_epochs_mig_reqs = []
    disclosed_epochs_mem_reqs = []
    for idx_job in range(njobs):
        job = job_list[idx_job]
        sched_vars = epoch_sched_vars[idx_job]
        nepochs, nsteps = sched_vars.shape
        assert future_steps + 1 == nsteps
        current_epochs = job.epoch_progress
        future_epochs_duration = copy.deepcopy(
            job.epoch_duration[current_epochs : current_epochs + nepochs]
        )
        disclosed_epochs_duration.append(future_epochs_duration)

        future_epochs_mig_reqs = [
            job.mig_req(epoch)
            for epoch in range(current_epochs, current_epochs + nepochs)
        ]
        disclosed_epochs_mig_reqs.append(future_epochs_mig_reqs)

        future_epochs_mem_reqs = [
            job.mem_req(epoch)
            for epoch in range(current_epochs, current_epochs + nepochs)
        ]
        disclosed_epochs_mem_reqs.append(future_epochs_mem_reqs)

    return (
        disclosed_epochs_duration,
        disclosed_epochs_mig_reqs,
        disclosed_epochs_mem_reqs,
    )


def construct_sched_consts(epochs_sched_vars):
    consts = []
    for sched_vars in epochs_sched_vars:
        nepochs, nsteps = sched_vars.shape
        expr_epochs_sched_timestep = []
        for iepoch in range(nepochs):
            consts.append(sched_vars[iepoch, :] >= 0)
            consts.append(cvxpy.sum(sched_vars[iepoch, :]) == 1)
            epoch_sched_timestep = cvxpy.sum(
                cvxpy.multiply(np.arange(1, nsteps + 1), sched_vars[iepoch, :])
            )
            expr_epochs_sched_timestep.append(epoch_sched_timestep)
            if iepoch > 0:
                consts.append(
                    expr_epochs_sched_timestep[iepoch]
                    >= expr_epochs_sched_timestep[iepoch - 1]
                )
    return consts


def construct_alloc_consts(
    ngpus,
    gpu_mig,
    gpu_mem,
    job_list,
    jobs_epochs_sched_vars,
    jobs_steps_mig_vars,
    jobs_steps_mem_vars,
    jobs_epochs_duration,
    jobs_epochs_mig_reqs,
    jobs_epochs_mem_reqs,
    future_steps,
    step_duration,
):
    consts = []
    njobs = len(job_list)
    mig_lim = gpu_mig
    mem_lim = gpu_mem

    for jobidx in range(njobs):
        job_scheds = jobs_epochs_sched_vars[jobidx]
        job_durations = jobs_epochs_duration[jobidx]
        job_mig_reqs = jobs_epochs_mig_reqs[jobidx]
        job_mem_reqs = jobs_epochs_mem_reqs[jobidx]
        job_mig_alloc = jobs_steps_mig_vars[jobidx, :]
        job_mem_alloc = jobs_steps_mem_vars[jobidx, :]
        nepochs, nsteps = job_scheds.shape
        assert future_steps + 1 == nsteps
        for istep in range(future_steps):

            sched_step = job_scheds[:, istep]
            sched_nepochs_step = cvxpy.sum(sched_step)
            z = cvxpy.Variable(boolean=True)
            sched_nepochs_step <= z * nepochs
            z <= nepochs * sched_nepochs_step

            mig_step_alloc = job_mig_alloc[istep]
            consts.append(mig_step_alloc <= z * mig_lim)

            mem_step_alloc = job_mem_alloc[istep]
            consts.append(mem_step_alloc <= z * mem_lim)

            expr_mig_step_utility = cvxpy.min(
                cvxpy.hstack(
                    [
                        mig_step_alloc / float(mig_req)
                        for mig_req in job_mig_reqs
                    ]
                )
            )
            expr_mem_step_utility = cvxpy.min(
                cvxpy.hstack(
                    [
                        mem_step_alloc / float(mem_req)
                        for mem_req in job_mem_reqs
                    ]
                )
            )
            expr_step_utility = cvxpy.minimum(
                expr_mig_step_utility, expr_mem_step_utility, 1.0
            )

            consts.append(
                cvxpy.sum(
                    cvxpy.multiply(
                        job_scheds[:, istep], np.array(job_durations)
                    )
                )
                <= step_duration * expr_step_utility
            )

    for istep in range(future_steps):
        jobs_mig_step_alloc = jobs_steps_mig_vars[:, istep]
        jobs_mem_step_alloc = jobs_steps_mem_vars[:, istep]
        job_nwokers = np.array(
            [job_list[jobidx].nworkers for jobidx in range(njobs)]
        )
        expr_mig_alloc_cluster = cvxpy.sum(
            cvxpy.multiply(jobs_mig_step_alloc, job_nwokers)
        )
        expr_mem_alloc_cluster = cvxpy.sum(
            cvxpy.multiply(jobs_mem_step_alloc, job_nwokers)
        )
        consts.append(expr_mig_alloc_cluster <= mig_lim * ngpus)
        consts.append(expr_mem_alloc_cluster <= mem_lim * ngpus)

    return consts


def compute_alloc_jobs_progresses(
    job_list, jobs_epochs_sched_vars, future_steps
):
    jobs_progresses = []
    njobs = len(job_list)
    for jobidx in range(njobs):
        job = job_list[jobidx]
        job_cur_epochs = job.epoch_progress
        job_overall_epochs = job.epochs
        job_epochs_sched = jobs_epochs_sched_vars[jobidx]
        future_nepochs, nsteps = job_epochs_sched.shape
        assert future_steps + 1 == nsteps
        job_future_epochs = cvxpy.sum(job_epochs_sched[:, 0:future_steps])
        job_alloc_progress = job_cur_epochs + job_future_epochs
        jobs_progresses.append(job_alloc_progress / float(job_overall_epochs))

    return jobs_progresses


def compute_jobs_social_welfare(
    ngpus,
    gpu_mig,
    gpu_mem,
    job_list,
    job_progresses,
    jobs_steps_mig_vars,
    jobs_steps_mem_vars,
    future_steps,
    job_weights=None,
    stability=1e-6,
):
    njobs = len(job_progresses)
    mig_lim = gpu_mig
    mem_lim = gpu_mem

    if job_weights == None:
        job_weights = np.array([1.0] * njobs)

    expr_sw = cvxpy.sum(
        cvxpy.multiply(
            job_weights,
            cvxpy.hstack(
                [
                    cvxpy.log(stability + expr_progress)
                    for expr_progress in job_progresses
                ]
            ),
        )
    )

    expr_cluster_usage_steps = []
    for istep in range(future_steps):
        jobs_mig_step_alloc = jobs_steps_mig_vars[:, istep]
        jobs_mem_step_alloc = jobs_steps_mem_vars[:, istep]
        job_nwokers = np.array(
            [job_list[jobidx].nworkers for jobidx in range(njobs)]
        )
        expr_mig_cluster_usage = cvxpy.sum(
            cvxpy.multiply(jobs_mig_step_alloc, job_nwokers)
        ) / (mig_lim * ngpus)
        expr_mem_cluster_usage = cvxpy.sum(
            cvxpy.multiply(jobs_mem_step_alloc, job_nwokers)
        ) / (mem_lim * ngpus)
        expr_cluster_usage_steps.append(
            cvxpy.minimum(expr_mig_cluster_usage, expr_mem_cluster_usage)
        )

    opt = expr_sw + 1e-2 * cvxpy.log(
        cvxpy.sum(expr_cluster_usage_steps) / future_steps
    )

    return opt


def social_welfare_optimization(
    ngpus,
    gpu_mig,
    gpu_mem,
    job_list,
    future_steps,
    step_duration,
    solver_reltol,
    solver_maxtime,
):

    print(
        f"social_welfare_optimization got a job list of length {len(job_list)}"
    )

    jobs_epochs_sched_vars = construct_epoch_sched_vars(
        job_list=job_list,
        future_steps=future_steps,
        step_duration=step_duration,
    )

    jobs_steps_mig_vars, jobs_steps_mem_vars = construct_steps_alloc_vars(
        job_list=job_list, future_steps=future_steps
    )

    jobs_sched_consts = construct_sched_consts(
        epochs_sched_vars=jobs_epochs_sched_vars
    )

    (
        jobs_epochs_duration,
        jobs_epochs_mig_reqs,
        jobs_epochs_mem_reqs,
    ) = disclose_future_epochs_info(
        job_list=job_list,
        epoch_sched_vars=jobs_epochs_sched_vars,
        future_steps=future_steps,
    )

    jobs_alloc_consts = construct_alloc_consts(
        ngpus=ngpus,
        gpu_mig=gpu_mig,
        gpu_mem=gpu_mem,
        job_list=job_list,
        jobs_epochs_sched_vars=jobs_epochs_sched_vars,
        jobs_steps_mig_vars=jobs_steps_mig_vars,
        jobs_steps_mem_vars=jobs_steps_mem_vars,
        jobs_epochs_duration=jobs_epochs_duration,
        jobs_epochs_mig_reqs=jobs_epochs_mig_reqs,
        jobs_epochs_mem_reqs=jobs_epochs_mem_reqs,
        future_steps=future_steps,
        step_duration=step_duration,
    )

    jobs_alloc_progresses = compute_alloc_jobs_progresses(
        job_list=job_list,
        jobs_epochs_sched_vars=jobs_epochs_sched_vars,
        future_steps=future_steps,
    )

    jobs_welfare = compute_jobs_social_welfare(
        ngpus=ngpus,
        gpu_mig=gpu_mig,
        gpu_mem=gpu_mem,
        job_list=job_list,
        jobs_steps_mig_vars=jobs_steps_mig_vars,
        jobs_steps_mem_vars=jobs_steps_mem_vars,
        future_steps=future_steps,
        job_progresses=jobs_alloc_progresses,
    )

    objective = cvxpy.Maximize(jobs_welfare)
    constraints = jobs_sched_consts + jobs_alloc_consts
    problem = cvxpy.Problem(objective=objective, constraints=constraints)
    mosek_options = {
        mosek.dparam.mio_tol_rel_gap: solver_reltol,
        mosek.dparam.optimizer_max_time: solver_maxtime,
    }
    result = problem.solve(
        solver=cvxpy.MOSEK, verbose=True, mosek_params=mosek_options
    )
    assert result is not None
    return (
        [job.value for job in jobs_epochs_sched_vars],
        jobs_steps_mig_vars.value,
        jobs_steps_mem_vars.value,
    )
