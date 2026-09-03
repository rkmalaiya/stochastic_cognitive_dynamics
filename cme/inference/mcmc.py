"""
Posterior sampling by MCMC (NUTS).
"""

import os
import time

import arviz as az
import jax
import numpy as np
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median

from cme.inference.numpyro_model import model
from cme.utils import common_logging as cl
from cme.utils import common_utils as cu

log = cl.get_logger("inference.mcmc")


def sample_posterior_params(DT, X, n_states, start_width, response_width, delta, measurement_prob,
                            num_warmup=100, samples_n=500, num_chains=4, batch_size=2, max_tree_depth=10,
                            params_type = "Centralized|NonCentralized", model_type="Markov|Quantum", transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT"):

    #kernel = HMCECS(NUTS(model), num_blocks=10)
    #mcmc_chain = MCMC(kernel, num_warmup=num_warmup, num_samples=samples_n, num_chains=num_chains)
    #mcmc_chain.run(cu.get_rng(), n_states, start_width,  sigma, tau, DT, X, I, J, s_0, batch_size=batch_size, extra_fields=('hmc_state',))

    # Adaptive mass matrix and increased target_accept_prob for better convergence with non-centered params
    # Previous NUTS configuration retained for reference:
    # kernel = NUTS(model, forward_mode_differentiation=False, adapt_mass_matrix=True, adapt_step_size = True,
    #               dense_mass=True, init_strategy=init_to_median(num_samples=20),
    #               target_accept_prob=0.8 if model_type=="Quantum" else 0.9)
    # mcmc_chain = MCMC(kernel, num_warmup=num_warmup, num_samples=samples_n, num_chains=num_chains,
    #                   chain_method="vectorized" if jax.default_backend() == "gpu" else "parallel",
    #                   progress_bar=True, jit_model_args=False)

    kernel = NUTS(model, forward_mode_differentiation=False, adapt_mass_matrix=True, adapt_step_size = True,
                  dense_mass=True, init_strategy=init_to_median(num_samples=20),
                  target_accept_prob=0.8 if model_type=="Quantum" else 0.9,
                  max_tree_depth=max_tree_depth)
    # "vectorized" vmaps the chains, so NUTS's tree-building while_loop runs until the
    # SLOWEST chain finishes - every chain pays the maximum tree depth of the group,
    # every iteration. With mean ~473 steps and max 1023 that alone is ~2x waste, on top
    # of losing the 4x from running four chains concurrently. Measured on Juno as
    # ~20 s/it vectorized vs ~1.2 s/it parallel.
    # Original CPU/GPU chain selection retained for reference:
    # chain_method = "sequential" if num_chains == 1 else ("vectorized" if jax.default_backend() == "gpu" else "parallel")
    # Vectorized-only version retained for reference:
    # chain_method = "vectorized"
    chain_method = "sequential" if num_chains == 1 else "parallel"
    # Progress bar kept on everywhere - it logs usefully into the .err file under SLURM.
    # TTY-only version retained for reference:
    # show_progress = sys.stdout.isatty()
    # mcmc_chain = MCMC(kernel, num_warmup=num_warmup, num_samples=samples_n, num_chains=num_chains,
    #                   chain_method=chain_method, progress_bar=show_progress, jit_model_args=False)
    mcmc_chain = MCMC(kernel, num_warmup=num_warmup, num_samples=samples_n, num_chains=num_chains,
                      chain_method=chain_method, progress_bar=True, jit_model_args=False)
    start_run = time.perf_counter()
    mcmc_chain.run(cu.get_rng(), n_states, start_width, response_width, delta, X, DT, measurement_prob,
                   params_type = params_type, transition_type=transition_type,
                   likelihood_type=likelihood_type, model_type=model_type,
                   # extra_fields=('potential_energy',)
                   extra_fields=('potential_energy', 'num_steps', 'accept_prob', 'diverging', 'adapt_state.step_size'))
    run_secs = time.perf_counter() - start_run

    mcmc_diagnostics = mcmc_chain.get_extra_fields()
    num_steps = np.asarray(mcmc_diagnostics["num_steps"]) # num_chains*samples_n
    divergences = np.asarray(mcmc_diagnostics["diverging"]) # num_chains*samples_n
    log.info(f"NUTS diagnostics - chains: {num_chains}, mean steps: {num_steps.mean():.2f}, max steps: {num_steps.max()}, divergences: {divergences.sum()}")

    # One line to compare a Juno run against a laptop run. ms_per_it and us_per_grad are
    # the numbers that matter: laptop reference is ~21 ms/it at n_states=51, I=1, J=30,
    # 4 chains, chain_method=parallel. cores_allowed is what SLURM actually granted -
    # if it is far below cpu_count then JAX/Eigen are oversubscribing the allocation.
    iters = num_warmup + samples_n
    cores_allowed = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    log.info(
        "PERF: model=%s method=%s chains=%s iters=%s wall=%.1fs %.1f ms/it "
        "%.0f us/grad steps_mean=%.0f steps_max=%s div=%s | devices=%s "
        "cores_allowed=%s cpu_count=%s OMP=%s XLA_FLAGS=%s",
        model_type, chain_method, num_chains, iters, run_secs,
        run_secs / iters * 1e3,
        run_secs / iters / max(num_steps.mean(), 1) * 1e6,
        num_steps.mean(), num_steps.max(), divergences.sum(),
        jax.local_device_count(), cores_allowed, os.cpu_count(),
        os.environ.get("OMP_NUM_THREADS", "unset"),
        os.environ.get("XLA_FLAGS", "unset"),
    )

    return mcmc_chain#, post_likl


def get_arviz_model(mcmc_chain):
    return az.from_numpyro(mcmc_chain)
