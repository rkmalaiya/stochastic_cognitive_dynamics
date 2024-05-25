import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from dataclasses import dataclass
from jax.scipy.stats import norm
from jax import  random
from numpyro.distributions.distribution import Distribution
from numpyro.infer import MCMC, NUTS, Predictive

@dataclass
class args:
    seed = 12345678
    num_data = 1000
    intercept = 1.5
    slope = -1.2
    noise_scale = 0.5
    num_samples = 1000
    num_warmup = 2000
    num_chains = 4
    device = "cpu"

numpyro.set_platform(args.device)
numpyro.set_host_device_count(args.num_chains)

def gen_data(rng, num_data: int, 
             intercept:float, slope:float, noise_scale:float):
    X = np.linspace(0, 1, num_data)
    # y = a + b*x 
    regression_line = intercept + slope * X
    # add noise
    y = regression_line + rng.normal(scale=0.25, size=num_data)
    return X, y


def logp(X, y, a, b, sigma):
    """The likelihood function for a linear model
    y ~ ax+b+error
    """
    y_hat = a * X + b  # BUT WE DON'T KNOW IT in my case and function returns L for me.
    L = jnp.sum(jnp.log(norm.pdf(y - y_hat, loc = 0, scale=sigma)))
    return L

def model(X=None, y=None):
    a = numpyro.sample("a", dist.Normal())
    b = numpyro.sample("b", dist.Normal())
    sigma = numpyro.sample("sigma", dist.HalfNormal())
    if X is not None:
        log_density = logp(X=X, y=y, a=a, b=b, sigma=sigma)
        numpyro.factor("custom_logp", log_density)

# RNG for numpy
rng_numpy = np.random.default_rng(args.seed)
# RNG for jax
rng_trace, rng_prior, rng_post = random.split(random.PRNGKey(args.seed), 3)
# Generate data

X, y = gen_data(rng_numpy, num_data=args.num_data,
               intercept=args.intercept, slope=args.slope,
               noise_scale=args.noise_scale)
designed_label = f"y = {args.slope}x + {args.intercept} + N(0, {args.noise_scale})"

# Run inference
mcmc = MCMC(
    NUTS(model),
    num_warmup=args.num_warmup,
    num_samples=args.num_samples,
    num_chains=args.num_chains,
    progress_bar=True
)
mcmc.run(rng_trace, X=X, y=y)
posterior_samples = mcmc.get_samples()
mcmc.print_summary()
print(f"designed: {designed_label}")

## MY QUESTION IS HERE
x_test = 0.1
print("\n predictive mcmc")
# Run predictive
mcmc_pred = MCMC(
    NUTS(potential_fn=lambda y : -logp(X=x_test, y=y,
                                       a = posterior_samples["a"], 
                                       b = posterior_samples["b"],
                                       sigma = posterior_samples["sigma"])),
    num_warmup=args.num_warmup,
    num_samples=args.num_samples,
    num_chains=args.num_chains,
    progress_bar=True
)
mcmc_pred.run(rng_trace, init_params=jnp.repeat(0., args.num_chains))
mcmc_pred.print_summary()
print(f"This should be around y = {args.slope*x_test+args.intercept} for x_test={x_test}")