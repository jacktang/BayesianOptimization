import numpy as np
import jax.numpy as jnp
from jax import random, jit, vmap
from functools import partial
import numpyro
import gc
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular
import warnings
from bayes_opt.gp_base import GPBase

def chunk_vmap(fun, array, chunk_size=4):
    """Process large arrays in smaller chunks to reduce memory usage."""
    L = array[0].shape[0]
    chunks = get_chunks(L, chunk_size)
    results = [vmap(fun)(*tuple([a[chunk] for a in array])) for chunk in chunks]
    num_results = len(results[0])
    return tuple([jnp.concatenate([r[k] for r in results]) for k in range(num_results)])

def get_chunks(L, chunk_size):
    """Split length L into chunks of size chunk_size."""
    num_chunks = L // chunk_size
    chunks = [jnp.arange(i * chunk_size, (i + 1) * chunk_size) for i in range(num_chunks)]
    if L % chunk_size != 0:
        chunks.append(np.arange(L - L % chunk_size, L))
    return chunks

class SAASGP(GPBase):
    def __init__(
        self,
        alpha=0.1,
        num_warmup=512,
        num_samples=256,
        max_tree_depth=7,
        num_chains=1,
        thinning=16,
        kernel="matern",
        random_state=None,
        verbose=True,
        observation_variance=0.0,
    ):
        """Initialize SAASGP with memory-efficient settings."""
        self.alpha = alpha
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.max_tree_depth = max_tree_depth
        self.num_chains = num_chains
        self.thinning = thinning
        self.kernel = kernel
        self.verbose = verbose
        self.random_state = random_state if random_state is not None else 0
        self.observation_variance = observation_variance
        self.learn_noise = observation_variance == 0.0
        
        # Initialize state
        self.fitted_ = False
        self.X_train_ = None
        self.y_train_ = None
        self.y_mean_ = None
        self.y_std_ = None
        self.noise_ = 1e-6
        self.flat_samples = None
        self.Ls = None
        
        # Initialize JAX random key
        self.rng_key = random.PRNGKey(self.random_state)

    def model(self, X, y=None):
        """Define the SAASGP model with SAAS prior."""
        N, P = X.shape
        
        # Sample kernel hyperparameters
        var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 10.0))
        noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0)) if self.learn_noise else self.observation_variance
        tausq = numpyro.sample("kernel_tausq", dist.HalfCauchy(self.alpha))
        
        # Sample and reparameterize length scales
        inv_length_sq = numpyro.sample("_kernel_inv_length_sq", dist.HalfCauchy(jnp.ones(P)))
        inv_length_sq = numpyro.deterministic("kernel_inv_length_sq", tausq * inv_length_sq)
        
        # Compute kernel matrix
        K = self._compute_kernel_matrix(X, X, var, inv_length_sq)
        K = K + jnp.eye(N) * (noise + 1e-6)
        
        # Sample observations
        numpyro.sample("y", dist.MultivariateNormal(loc=jnp.zeros(N), covariance_matrix=K), obs=y)

    def _compute_kernel_matrix(self, X1, X2, var, inv_length_sq):
        """Compute kernel matrix using current parameters."""
        if self.kernel == "matern":
            deltaXsq = jnp.square(X1[:, None, :] - X2) * inv_length_sq
            dsq = jnp.sum(deltaXsq, axis=-1)
            r = jnp.sqrt(jnp.clip(dsq, a_min=1e-12))
            sqrt5 = jnp.sqrt(5.0)
            return var * (1.0 + sqrt5*r + 5.0/3.0*dsq) * jnp.exp(-sqrt5*r)
        else:  # RBF kernel
            deltaXsq = jnp.square(X1[:, None, :] - X2) * inv_length_sq
            return var * jnp.exp(-0.5 * jnp.sum(deltaXsq, axis=-1))

    def fit(self, X, y):
        """Fit SAASGP to training data with efficient memory handling."""
        try:
            # Clear memory
            gc.collect()
            
            self.X_train_ = np.array(X)
            self.y_train_ = np.array(y)
            
            # Normalize training data
            self.y_mean_ = np.mean(self.y_train_)
            self.y_std_ = np.std(self.y_train_) + 1e-8
            y_normalized = (self.y_train_ - self.y_mean_) / self.y_std_
            
            # Clear any previous MCMC samples
            if hasattr(self, 'flat_samples'):
                del self.flat_samples
                gc.collect()

            # Run MCMC
            kernel = NUTS(self.model, max_tree_depth=self.max_tree_depth)
            mcmc = MCMC(
                kernel,
                num_warmup=self.num_warmup,
                num_samples=self.num_samples,
                num_chains=self.num_chains,
                progress_bar=self.verbose,
                thinning=self.thinning,
            )
            
            self.rng_key, subkey = random.split(self.rng_key)
            mcmc.run(subkey, self.X_train_, y_normalized)
            
            self.flat_samples = mcmc.get_samples()
            self.fitted_ = True
            
            # Force garbage collection
            gc.collect()
            
            return self
            
        except Exception as e:
            if self.verbose:
                print(f"Error during SAASGP fitting: {e}")
            self.fitted_ = False
            if hasattr(self, 'flat_samples'):
                del self.flat_samples
            gc.collect()
            return self

    def predict(self, X, return_std=False):
        """Make predictions with chunked computation."""
        if not self.fitted_:
            raise ValueError("Model has not been fitted yet.")
            
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        try:
            def _predict_chunk(X_chunk):
                # Use median of hyperparameters for prediction
                var = np.median(self.flat_samples['kernel_var'])
                inv_length_sq = np.median(self.flat_samples['kernel_inv_length_sq'], axis=0)
                noise = np.median(self.flat_samples['kernel_noise']) if self.learn_noise else self.observation_variance
                
                # Compute kernel matrices for chunk
                K = self._compute_kernel_matrix(X_chunk, X_chunk, var, inv_length_sq)
                K_train = self._compute_kernel_matrix(self.X_train_, self.X_train_, var, inv_length_sq)
                K_cross = self._compute_kernel_matrix(X_chunk, self.X_train_, var, inv_length_sq)
                
                # Add noise to diagonal
                K_train = K_train + jnp.eye(len(self.X_train_)) * (noise + self.noise_)
                
                # Compute predictive mean and variance
                L = cho_factor(K_train)[0]
                alpha = cho_solve((L, True), self.y_train_)
                mean = jnp.dot(K_cross, alpha)
                
                if return_std:
                    v = solve_triangular(L, K_cross.T, lower=True)
                    var = jnp.diag(K - jnp.dot(v.T, v))
                    std = jnp.sqrt(jnp.maximum(var, 1e-6))
                    return mean, std
                return mean
            
            # Process in chunks
            chunk_size = 100  # Adjust based on memory constraints
            n_chunks = (len(X) + chunk_size - 1) // chunk_size
            
            results = []
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(X))
                chunk_result = _predict_chunk(X[start_idx:end_idx])
                results.append(chunk_result)
            
            # Combine results
            if return_std:
                means, stds = zip(*results)
                mean = np.concatenate(means)
                std = np.concatenate(stds)
                return mean * self.y_std_ + self.y_mean_, std * self.y_std_
            else:
                mean = np.concatenate(results)
                return mean * self.y_std_ + self.y_mean_
                
        except Exception as e:
            if self.verbose:
                print(f"Error during prediction: {e}")
            if return_std:
                return np.zeros(len(X)), np.ones(len(X))
            return np.zeros(len(X))

    def sample_y(self, X, n_samples=1):
        """Draw samples from GP posterior with chunked computation."""
        if not self.fitted_:
            raise ValueError("Model has not been fitted yet.")
            
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        try:
            mean, std = self.predict(X, return_std=True)
            
            # Use JAX's random number generator for consistency
            rng_key = random.PRNGKey(self.random_state)
            samples = mean[:, None] + std[:, None] * random.normal(
                rng_key,
                shape=(len(X), n_samples)
            )
            
            return np.array(samples)
            
        except Exception as e:
            if self.verbose:
                print(f"Error during sampling: {e}")
            return np.zeros((len(X), n_samples))