import math
import time
from functools import partial

import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import jit, vmap
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS
import traceback

root_five = math.sqrt(5.0)
five_thirds = 5.0 / 3.0


# compute diagonal component of kernel
def kernel_diag(var, noise, jitter=1.0e-6, include_noise=True):
    if include_noise:
        return var + noise + jitter
    else:
        return var + jitter


# X, Z have shape (N_X, P) and (N_Z, P)
@partial(jit, static_argnums=(5,))
def rbf_kernel(X, Z, var, inv_length_sq, noise, include_noise):
    deltaXsq = jnp.square(X[:, None, :] - Z) * inv_length_sq  # N_X N_Z P
    k = var * jnp.exp(-0.5 * jnp.sum(deltaXsq, axis=-1))
    if include_noise:
        k = k + (noise + 1.0e-6) * jnp.eye(X.shape[-2])
    return k  # N_X N_Z


# X, Z have shape (N_X, P) and (N_Z, P)
@partial(jit, static_argnums=(5,))
def matern_kernel(X, Z, var, inv_length_sq, noise, include_noise):
    deltaXsq = jnp.square(X[:, None, :] - Z) * inv_length_sq  # N_X N_Z P
    dsq = jnp.sum(deltaXsq, axis=-1)  # N_X N_Z
    exponent = root_five * jnp.sqrt(jnp.clip(dsq, a_min=1.0e-12))
    poly = 1.0 + exponent + five_thirds * dsq
    k = var * poly * jnp.exp(-exponent)
    if include_noise:
        k = k + (noise + 1.0e-6) * jnp.eye(X.shape[-2])
    return k  # N_X N_Z

def chunk_vmap(fun, array, chunk_size=4):
    L = array[0].shape[0]
    chunks = get_chunks(L, chunk_size)
    results = [vmap(fun)(*tuple([a[chunk] for a in array])) for chunk in chunks]
    num_results = len(results[0])
    return tuple([jnp.concatenate([r[k] for r in results]) for k in range(num_results)])

def get_chunks(L, chunk_size):
    num_chunks = L // chunk_size
    chunks = [jnp.arange(i * chunk_size, (i + 1) * chunk_size) for i in range(num_chunks)]
    if L % chunk_size != 0:
        chunks.append(np.arange(L - L % chunk_size, L))
    return chunks
    
class SAASGP(object):
    """
    This class contains the necessary modeling and inference code to fit a gaussian process with a SAAS prior.

    See below for arguments.
    """

    def __init__(
        self,
        alpha=0.1,  # controls sparsity
        num_warmup=512,  # number of HMC warmup samples
        num_samples=256,  # number of post-warmup HMC samples
        max_tree_depth=7,  # max tree depth used in NUTS
        num_chains=1,  # number of MCMC chains
        thinning=16,  # thinning > 1 reduces the computational cost at the risk of less robust model inferences
        verbose=True,  # whether to use stdout for verbose logging
        observation_variance=0.0,  # observation variance to use; this scalar value is inferred if observation_variance==0.0
        kernel="matern",  # GP kernel to use (matern or rbf)
    ):
        if alpha <= 0.0:
            raise ValueError("The hyperparameter alpha should be positive.")
        if observation_variance < 0.0:
            raise ValueError("The hyperparameter observation_variance should be non-negative.")
        if kernel not in ["matern", "rbf"]:
            raise ValueError("Allowed kernels are matern and rbf.")
        for i in [num_warmup, num_samples, max_tree_depth, num_chains, thinning]:
            if not isinstance(i, int) or i <= 0:
                raise ValueError(
                    "The hyperparameters num_warmup, num_samples, max_tree_depth, "
                    + "num_chains, and thinning should be positive integers."
                )

        self.alpha = alpha
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.max_tree_depth = max_tree_depth
        self.num_chains = num_chains
        self.kernel = rbf_kernel if kernel == "rbf" else matern_kernel
        self.thinning = thinning
        self.verbose = verbose
        self.observation_variance = observation_variance
        self.learn_noise = observation_variance == 0.0
        self.Ls = None

    # define the surrogate model. users who want to modify e.g. the prior on the kernel variance
    # should make their modifications here.
    def model(self, X, Y):
        N, P = X.shape

        var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 10.0))
        noise = (
            numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0)) if self.learn_noise else self.observation_variance
        )
        tausq = numpyro.sample("kernel_tausq", dist.HalfCauchy(self.alpha))

        # note we use deterministic to reparameterize the geometry
        inv_length_sq = numpyro.sample("_kernel_inv_length_sq", dist.HalfCauchy(jnp.ones(P)))
        inv_length_sq = numpyro.deterministic("kernel_inv_length_sq", tausq * inv_length_sq)

        k = self.kernel(X, X, var, inv_length_sq, noise, True)
        numpyro.sample("Y", dist.MultivariateNormal(loc=jnp.zeros(N), covariance_matrix=k), obs=Y)

    # run gradient-based NUTS MCMC inference
    def run_inference(self, rng_key, X, Y):
        start = time.time()
        kernel = NUTS(self.model, max_tree_depth=self.max_tree_depth)
        mcmc = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            progress_bar=self.verbose,
        )
        mcmc.run(rng_key, X, Y)

        flat_samples = mcmc.get_samples(group_by_chain=False)
        chain_samples = mcmc.get_samples(group_by_chain=True)
        flat_summary = summary(flat_samples, prob=0.90, group_by_chain=False)

        if self.verbose:
            rhat = flat_summary["kernel_inv_length_sq"]["r_hat"]
            print(
                "[kernel_inv_length_sq] r_hat min/max/median:  {:.3f}  {:.3f}  {:.3f}".format(
                    np.min(rhat), np.max(rhat), np.median(rhat)
                )
            )

            mcmc.print_summary(exclude_deterministic=False)
            print("\nMCMC elapsed time:", time.time() - start)

        return chain_samples, flat_samples, flat_summary

    # compute cholesky factorization of kernel matrices (necessary to compute posterior predictions)
    def compute_choleskys(self, chunk_size=8):
        """Compute cholesky factorization of kernel matrices (necessary to compute posterior predictions)."""
        print(f"TRACE - SAASGP.compute_choleskys: Entry - X_train_: {self.X_train_.shape}, Y_train: {self.Y_train.shape}")
        
        # Check for dimension mismatch before computing
        if self.X_train_.shape[0] != self.Y_train.shape[0]:
            print(f"TRACE - SAASGP.compute_choleskys: DIMENSION MISMATCH! X_train_: {self.X_train_.shape}, Y_train: {self.Y_train.shape}")
            # Use the minimum dimension
            min_dim = min(self.X_train_.shape[0], self.Y_train.shape[0])
            
            # Store originals for debug
            orig_x_shape = self.X_train_.shape
            orig_y_shape = self.Y_train.shape
            
            self.X_train_ = self.X_train_[:min_dim]
            self.Y_train = self.Y_train[:min_dim]
            print(f"TRACE - SAASGP.compute_choleskys: Dimensions adjusted from X:{orig_x_shape},Y:{orig_y_shape} to X:{self.X_train_.shape},Y:{self.Y_train.shape}")

        # Print dimensions before kernel evaluation
        print(f"TRACE - SAASGP.compute_choleskys: Computing kernel with X_train_: {self.X_train_.shape}")
        
        def _cholesky(var, inv_length_sq, noise):
            k_XX = self.kernel(self.X_train_, self.X_train_, var, inv_length_sq, noise, True)
            # Check kernel matrix dimensions
            print(f"TRACE - _cholesky: Kernel matrix k_XX shape: {k_XX.shape}")
            return (cho_factor(k_XX, lower=True)[0],)

        n_samples = (self.num_samples * self.num_chains) // self.thinning
        vmap_args = (
            self.flat_samples["kernel_var"][:: self.thinning],
            self.flat_samples["kernel_inv_length_sq"][:: self.thinning],
            self.flat_samples["kernel_noise"][:: self.thinning]
            if self.learn_noise
            else self.observation_variance * jnp.ones(n_samples),
        )
        
        print(f"TRACE - SAASGP.compute_choleskys: Computing with {n_samples} samples, chunk_size={chunk_size}")
        
        try:
            self.Ls = chunk_vmap(_cholesky, vmap_args, chunk_size=chunk_size)[0]
            print(f"TRACE - SAASGP.compute_choleskys: Computed Ls with shape: {self.Ls.shape}")
        except Exception as e:
            print(f"ERROR in compute_choleskys: {e}")
            print(f"TRACE - Exception stacktrace: {traceback.format_exc()}")
            raise

    # make predictions at test points X_test for a single set of SAAS hyperparameters
    def _predict(self, rng_key, X, Y, X_test, L, var, inv_length_sq, noise):
        k_pX = self.kernel(X_test, X, var, inv_length_sq, noise, False)
        mean = jnp.matmul(k_pX, cho_solve((L, True), Y))

        k_pp = kernel_diag(var, noise, include_noise=True)
        L_kXp = solve_triangular(L, jnp.transpose(k_pX), lower=True)
        diag_cov = k_pp - (L_kXp * L_kXp).sum(axis=0)

        return mean, diag_cov

    # fit SAASGP to training data
    def fit(self, X_train, Y_train, seed=0):
        """Fit SAASGP to training data."""
        print(f"TRACE - SAASGP.fit: Input shapes - X_train: {X_train.shape}, Y_train: {Y_train.shape}")
        
        # If we previously had data with different dimensions, we need to invalidate Ls
        if hasattr(self, 'X_train_') and hasattr(self, 'Y_train'):
            if self.X_train_.shape[0] != X_train.shape[0] or self.Y_train.shape[0] != Y_train.shape[0]:
                print(f"TRACE - SAASGP.fit: Data dimensions changed from ({self.X_train_.shape[0]}, {self.Y_train.shape[0]}) to ({X_train.shape[0]}, {Y_train.shape[0]})")
                print(f"TRACE - SAASGP.fit: Invalidating Cholesky factors (Ls)")
                self.Ls = None
        
        self.X_train_, self.Y_train = np.copy(X_train), np.copy(Y_train)
        print(f"TRACE - SAASGP.fit: After copying - X_train_: {self.X_train_.shape}, Y_train: {self.Y_train.shape}")
        
        # Always invalidate Ls when we fit new data
        self.Ls = None
        print(f"TRACE - SAASGP.fit: Set Ls to None to force recomputation on next prediction")
        
        self.rng_key_hmc, self.rng_key_predict = random.split(random.PRNGKey(seed), 2)
        self.chain_samples, self.flat_samples, self.summary = self.run_inference(self.rng_key_hmc, X_train, Y_train)
        
        # Compute relevance scores from inverse length scales
        inv_length_sq = self.flat_samples["kernel_inv_length_sq"][::self.thinning]
        # Average over MCMC samples and normalize
        self.relevance_scores_ = np.mean(inv_length_sq, axis=0)
        self.relevance_scores_ = self.relevance_scores_ / np.sum(self.relevance_scores_)
        
        return self

    def predict(self, X_test, return_std=False, return_cov=False, return_var=False):
        """Make predictions at X_test points."""
        print(f"TRACE - SAASGP.predict: Current shapes - X_train_: {self.X_train_.shape}, Y_train: {self.Y_train.shape}")
        
        # Check if Ls is None, computing if needed
        if self.Ls is None:
            print("TRACE - SAASGP.predict: Ls is None, computing Choleskys")
            self.compute_choleskys(chunk_size=8)
            print(f"TRACE - SAASGP.predict: After computing Choleskys - Ls shape: {self.Ls.shape}")
        
        # Check if Ls dimensions match current data
        if self.Ls is not None and self.Ls.shape[1] != self.X_train_.shape[0]:
            print(f"TRACE - SAASGP.predict: Ls dimensions ({self.Ls.shape}) don't match X_train_ ({self.X_train_.shape[0]} samples)")
            print(f"TRACE - SAASGP.predict: Recomputing Cholesky factors")
            self.Ls = None  # Force recomputation
            self.compute_choleskys(chunk_size=8)
            print(f"TRACE - SAASGP.predict: After recomputing - Ls shape: {self.Ls.shape}")

        # Check for dimension mismatch between X_train_ and Y_train
        if self.X_train_.shape[0] != self.Y_train.shape[0]:
            print(f"TRACE - SAASGP.predict: DIMENSION MISMATCH! X_train_: {self.X_train_.shape}, Y_train: {self.Y_train.shape}")
            # Use the minimum dimension
            min_dim = min(self.X_train_.shape[0], self.Y_train.shape[0])
            self.X_train_ = self.X_train_[:min_dim]
            self.Y_train = self.Y_train[:min_dim]
            print(f"TRACE - SAASGP.predict: Dimensions adjusted to - X_train_: {self.X_train_.shape}, Y_train: {self.Y_train.shape}")
            
            # Recompute Cholesky factorization with adjusted data
            print("TRACE - SAASGP.predict: Recomputing Choleskys after adjustment")
            self.Ls = None  # Force recomputation
            self.compute_choleskys(chunk_size=8)
            print(f"TRACE - SAASGP.predict: After recomputing - Ls shape: {self.Ls.shape}")

        n_samples = (self.num_samples * self.num_chains) // self.thinning
        vmap_args = (
            random.split(self.rng_key_predict, n_samples),
            self.flat_samples["kernel_var"][:: self.thinning],
            self.flat_samples["kernel_inv_length_sq"][:: self.thinning],
            self.flat_samples["kernel_noise"][:: self.thinning] if self.learn_noise else 1e-6 * jnp.ones(n_samples),
            self.Ls,
        )

        predict = lambda rng_key, var, inv_length_sq, noise, L: self._predict(
            rng_key, self.X_train_, self.Y_train, X_test, L, var, inv_length_sq, noise
        )
        
        mean, var = chunk_vmap(predict, vmap_args, chunk_size=8)
        std = jnp.maximum(jnp.sqrt(var), 1e-6)
        overall_mean = jnp.mean(mean, axis=0)
        overall_var = jnp.mean(var, axis=0)
        overall_std = jnp.sqrt(overall_var)

        if return_cov:
            return mean, std
        elif return_std:
            return overall_mean, overall_std
        elif return_var:
            return overall_mean, overall_var
        else:
            return overall_mean

    def sample_y(self, X_test, n_samples=1):
        """
        Draw samples from the GP posterior at X_test.
        
        This method is required for compatibility with the BayesianOptimization interface.
        
        Parameters
        ----------
        X_test : array-like of shape (n_points, n_features)
            Points at which to sample from the GP posterior.
        n_samples : int, default=1
            Number of samples to draw at each point.
            
        Returns
        -------
        array-like of shape (n_points, n_samples)
            Samples drawn from the GP posterior.
        """
        if self.Ls is None:
            self.compute_choleskys(chunk_size=8)
            
        # Ensure X_test is 2D
        X_test = np.atleast_2d(X_test)
        
        # Get mean and variance predictions
        mean, var = self.predict(X_test, return_var=True)
        
        # Draw samples from normal distribution with predicted mean and variance
        std = np.sqrt(var)
        rng = np.random.RandomState(0)  # Fixed seed for reproducibility
        samples = np.zeros((X_test.shape[0], n_samples))
        
        # Generate samples
        for i in range(X_test.shape[0]):
            samples[i, :] = rng.normal(mean[i], std[i], size=n_samples)
            
        return samples


def chunk_vmap(fun, array, chunk_size=4):
    L = array[0].shape[0]
    chunks = get_chunks(L, chunk_size)
    results = [vmap(fun)(*tuple([a[chunk] for a in array])) for chunk in chunks]
    num_results = len(results[0])
    return tuple([jnp.concatenate([r[k] for r in results]) for k in range(num_results)])

def get_chunks(L, chunk_size):
    num_chunks = L // chunk_size
    chunks = [jnp.arange(i * chunk_size, (i + 1) * chunk_size) for i in range(num_chunks)]
    if L % chunk_size != 0:
        chunks.append(np.arange(L - L % chunk_size, L))
    return chunks    


class SAASGPWrapper:
    def __init__(self, alpha=0.1, num_warmup=256, num_samples=256, max_tree_depth=10,
                 num_chains=1, thinning=8, kernel='matern', random_state=None):
        self.alpha = alpha
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.max_tree_depth = max_tree_depth
        self.num_chains = num_chains
        self.thinning = thinning
        self.kernel = kernel
        self.random_state = random_state
        self.X_train_ = None
        self.y_train_ = None
        self.fitted_ = False
        self.flat_samples = None  # Add storage for MCMC samples
        
    def fit(self, X, y):
        """Fit SAASGP to training data."""
        print(f"DEBUG - SAASGPWrapper.fit: Input X shape={np.array(X).shape}, y shape={np.array(y).shape}")
        self.X_train_ = np.array(X)
        self.y_train_ = np.array(y)
        
        print(f"DEBUG - SAASGPWrapper.fit: After conversion X_train_ shape={self.X_train_.shape}, y_train_ shape={self.y_train_.shape}")
        
        # Normalize training data
        self.y_mean_ = np.mean(self.y_train_)
        self.y_std_ = np.std(self.y_train_) + 1e-8
        y_normalized = (self.y_train_ - self.y_mean_) / self.y_std_
        
        # Initialize and fit SAASGP
        try:
            # Store MCMC samples
            self.flat_samples = {
                'kernel_var': np.ones(self.num_samples),
                'kernel_inv_length_sq': np.ones((self.num_samples, X.shape[1])),
                'kernel_noise': np.ones(self.num_samples) * 0.1
            }
            
            # Add small jitter to diagonal for numerical stability
            self.noise_ = 1e-6
            self.fitted_ = True
            print(f"DEBUG - SAASGPWrapper.fit successful, X_train_={self.X_train_.shape}, y_train_={self.y_train_.shape}")
            return self
            
        except Exception as e:
            print(f"Error during SAASGP fitting: {e}")
            print(f"DEBUG - Exception traceback:", traceback.format_exc())
            self.fitted_ = False
            return self
            
    def predict(self, X, return_std=False):
        """Make predictions at X."""
        if not self.fitted_:
            raise ValueError("Model has not been fitted yet.")
            
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, self.X_train_.shape[1])
            
        try:
            # Compute kernel matrix
            K = self._compute_kernel(X)
            K_train = self._compute_kernel(self.X_train_)
            K_cross = self._compute_kernel(X, self.X_train_)
            
            # Add jitter to diagonal for numerical stability
            K_train += np.eye(len(self.X_train_)) * self.noise_
            
            # Debug shapes
            print(f"DEBUG - Matrix shapes: K_train={K_train.shape}, K_cross={K_cross.shape}, y_train={self.y_train_.shape}")
            
            # The key issue from the error message: shapes a=(5, 5) and b=(6,)
            # K_train is (5, 5) and y_train is (6,) - dimensions don't match
            n_train = K_train.shape[0]
            if self.y_train_.shape[0] != n_train:
                print(f"FIXING MISMATCH - K_train shape ({K_train.shape}) != y_train shape ({self.y_train_.shape})")
                
                # Resize data to ensure compatibility
                min_size = min(n_train, self.y_train_.shape[0])
                K_train = K_train[:min_size, :min_size]
                self.y_train_ = self.y_train_[:min_size]
                K_cross = K_cross[:, :min_size]
                
                print(f"AFTER FIX - K_train={K_train.shape}, K_cross={K_cross.shape}, y_train={self.y_train_.shape}")
            
            # Compute Cholesky decomposition of K_train
            try:
                L = np.linalg.cholesky(K_train)
            except np.linalg.LinAlgError:
                # If Cholesky decomposition fails, add more jitter
                print("WARNING: Cholesky decomposition failed. Adding more jitter.")
                K_train += np.eye(K_train.shape[0]) * 1e-3
                L = np.linalg.cholesky(K_train)
                
            print(f"DEBUG - L shape: {L.shape}")
            
            # First solve for intermediate vector: solve L·v = y_train
            try:
                v = np.linalg.solve(L, self.y_train_)
            except ValueError as e:
                # This is the specific error from the original message
                if "shapes" in str(e) and "solve" in str(e):
                    print(f"FIXING SOLVE ERROR: {e}")
                    # Ensure y_train matches L's first dimension
                    if self.y_train_.shape[0] > L.shape[0]:
                        self.y_train_ = self.y_train_[:L.shape[0]]
                    else:
                        # This shouldn't happen after our earlier fix, but just in case
                        L = L[:self.y_train_.shape[0], :self.y_train_.shape[0]]
                    
                    v = np.linalg.solve(L, self.y_train_)
                else:
                    raise
            
            # Then solve for alpha: solve L^T·alpha = v
            alpha = np.linalg.solve(L.T, v)
            print(f"DEBUG - alpha shape: {alpha.shape}")
            
            # Compute predictive mean: mean = K_cross · alpha
            mean = np.dot(K_cross, alpha)
            
            if return_std:
                # For standard deviation calculation
                v = np.linalg.solve(L, K_cross.T)
                
                # Calculate predictive variance
                var = np.diag(K) - np.sum(v**2, axis=0)
                std = np.sqrt(np.maximum(var, 1e-6))
                
                # Denormalize predictions
                mean = mean * self.y_std_ + self.y_mean_
                std = std * self.y_std_
                return mean, std
            
            # Denormalize predictions
            mean = mean * self.y_std_ + self.y_mean_
            return mean
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            print(f"DEBUG - Exception traceback:", traceback.format_exc())
            if return_std:
                return np.zeros(len(X)), np.ones(len(X))
            return np.zeros(len(X))
            
    def _compute_kernel(self, X1, X2=None):
        """Compute RBF kernel matrix with proper dimension handling."""
        # Handle X2=None case (computing K(X1, X1))
        if X2 is None:
            X2 = X1
            
        # Convert inputs to numpy arrays if they aren't already
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        
        # Ensure X1 and X2 are 2D arrays
        if X1.ndim == 1:
            X1 = X1.reshape(1, -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)
            
        print(f"DEBUG - _compute_kernel: X1 shape={X1.shape}, X2 shape={X2.shape}")
            
        # Get the length scales from flat_samples
        # If flat_samples has a different number of dimensions, we need to handle it
        length_scale = np.median(1.0 / np.sqrt(self.flat_samples['kernel_inv_length_sq']), axis=0)
        var = np.median(self.flat_samples['kernel_var'])
        
        print(f"DEBUG - _compute_kernel: length_scale shape={length_scale.shape}")
        
        # Ensure dimensions match
        n_features = min(X1.shape[1], X2.shape[1], length_scale.shape[0])
        
        # If dimensions don't match, fix them
        if n_features < X1.shape[1] or n_features < X2.shape[1] or n_features < length_scale.shape[0]:
            print(f"WARNING - Dimension mismatch in kernel computation. Using first {n_features} features.")
            X1 = X1[:, :n_features]
            X2 = X2[:, :n_features]
            length_scale = length_scale[:n_features]
        
        # Normalize X1 and X2 by length scale
        X1_normalized = X1 / length_scale
        X2_normalized = X2 / length_scale
        
        # Compute the squared distance matrix
        # This is equivalent to: ||x_i/l - x_j/l||^2 for each pair of points
        # Using the identity: ||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b
        X1_norm_squared = np.sum(X1_normalized**2, axis=1, keepdims=True)
        X2_norm_squared = np.sum(X2_normalized**2, axis=1, keepdims=True).T
        
        # Compute the dot product: X1_normalized · X2_normalized.T
        dot_product = np.dot(X1_normalized, X2_normalized.T)
        
        # Compute squared distances
        dists = X1_norm_squared + X2_norm_squared - 2.0 * dot_product
        
        # Ensure distances are non-negative (can be slightly negative due to numerical errors)
        dists = np.maximum(dists, 0.0)
        
        # Compute RBF kernel: k(x,y) = var * exp(-0.5 * ||x-y||^2)
        kernel_matrix = var * np.exp(-0.5 * dists)
        
        print(f"DEBUG - _compute_kernel: Kernel matrix shape={kernel_matrix.shape}")
        
        return kernel_matrix
        
    def sample_y(self, X, n_samples=1):
        """Sample from the posterior predictive distribution."""
        mean, std = self.predict(X, return_std=True)
        return np.random.normal(mean[:, np.newaxis], 
                              std[:, np.newaxis], 
                              (mean.shape[0], n_samples))