import numpy as np
from typing import Optional
from scipy.stats import norm
import jax.numpy as jnp

class SAASExpectedImprovement:
    """
    Expected Improvement acquisition function adapted for SAAS GP.
    
    This implementation works with sample-based posteriors from SAASGP
    rather than analytical expressions.
    
    Parameters
    ----------
    xi : float, default=0.01
        Exploration-exploitation trade-off parameter.
    """
    
    def __init__(self, xi: float = 0.01, random_state: Optional[int] = None):
        """Initialize the SAASExpectedImprovement acquisition function."""
        self.xi = xi
        if random_state is not None:
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = np.random.RandomState()
            
    def __call__(self, gp, X, y_best):
        """
        Calculate expected improvement at points X based on MCMC samples.
        
        Parameters
        ----------
        gp : SAASGP instance
            The sparse axis-aligned subspace GP surrogate model.
        X : array-like of shape (n_samples, n_features)
            Points at which to calculate the acquisition function.
        y_best : float
            The current best observed value.
            
        Returns
        -------
        ei : array-like of shape (n_samples,)
            Expected improvement at points X.
        """
        import traceback
        print(f"DEBUG - SAASExpectedImprovement.__call__: X shape={np.array(X).shape}, y_best={y_best}")
        
        # Add logging of GP state
        print(f"TRACE - SAASExpectedImprovement.__call__: GP state - X_train_: {gp.X_train_.shape}, Y_train: {gp.Y_train.shape}")
        print(f"TRACE - SAASExpectedImprovement.__call__: GP Ls: {'None' if gp.Ls is None else gp.Ls.shape}")
        
        try:
            # Get mean and standard deviation predictions using MCMC samples
            print("TRACE - SAASExpectedImprovement.__call__: Calling gp.predict()...")
            mean, std = gp.predict(X, return_std=True)
            print(f"DEBUG - EI calculation: mean shape={mean.shape}, std shape={std.shape}")
            
            # Calculate improvement
            improvement = mean - y_best - self.xi
            
            # Calculate Z score
            z = improvement / (std + 1e-9)
            
            # Calculate expected improvement using analytical formula
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            
            # Set EI to 0 where std is 0 (no uncertainty)
            # Use jnp.where instead of direct assignment since JAX arrays are immutable
            ei = jnp.where(std == 0, 0.0, ei)
            
            print(f"DEBUG - EI result shape={ei.shape}")
            
            # Convert to numpy array for compatibility
            return np.array(ei)
        except Exception as e:
            print(f"ERROR in acquisition function: {e}")
            print(f"TRACE - Exception information:")
            print(f"  GP model state: X_train_={gp.X_train_.shape}, Y_train={gp.Y_train.shape}")
            print(f"  GP Ls shape: {'None' if gp.Ls is None else gp.Ls.shape}")
            print(f"Traceback: {traceback.format_exc()}")
            # Return zeros as a fallback
            return np.zeros(X.shape[0])
    
    def sample_based_ei(self, gp, X, y_best, n_samples=32):
        """
        Calculate expected improvement at points X using direct MCMC samples.
        
        This method is computationally more expensive but more accurate for
        complex posteriors.
        
        Parameters
        ----------
        gp : SAASGP instance
            The sparse axis-aligned subspace GP surrogate model.
        X : array-like of shape (n_samples, n_features)
            Points at which to calculate the acquisition function.
        y_best : float
            The current best observed value.
        n_samples : int, default=32
            Number of MCMC samples to use for the calculation.
            
        Returns
        -------
        ei : array-like of shape (n_samples,)
            Expected improvement at points X.
        """
        # Get posterior samples at X
        y_samples = gp.sample_y(X, n_samples=n_samples)
        
        # Calculate improvement for each sample
        improvement = np.maximum(y_samples - y_best - self.xi, 0)
        
        # Average over samples
        expected_improvement = np.mean(improvement, axis=1)
        
        return expected_improvement