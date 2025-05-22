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
            
    def __call__(self, gp, X, y_best, constraint=None):
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
        constraint : ConstraintModel, optional
            Constraint model for constrained optimization.
            
        Returns
        -------
        ei : array-like of shape (n_samples,)
            Expected improvement at points X.
        """
        # Get mean and standard deviation predictions using MCMC samples
        mean, std = gp.predict(X, return_std=True)
        
        # Calculate improvement
        # Dynamically adjust xi based on optimization progress
        n_points = len(gp.X_train_)
        adaptive_xi = self.xi * np.exp(-0.1 * n_points)  # Decrease exploration over time
        improvement = mean - y_best - adaptive_xi
        
        # Calculate Z score with numerical stability
        z = improvement / (std + 1e-9)
        
        # Calculate expected improvement using analytical formula
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        
        # Set EI to 0 where std is 0 (no uncertainty)
        ei = jnp.where(std == 0, 0.0, ei)
        
        # Handle constraints if present
        if constraint is not None:
            # Get probability of satisfying constraints
            prob_feasible = constraint.predict(X)
            
            # Calculate constrained expected improvement
            # Use a softer penalty for points near constraint boundaries
            penalty = 1.0 / (1.0 + np.exp(-10 * (prob_feasible - 0.5)))
            ei = ei * penalty
        
        # Convert to numpy array for compatibility
        return np.array(ei)
            
    
    def sample_based_ei(self, gp, X, y_best, constraint=None, n_samples=32):
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
        constraint : ConstraintModel, optional
            Constraint model for constrained optimization.
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
        # Use adaptive xi here too
        n_points = len(gp.X_train_)
        adaptive_xi = self.xi * np.exp(-0.1 * n_points)
        improvement = np.maximum(y_samples - y_best - adaptive_xi, 0)
        
        # Average over samples
        expected_improvement = np.mean(improvement, axis=1)
        
        # Handle constraints if present
        if constraint is not None:
            try:
                prob_feasible = constraint.predict(X)
                penalty = 1.0 / (1.0 + np.exp(-10 * (prob_feasible - 0.5)))
                expected_improvement = expected_improvement * penalty
            except ValueError as e:
                # If constraint model hasn't been fit yet, return unconstrained EI
                print(f"Warning: {str(e)}. Using unconstrained EI.")
                pass
        
        return expected_improvement