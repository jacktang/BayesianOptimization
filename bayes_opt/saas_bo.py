"""SAAS-based Bayesian Optimization.

This module provides a SAAS-based extension to the standard BayesianOptimization class.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

from scipy.optimize import NonlinearConstraint
from numpy.random import RandomState

from bayes_opt.bayesian_optimization import BayesianOptimization, ensure_rng
from bayes_opt.domain_reduction import DomainTransformer
from bayes_opt.target_space import TargetSpace
from bayes_opt.saasgp import SAASGP
from bayes_opt.saasei import SAASExpectedImprovement
from bayes_opt.constraint import ConstraintModel


class SAASBayesianOptimization:
    """Bayesian optimization using Sparse Axis-Aligned Subspace Gaussian Process.
    
    This class extends the standard BayesianOptimization class with a SAAS-based
    surrogate model that is especially effective for high-dimensional problems.
    
    Parameters
    ----------
    f : callable
        Function to be maximized.
    pbounds : dict
        Dictionary with parameters names as keys and a tuple with minimum
        and maximum values as values.
    constraint : NonlinearConstraint, optional (default=None)
        Constraint function and bounds. If provided, the optimization will be constrained.
    alpha : float, default=0.1
        Controls sparsity in the SAAS prior.
    num_warmup : int, default=256
        Number of MCMC warmup iterations.
    num_samples : int, default=256
        Number of MCMC samples to draw after warmup.
    thinning : int, default=8
        Thinning factor for MCMC samples.
    random_state : int or numpy.random.RandomState, default=None
        If an integer, specifies the seed for the random number generator.
    verbose : int, default=1
        Verbosity level.
    bounds_transformer : DomainTransformer, default=None
        If provided, applies a transformation to the bounds.
    allow_duplicate_points : bool, default=False
        If True, allows the optimizer to suggest duplicate points.
    """
    
    def __init__(
        self,
        f: Callable,
        pbounds: Mapping[str, Tuple[float, float]],
        constraint: Optional[NonlinearConstraint] = None,
        alpha: float = 0.1,
        num_warmup: int = 256,
        num_samples: int = 256,
        max_tree_depth: int = 10,
        num_chains: int = 1,
        thinning: int = 8,
        random_state: Optional[Union[int, RandomState]] = None,
        verbose: int = 1,
        bounds_transformer: Optional[DomainTransformer] = None,
        allow_duplicate_points: bool = False,
    ):
        self._random_state = ensure_rng(random_state)
        self._verbose = verbose
        self._allow_duplicate_points = allow_duplicate_points
        self._bounds_transformer = bounds_transformer
        
        # Process constraint if provided
        if constraint is not None:
            self._constraint = ConstraintModel(
                fun=constraint.fun,
                lb=constraint.lb,
                ub=constraint.ub,
                random_state=random_state
            )
            self.is_constrained = True
        else:
            self._constraint = None
            self.is_constrained = False
        
        # Create target space
        self._space = TargetSpace(
            f, pbounds, 
            constraint=self._constraint,
            random_state=random_state, 
            allow_duplicate_points=allow_duplicate_points
        )
        
        # Initialize SAASGP surrogate model
        self._gp = SAASGP(
            alpha=alpha,
            num_warmup=num_warmup,
            num_samples=num_samples,
            max_tree_depth=max_tree_depth,
            num_chains=num_chains,
            thinning=thinning,
            verbose=verbose > 1,
            kernel="matern"
        )
        
        # Save the random_state value for later use in fit
        self._seed = random_state if isinstance(random_state, int) else np.random.randint(0, 10000)
        
        # Initialize acquisition function
        self._acquisition_function = SAASExpectedImprovement(
            xi=0.01,
            random_state=random_state
        )
        
        # Apply bounds transformer if provided
        if self._bounds_transformer:
            self._bounds_transformer.initialize(self._space)
        
        # Track number of points at last GP fit to detect when a new fit is needed
        self._n_points_at_last_fit = 0
    
    @property
    def space(self) -> TargetSpace:
        """Get the optimization space."""
        return self._space
    
    @property
    def max(self) -> Dict[str, Any]:
        """Get maximum observed target value and corresponding parameters."""
        return self._space.max()
    
    @property
    def res(self) -> list:
        """Get all target values and corresponding parameters."""
        return self._space.res()
    
    @property
    def constraint(self) -> Optional[ConstraintModel]:
        """Return the constraint associated with the optimizer, if any."""
        return self._constraint
    
    def _fit_gp(self):
        """Fit the GP model with the current data."""
        # Always fit with all current data
        X = self._space.params
        y = self._space.target
        
        print(f"TRACE - SAASBayesianOptimization._fit_gp: Entry point")
        print(f"TRACE - SAASBayesianOptimization._fit_gp: Space has {len(self._space)} points")
        print(f"TRACE - SAASBayesianOptimization._fit_gp: X shape={X.shape}, y shape={y.shape}")
        
        if hasattr(self, '_n_points_at_last_fit'):
            print(f"TRACE - SAASBayesianOptimization._fit_gp: Points at last fit = {self._n_points_at_last_fit}")
        
        # Check if the GP model has already been initialized 
        if hasattr(self._gp, 'X_train_') and hasattr(self._gp, 'Y_train'):
            print(f"TRACE - SAASBayesianOptimization._fit_gp: Current GP model state - X_train_: {self._gp.X_train_.shape}, Y_train: {self._gp.Y_train.shape}")
        else:
            print(f"TRACE - SAASBayesianOptimization._fit_gp: GP model not yet initialized")
        
        if len(X) < 2:
            if self._verbose:
                print("Not enough points to fit GP model. Need at least 2 points.")
            return False
            
        if self._verbose > 1:
            print(f"Fitting GP with {len(X)} points. X shape: {X.shape}, y shape: {y.shape}")
        
        # Verify that dimensions are consistent
        if X.shape[0] != y.shape[0]:
            print(f"TRACE - SAASBayesianOptimization._fit_gp: DIMENSION MISMATCH! X ({X.shape}) and y ({y.shape})")
            # Keep only the matching dimensions
            min_samples = min(X.shape[0], y.shape[0])
            X = X[:min_samples]
            y = y[:min_samples]
            print(f"TRACE - SAASBayesianOptimization._fit_gp: After fixing: X shape={X.shape}, y shape={y.shape}")
            
            # Reinitialize the GP model after fixing dimensions
            print("TRACE - SAASBayesianOptimization._fit_gp: Reinitializing SAASGP model due to dimension mismatch")
            self._gp = SAASGP(
                alpha=self._gp.alpha,
                num_warmup=self._gp.num_warmup,
                num_samples=self._gp.num_samples,
                max_tree_depth=self._gp.max_tree_depth,
                num_chains=self._gp.num_chains,
                thinning=self._gp.thinning,
                verbose=self._verbose > 1,
                kernel="matern"
            )
        
        # Always create a fresh GP for each fit to avoid any state issues
        try:
            print(f"TRACE - SAASBayesianOptimization._fit_gp: Calling gp.fit() with X:{X.shape}, y:{y.shape}")
            self._gp.fit(X, y, seed=self._seed)
            self._n_points_at_last_fit = len(X)
            print(f"TRACE - SAASBayesianOptimization._fit_gp: GP fit successful, _n_points_at_last_fit = {self._n_points_at_last_fit}")
            return True
        except Exception as e:
            print(f"ERROR during GP fitting: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return False
    
    def suggest(self, n_random: int = 10000, n_optimize: int = 5) -> Dict[str, float]:
        """Suggest next point to probe using SAAS-based acquisition function.
        
        Parameters
        ----------
        n_random : int, default=10000
            Number of random points to evaluate acquisition function.
        n_optimize : int, default=5
            Number of optimization runs for acquisition function.
            
        Returns
        -------
        dict
            Dictionary with suggested parameter values.
        """
        # Check if we have enough points to train a GP
        if len(self._space) < 2:
            if self._verbose:
                print("Not enough data for GP model. Using random sampling.")
            params = self._space.random_sample()
            return self._space.array_to_params(params) if isinstance(params, np.ndarray) else params
        
        # Always refit the GP with all current data
        fit_success = self._fit_gp()
        if not fit_success:
            print("GP fitting failed. Using random sampling.")
            params = self._space.random_sample()
            return self._space.array_to_params(params) if isinstance(params, np.ndarray) else params
        
        # Find the best feasible value
        if self.is_constrained:
            feasible_targets = []
            for res in self._space.res():
                if res.get('allowed', False):  # Only consider allowed points
                    feasible_targets.append(res['target'])
            y_best = max(feasible_targets) if feasible_targets else float('-inf')
        else:
            y_best = np.max(self._space.target)
        
        # Generate random points
        bounds = self._space.bounds
        random_points = np.zeros((n_random, self._space.dim))
        for i, (lower, upper) in enumerate(bounds):
            random_points[:, i] = self._random_state.uniform(lower, upper, size=n_random)
        
        # Evaluate acquisition function at random points
        # try:
        # Pass constraint directly to acquisition function
        acq_values = self._acquisition_function(
            self._gp, 
            random_points, 
            y_best,
            constraint=self._constraint if self.is_constrained else None
        )
        # except Exception as e:
        #     print(f"Error in acquisition function evaluation: {e}")
        #     print("Falling back to random sampling.")
        #     params = self._space.random_sample()
        #     return self._space.array_to_params(params) if isinstance(params, np.ndarray) else params
        
        # Find the best point from random sampling
        if len(acq_values) == 0 or np.all(np.isnan(acq_values)):
            print("All acquisition values are invalid. Using random sampling.")
            params = self._space.random_sample()
            return self._space.array_to_params(params) if isinstance(params, np.ndarray) else params
            
        # Filter out NaNs if any
        valid_indices = ~np.isnan(acq_values)
        if not np.any(valid_indices):
            print("No valid acquisition values. Using random sampling.")
            params = self._space.random_sample()
            return self._space.array_to_params(params) if isinstance(params, np.ndarray) else params
            
        acq_values = acq_values[valid_indices]
        random_points = random_points[valid_indices]
        
        best_idx = np.argmax(acq_values)
        x_best = random_points[best_idx]
        
        # Convert the suggested point to a dictionary
        return self._space.array_to_params(x_best)
            
 
    def register(
        self, 
        params: Dict[str, float], 
        target: float,
        constraint_value: Optional[Union[float, np.ndarray]] = None
    ) -> None:
        """Register an observation (parameters and target value).
        
        Parameters
        ----------
        params : dict
            Dictionary with the parameter values.
        target : float
            Target function value.
        constraint_value : float or numpy.ndarray, optional
            Value(s) of the constraint function(s) at the observation.
        """
        if self.is_constrained and constraint_value is None:
            # If constrained but no constraint value provided, evaluate constraint
            constraint_value = self._constraint.fun(**params)
        
        self._space.register(params, target, constraint_value)
    
    def maximize(
        self,
        init_points: int = 5,
        n_iter: int = 25,
        acq_samples: int = 32,
    ) -> None:
        """Maximize the objective function.
        
        Parameters
        ----------
        init_points : int, default=5
            Number of initial random points to probe.
        n_iter : int, default=25
            Number of iterations of Bayesian optimization.
        acq_samples : int, default=32
            Number of MCMC samples for acquisition function evaluation.
        """
        # Generate initial random points
        initial_points = []
        initial_targets = []
        initial_constraints = []
        
        for _ in range(init_points):
            params = self._space.random_sample()
            # Convert array to dictionary if needed
            if isinstance(params, np.ndarray):
                params = self._space.array_to_params(params)
            
            # Evaluate target function
            target = self._space.target_func(**params)
            
            # Evaluate constraint if present
            constraint_value = None
            if self.is_constrained:
                constraint_value = self._constraint.fun(**params)
            
            # Store values
            initial_points.append(self._space.params_to_array(params))
            initial_targets.append(target)
            if constraint_value is not None:
                initial_constraints.append(constraint_value)
            
            # Register the point
            self.register(params, target, constraint_value)
            
            if self._verbose >= 1:
                status = "allowed" if not self.is_constrained or self._constraint.allowed(constraint_value) else "not allowed"
                print(f"Initial point {_+1}/{init_points} | Target: {target:.4f} | Status: {status}")
        
        # Fit constraint model with initial points if using constraints
        if self.is_constrained and initial_constraints:
            X_init = np.vstack(initial_points)
            Y_init = np.vstack(initial_constraints) if len(initial_constraints[0].shape) > 0 else np.array(initial_constraints)
            self._constraint.fit(X_init, Y_init)
        
        # Main Bayesian optimization loop
        for i in range(n_iter):
            # Find the next point to probe
            next_point = self.suggest()  # This now always returns a dictionary
            
            # Evaluate target function
            target = self._space.target_func(**next_point)
            
            # Evaluate constraint if present
            constraint_value = None
            if self.is_constrained:
                constraint_value = self._constraint.fun(**next_point)
                
                # Update constraint model with new point
                X_new = self._space.params_to_array(next_point).reshape(1, -1)
                Y_new = np.array(constraint_value).reshape(1, -1) if isinstance(constraint_value, np.ndarray) else np.array([constraint_value])
                self._constraint.fit(X_new, Y_new)
            
            # Register the point
            self.register(next_point, target, constraint_value)
            
            if self._verbose >= 1:
                current_max = self._space.max()
                if current_max is not None:
                    best_target = current_max["target"]
                    status = "allowed" if not self.is_constrained or self._constraint.allowed(constraint_value) else "not allowed"
                    print(f"Iteration {i+1}/{n_iter} | Target: {target:.4f} | Best: {best_target:.4f} | Status: {status}")
                
            # Apply bounds transformer if provided
            if self._bounds_transformer and i >= init_points:
                self._bounds_transformer.transform(self._space)
    
    def set_bounds(self, new_bounds: Dict[str, Tuple[float, float]]) -> None:
        """Update the optimization bounds.
        
        Parameters
        ----------
        new_bounds : dict
            Dictionary with the new parameter bounds.
        """
        self._space.set_bounds(new_bounds)
        
        if self._bounds_transformer:
            self._bounds_transformer.initialize(self._space)

    def get_relevance_scores(self) -> np.ndarray:
        """Get the relevance scores for each dimension.
        
        Returns
        -------
        numpy.ndarray
            Array of relevance scores for each dimension.
            Higher scores indicate more relevant dimensions.
        """
        if not hasattr(self._gp, 'relevance_scores_'):
            # If GP hasn't been fit yet or doesn't have scores, return zeros
            return np.zeros(self._space.dim)
        return self._gp.relevance_scores_


# Example usage function
def saas_optimize(
    f: Callable,
    pbounds: Dict[str, Tuple[float, float]],
    n_iter: int = 25,
    init_points: int = 5,
    alpha: float = 0.1,
    num_warmup: int = 256,
    num_samples: int = 256,
    thinning: int = 8,
    random_state: Optional[int] = None,
    verbose: int = 1
) -> SAASBayesianOptimization:
    """Run SAAS-based Bayesian optimization on function f.
    
    Parameters
    ----------
    f : callable
        Function to be maximized.
    pbounds : dict
        Dictionary with parameters names as keys and a tuple with minimum
        and maximum values as values.
    n_iter : int, default=25
        Number of iterations of Bayesian optimization.
    init_points : int, default=5
        Number of initial random points to probe.
    alpha : float, default=0.1
        Controls sparsity in the SAAS prior.
    num_warmup : int, default=256
        Number of MCMC warmup iterations.
    num_samples : int, default=256
        Number of MCMC samples to draw after warmup.
    thinning : int, default=8
        Thinning factor for MCMC samples.
    random_state : int, default=None
        Seed for the random number generator.
    verbose : int, default=1
        Verbosity level.
        
    Returns
    -------
    SAASBayesianOptimization
        The optimizer object.
    """
    optimizer = SAASBayesianOptimization(
        f=f,
        pbounds=pbounds,
        alpha=alpha,
        num_warmup=num_warmup,
        num_samples=num_samples,
        thinning=thinning,
        random_state=random_state,
        verbose=verbose
    )
    
    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter,
    )
    
    return optimizer 