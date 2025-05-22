"""Comparison of SAAS and standard Bayesian optimization on a simple test function.

This script compares the performance of SAAS-based and standard
Bayesian optimization on a 10D test function where only 2 dimensions are relevant.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.optimize import NonlinearConstraint
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bayes_opt import BayesianOptimization
from bayes_opt.saas_bo import SAASBayesianOptimization

@dataclass
class OptimizationResult:
    """Class to store results of a single optimization run."""
    best_value: float
    convergence_path: List[float]
    runtime: float
    n_feasible: int
    n_infeasible: int

def run_optimization_trial(optimizer_type: str, seed: int) -> OptimizationResult:
    """Run a single optimization trial with given parameters.
    
    Args:
        optimizer_type: Either 'standard' or 'saas'
        seed: Random seed for reproducibility
        
    Returns:
        OptimizationResult object containing the optimization metrics
    """
    # Problem setup
    pbounds = {'x': (0, 6), 'y': (0, 6)}
    
    def target_function(x, y):
        return np.cos(2*x)*np.cos(y) + np.sin(x)

    def constraint_function_2_dim(x, y):
        return np.array([
            - np.cos(x) * np.cos(y) + np.sin(x) * np.sin(y),
            - np.cos(x) * np.cos(-y) + np.sin(x) * np.sin(-y)])

    constraint_lower = np.array([-np.inf, -np.inf])
    constraint_upper = np.array([0.6, 0.6])
    
    constraint = NonlinearConstraint(constraint_function_2_dim, constraint_lower, constraint_upper)
    
    # Create optimizer
    start_time = time.time()
    if optimizer_type == 'standard':
        optimizer = BayesianOptimization(
            f=target_function,
            constraint=constraint,
            pbounds=pbounds,
            verbose=0,
            random_state=seed
        )
    else:  # saas
        optimizer = SAASBayesianOptimization(
            f=target_function,
            constraint=constraint,
            pbounds=pbounds,
            alpha=0.1,
            num_warmup=128,
            num_samples=64,
            thinning=4,
            random_state=seed,
            verbose=0
        )
    
    # Run optimization
    optimizer.maximize(init_points=5, n_iter=15)
    runtime = time.time() - start_time
    
    # Collect results
    res = optimizer.res
    convergence = []
    best_so_far = float('-inf')

    for r in res:
        if r['allowed']:
            best_so_far = max(best_so_far, r['target'])
        convergence.append(best_so_far)
    
    # Count feasible/infeasible points
    n_feasible = sum(1 for r in res if r['allowed'])
    n_infeasible = len(res) - n_feasible
    
    return OptimizationResult(
        best_value=optimizer.max['target'] if optimizer.max else float('-inf'),
        convergence_path=convergence,
        runtime=runtime,
        n_feasible=n_feasible,
        n_infeasible=n_infeasible
    )

def run_comparison(n_trials: int = 10) -> Tuple[List[OptimizationResult], List[OptimizationResult]]:
    """Run multiple trials of both optimization methods.
    
    Args:
        n_trials: Number of trials to run for each method
        
    Returns:
        Tuple of (standard_results, saas_results)
    """
    standard_results = []
    saas_results = []
    
    for i in range(n_trials):
        print(f"Running trial {i+1}/{n_trials}")
        standard_results.append(run_optimization_trial('standard', seed=i))
        saas_results.append(run_optimization_trial('saas', seed=i))
    
    return standard_results, saas_results

def plot_comparison_results(standard_results: List[OptimizationResult], 
                          saas_results: List[OptimizationResult]):
    """Create comparison plots for the optimization results."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 5))
    
    # 1. Convergence comparison
    std_convergence = np.array([r.convergence_path for r in standard_results])
    saas_convergence = np.array([r.convergence_path for r in saas_results])
    
    x = range(1, std_convergence.shape[1] + 1)
    ax1.plot(x, std_convergence.mean(axis=0), 'b-', label='Standard BO')
    ax1.fill_between(x, 
                    std_convergence.mean(axis=0) - std_convergence.std(axis=0),
                    std_convergence.mean(axis=0) + std_convergence.std(axis=0),
                    alpha=0.2)
    ax1.plot(x, saas_convergence.mean(axis=0), 'r-', label='SAAS BO')
    ax1.fill_between(x, 
                    saas_convergence.mean(axis=0) - saas_convergence.std(axis=0),
                    saas_convergence.mean(axis=0) + saas_convergence.std(axis=0),
                    alpha=0.2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Best feasible value')
    ax1.set_title('Convergence Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Runtime comparison
    std_times = [r.runtime for r in standard_results]
    saas_times = [r.runtime for r in saas_results]
    ax2.boxplot([std_times, saas_times], labels=['Standard BO', 'SAAS BO'])
    ax2.set_ylabel('Runtime (seconds)')
    ax2.set_title('Runtime Comparison')
    ax2.grid(True)
    
    # 3. Final value comparison
    std_values = [r.best_value for r in standard_results]
    saas_values = [r.best_value for r in saas_results]
    ax3.boxplot([std_values, saas_values], labels=['Standard BO', 'SAAS BO'])
    ax3.set_ylabel('Best feasible value found')
    ax3.set_title('Final Value Comparison')
    ax3.grid(True)
    
    # 4. Feasible vs Infeasible points
    std_feasible = np.mean([r.n_feasible for r in standard_results])
    std_infeasible = np.mean([r.n_infeasible for r in standard_results])
    saas_feasible = np.mean([r.n_feasible for r in saas_results])
    saas_infeasible = np.mean([r.n_infeasible for r in saas_results])
    
    x = np.arange(2)
    width = 0.35
    ax4.bar(x - width/2, [std_feasible, saas_feasible], width, label='Feasible')
    ax4.bar(x + width/2, [std_infeasible, saas_infeasible], width, label='Infeasible')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Standard BO', 'SAAS BO'])
    ax4.set_ylabel('Average number of points')
    ax4.set_title('Feasible vs Infeasible Points')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('constrained_optimization_comparison.png', dpi=90)
    plt.show()

def plot_constrained_opt(pbounds, target_function, optimizer):
    """
    Plots a number of interesting contours to visualize constrained 2-dimensional optimization.
    """

    # Set a few parameters
    n_constraints = optimizer.constraint.lb.size
    n_plots_per_row = 2+n_constraints

    # Construct the subplot titles
    if n_constraints==1:
        c_labels = ["constraint"]
    else:
        c_labels = [f"constraint {i+1}" for i in range(n_constraints)]
    labels_top = ["target"] + c_labels + ["masked target"]
    labels_bot = ["target estimate"] + [c + " estimate" for c in c_labels] + ["acquisition function"]
    labels = [labels_top, labels_bot]

    # Setup the grid to plot on
    x = np.linspace(pbounds['x'][0], pbounds['x'][1], 1000)
    y = np.linspace(pbounds['y'][0], pbounds['y'][1], 1000)
    xy = np.array([[x_i, y_j] for y_j in y for x_i in x])
    X, Y = np.meshgrid(x, y)

    # Evaluate the actual functions on the grid
    Z = target_function(X, Y)
    # This reshaping is a bit painful admittedly, but it's a consequence of np.meshgrid
    C = optimizer.constraint.fun(X, Y).reshape((n_constraints,) + Z.shape).swapaxes(0, -1)
    
    
    fig, axs = plt.subplots(2, n_plots_per_row, constrained_layout=True, figsize=(12,8))

    for i in range(2):
        for j in range(n_plots_per_row):
            axs[i, j].set_aspect("equal")
            axs[i, j].set_title(labels[i][j])
    
    
    # Extract & unpack the optimization results
    max_ = optimizer.max
    res = optimizer.res
    x_ = np.array([r["params"]['x'] for r in res])
    y_ = np.array([r["params"]['y'] for r in res])
    c_ = np.array([r["constraint"] for r in res])
    a_ = np.array([r["allowed"] for r in res])


    Z_est = optimizer._gp.predict(xy).reshape(Z.shape)
    C_est = optimizer.constraint.approx(xy).reshape(Z.shape + (n_constraints,))
    P_allowed = optimizer.constraint.predict(xy).reshape(Z.shape)

    Acq = np.where(Z_est >0, Z_est * P_allowed, Z_est / (0.5 + P_allowed))
    
    
    target_vbounds = np.min([Z, Z_est]), np.max([Z, Z_est])
    constraint_vbounds = np.min([C, C_est]), np.max([C, C_est])


    axs[0,0].contourf(X, Y, Z, cmap=plt.cm.coolwarm, vmin=target_vbounds[0], vmax=target_vbounds[1])
    for i in range(n_constraints):
        axs[0,1+i].contourf(X, Y, C[:,:,i], cmap=plt.cm.coolwarm, vmin=constraint_vbounds[0], vmax=constraint_vbounds[1])
    Z_mask = Z

    Z_mask[~np.squeeze(optimizer.constraint.allowed(C))] = np.nan
    axs[0,n_plots_per_row-1].contourf(X, Y, Z_mask, cmap=plt.cm.coolwarm, vmin=target_vbounds[0], vmax=target_vbounds[1])

    axs[1,0].contourf(X, Y, Z_est, cmap=plt.cm.coolwarm, vmin=target_vbounds[0], vmax=target_vbounds[1])
    for i in range(n_constraints):
        axs[1,1+i].contourf(X, Y, C_est[:, :, i], cmap=plt.cm.coolwarm, vmin=constraint_vbounds[0], vmax=constraint_vbounds[1])
    axs[1,n_plots_per_row-1].contourf(X, Y, Acq, cmap=plt.cm.coolwarm, vmin=0, vmax=1)

    for i in range(2):
        for j in range(n_plots_per_row):
            axs[i,j].scatter(x_[a_], y_[a_], c='white', s=80, edgecolors='black')
            axs[i,j].scatter(x_[~a_], y_[~a_], c='red', s=80, edgecolors='black')
            axs[i,j].scatter(max_["params"]['x'], max_["params"]['y'], s=80, c='green', edgecolors='black')

    return fig, axs

def main():
    """Run the comparison between standard and SAAS BO."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run statistical comparison
    print("\n=== Running Statistical Comparison ===")
    standard_results, saas_results = run_comparison(n_trials=10)
    
    # Plot statistical results
    plot_comparison_results(standard_results, saas_results)
    
    # Run single instance for detailed visualization
    print("\n=== Running Single Instance Visualization ===")
    pbounds = {'x': (0, 6), 'y': (0, 6)}
    
    def target_function(x, y):
        return np.cos(2*x)*np.cos(y) + np.sin(x)

    def constraint_function_2_dim(x, y):
        return np.array([
            - np.cos(x) * np.cos(y) + np.sin(x) * np.sin(y),
            - np.cos(x) * np.cos(-y) + np.sin(x) * np.sin(-y)])

    constraint_lower = np.array([-np.inf, -np.inf])
    constraint_upper = np.array([0.6, 0.6])
    
    constraint = NonlinearConstraint(constraint_function_2_dim, constraint_lower, constraint_upper)
    
    # Run both optimizers with exactly the same configuration
    optimizer = SAASBayesianOptimization(
        f=target_function,
        constraint=constraint,
        pbounds=pbounds,
        alpha=0.1,
        num_warmup=128,
        num_samples=64,
        thinning=4,
        random_state=42,
        verbose=1
    )
    
    optimizer.maximize(init_points=5, n_iter=15)
    
    # Create visualization of the final state
    plot_constrained_opt(pbounds, target_function, optimizer)
    plt.savefig('final_optimization_state.png')
    plt.show()

if __name__ == "__main__":
    main() 