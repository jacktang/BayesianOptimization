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
import time

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bayes_opt import BayesianOptimization
from bayes_opt.saas_bo import SAASBayesianOptimization

def test_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    """
    Test function for optimization comparison.
    
    Only x1 and x2 are relevant, the rest are noise dimensions.
    This is a 10D version of the 2D function:
    f(x1, x2) = sin(x1) * cos(x2) - 0.5 * x1^2 - 0.5 * x2^2
    """
    return np.sin(x1) * np.cos(x2) - 0.5 * x1**2 - 0.5 * x2**2

# Define parameter bounds
pbounds = {
    'x1': (0, 1), 'x2': (0, 1), 'x3': (0, 1), 'x4': (0, 1), 'x5': (0, 1),
    'x6': (0, 1), 'x7': (0, 1), 'x8': (0, 1), 'x9': (0, 1), 'x10': (0, 1)
}

def run_standard_bo(init_points=5, n_iter=15):
    """Run standard Bayesian optimization."""
    print("\n=== Running standard Bayesian optimization ===")
    print(f"Configuration: init_points={init_points}, n_iter={n_iter}")
    start_time = time.time()
    
    optimizer = BayesianOptimization(
        f=test_function,
        pbounds=pbounds,
        random_state=42,
        verbose=1
    )
    
    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter
    )
    
    elapsed_time = time.time() - start_time
    print(f"\nStandard BO completed in {elapsed_time:.2f} seconds")
    print(f"Total points sampled: {len(optimizer.res)}")
    
    return optimizer, elapsed_time

def run_saas_bo(init_points=5, n_iter=15):
    """Run SAAS-based Bayesian optimization."""
    print("\n=== Running SAAS-based Bayesian optimization ===")
    print(f"Configuration: init_points={init_points}, n_iter={n_iter}")
    start_time = time.time()
    
    optimizer = SAASBayesianOptimization(
        f=test_function,
        pbounds=pbounds,
        alpha=0.5,             # SAAS sparsity parameter
        num_warmup=128,        # MCMC warmup iterations
        num_samples=64,        # MCMC samples
        thinning=4,            # Thinning to reduce computation
        random_state=42,
        verbose=1
    )
    
    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter
    )
    
    elapsed_time = time.time() - start_time
    print(f"\nSAAS BO completed in {elapsed_time:.2f} seconds")
    print(f"Total points sampled: {len(optimizer.res)}")
    
    return optimizer, elapsed_time

def create_comparison_visualization(standard_opt, saas_opt, resolution=50, init_points=5):
    """Create visualizations comparing standard and SAAS optimization."""
    # Extract results
    std_results = standard_opt.res
    saas_results = saas_opt.res
    
    print(f"\n=== Visualization Summary ===")
    print(f"Standard BO points: {len(std_results)}")
    print(f"SAAS BO points: {len(saas_results)}")
    
    # Extract values for the relevant dimensions (x1 and x2)
    std_x1 = [res['params']['x1'] for res in std_results]
    std_x2 = [res['params']['x2'] for res in std_results]
    std_values = [res['target'] for res in std_results]
    
    saas_x1 = [res['params']['x1'] for res in saas_results]
    saas_x2 = [res['params']['x2'] for res in saas_results]
    saas_values = [res['target'] for res in saas_results]
    
    # Create grid for true function visualization
    x1_range = np.linspace(0, 1, resolution)
    x2_range = np.linspace(0, 1, resolution)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # Calculate true function values
    Z = np.zeros((resolution, resolution))
    default_args = [0.5] * 10  # Default values for non-relevant dimensions
    for i in range(resolution):
        for j in range(resolution):
            args = default_args.copy()
            args[0] = X1[i, j]  # x1
            args[1] = X2[i, j]  # x2
            Z[i, j] = test_function(*args)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(10, 5))
    
    # Plot 1: Standard BO results
    ax1 = fig.add_subplot(231)
    contour = ax1.contourf(X1, X2, Z, levels=20, cmap='viridis')
    # Add iteration numbers to points
    for i, (x1, x2) in enumerate(zip(std_x1, std_x2)):
        ax1.scatter(x1, x2, c='red', s=100, edgecolors='white')
        if i < init_points:
            ax1.annotate(f'R{i+1}', (x1, x2), xytext=(5, 5), textcoords='offset points', color='white')
        else:
            ax1.annotate(f'{i+1}', (x1, x2), xytext=(5, 5), textcoords='offset points', color='white')
    ax1.set_title('Standard BO: Sampled Points')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    plt.colorbar(contour, ax=ax1)
    
    # Plot 2: SAAS BO results
    ax2 = fig.add_subplot(232)
    contour = ax2.contourf(X1, X2, Z, levels=20, cmap='viridis')
    # Add iteration numbers to points
    for i, (x1, x2) in enumerate(zip(saas_x1, saas_x2)):
        ax2.scatter(x1, x2, c='red', s=100, edgecolors='white')
        if i < init_points:
            ax2.annotate(f'R{i+1}', (x1, x2), xytext=(5, 5), textcoords='offset points', color='white')
        else:
            ax2.annotate(f'{i+1}', (x1, x2), xytext=(5, 5), textcoords='offset points', color='white')
    ax2.set_title('SAAS BO: Sampled Points')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    plt.colorbar(contour, ax=ax2)
    
    # Plot 3: Convergence comparison
    ax3 = fig.add_subplot(233)
    std_best = np.maximum.accumulate(std_values)
    saas_best = np.maximum.accumulate(saas_values)
    iterations = range(1, len(std_values) + 1)
    
    ax3.plot(iterations, std_best, 'b-', label='Standard BO')
    ax3.plot(iterations, saas_best, 'r-', label='SAAS BO')
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Best Function Value')
    ax3.set_title('Convergence Comparison')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: 3D surface with Standard BO points
    ax4 = fig.add_subplot(234, projection='3d')
    surf = ax4.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)
    ax4.scatter(std_x1, std_x2, [test_function(*([x1, x2] + [0.5]*8)) 
                                for x1, x2 in zip(std_x1, std_x2)],
                c='red', marker='o')
    ax4.set_title('Standard BO: 3D View')
    ax4.set_xlabel('x1')
    ax4.set_ylabel('x2')
    
    # Plot 5: 3D surface with SAAS BO points
    ax5 = fig.add_subplot(235, projection='3d')
    surf = ax5.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)
    ax5.scatter(saas_x1, saas_x2, [test_function(*([x1, x2] + [0.5]*8)) 
                                  for x1, x2 in zip(saas_x1, saas_x2)],
                c='red', marker='o')
    ax5.set_title('SAAS BO: 3D View')
    ax5.set_xlabel('x1')
    ax5.set_ylabel('x2')
    
    # Plot 6: SAAS dimension relevance
    ax6 = fig.add_subplot(236)
    relevance = saas_opt.get_relevance_scores()
    dimensions = range(1, len(relevance) + 1)
    ax6.bar(dimensions, relevance)
    ax6.set_xlabel('Dimension')
    ax6.set_ylabel('Relevance Score')
    ax6.set_title('SAAS: Dimension Relevance')
    
    plt.tight_layout()
    plt.savefig('simple_optimization_comparison.png')
    plt.show()

def main():
    """Run the comparison between standard and SAAS BO."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run both optimizers with exactly the same configuration
    init_points = 5  # Initial random points
    n_iter = 15      # Optimization iterations
    
    print("\n=== Starting Optimization Comparison ===")
    print(f"Configuration:")
    print(f"- Initial random points: {init_points}")
    print(f"- Optimization iterations: {n_iter}")
    print(f"- Total expected points: {init_points + n_iter}")
    
    standard_opt, std_time = run_standard_bo(init_points, n_iter)
    saas_opt, saas_time = run_saas_bo(init_points, n_iter)
    
    # Create visualization
    create_comparison_visualization(standard_opt, saas_opt, init_points=init_points)
    
    # Print summary
    print("\n=== Optimization Comparison Summary ===")
    print(f"Standard BO:")
    print(f"- Time: {std_time:.2f} seconds")
    print(f"- Points sampled: {len(standard_opt.res)}")
    print(f"- Best value: {standard_opt.max['target']:.6f}")
    
    print(f"\nSAAS BO:")
    print(f"- Time: {saas_time:.2f} seconds")
    print(f"- Points sampled: {len(saas_opt.res)}")
    print(f"- Best value: {saas_opt.max['target']:.6f}")
    
    # Print SAAS relevance scores
    print("\nSAAS dimension relevance scores:")
    relevance = saas_opt.get_relevance_scores()
    for i, score in enumerate(relevance):
        print(f"Dimension {i+1}: {score:.6f}")

if __name__ == "__main__":
    main() 