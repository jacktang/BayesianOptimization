"""Utility functions for analyzing SAAS-based Bayesian Optimization results."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd


def analyze_dimension_importance(optimizer) -> np.ndarray:
    """
    Analyze the importance of each dimension based on SAAS inverse lengthscales.
    
    Parameters
    ----------
    optimizer : SAASBayesianOptimization
        The optimizer after running optimization.
        
    Returns
    -------
    np.ndarray
        Normalized importance scores for each dimension.
    """
    # Extract the inverse lengthscale samples and convert to numpy if needed
    inv_length_sq = np.array(optimizer._gp.flat_samples['kernel_inv_length_sq'])
    
    # Calculate the median inverse lengthscale for each dimension
    median_inv_length = np.median(inv_length_sq, axis=0)
    
    # Normalize to get relative importance
    importance = median_inv_length / np.sum(median_inv_length)
    
    return importance


def plot_dimension_importance(
    optimizer, 
    param_names: Optional[List[str]] = None,
    threshold: float = 0.05,
    figsize: Tuple[int, int] = (10, 6)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the importance of each dimension discovered by SAAS.
    
    Parameters
    ----------
    optimizer : SAASBayesianOptimization
        The optimizer after running optimization.
    param_names : list of str, optional
        Names of parameters. If None, uses the keys from optimizer's bounds.
    threshold : float, default=0.05
        Threshold for considering a dimension important.
    figsize : tuple, default=(10, 6)
        Figure size for the plot.
        
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        The created figure and axes.
    """
    # Get dimension importance
    importance = analyze_dimension_importance(optimizer)
    
    # Get parameter names if not provided
    if param_names is None:
        param_names = list(optimizer._space.keys)
    
    # Create a dataframe for plotting
    df = pd.DataFrame({
        'Parameter': param_names,
        'Importance': importance
    })
    
    # Sort by importance
    df = df.sort_values('Importance', ascending=False)
    
    # Classify as important or not based on threshold
    df['Category'] = ['Important' if imp > threshold else 'Not Important' 
                     for imp in df['Importance']]
    
    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        x='Parameter', 
        y='Importance', 
        hue='Category',
        palette={'Important': 'darkblue', 'Not Important': 'lightgray'},
        data=df,
        ax=ax
    )
    
    ax.set_title('Dimension Importance from SAAS')
    ax.set_xlabel('Parameter')
    ax.set_ylabel('Relative Importance')
    ax.axhline(y=threshold, linestyle='--', color='red', alpha=0.7)
    ax.text(
        len(param_names) - 1, 
        threshold + 0.01, 
        f'Threshold: {threshold}', 
        ha='right', 
        va='bottom', 
        color='red'
    )
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig, ax


def plot_lengthscale_evolution(
    optimizer,
    param_names: Optional[List[str]] = None,
    top_k: int = 3,
    figsize: Tuple[int, int] = (12, 6)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize the posterior distribution of inverse lengthscales.
    
    Parameters
    ----------
    optimizer : SAASBayesianOptimization
        The optimizer after running optimization.
    param_names : list of str, optional
        Names of parameters. If None, uses the keys from optimizer's bounds.
    top_k : int, default=3
        Number of top dimensions to highlight.
    figsize : tuple, default=(12, 6)
        Figure size for the plot.
        
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        The created figure and axes.
    """
    # Get inverse lengthscale samples and convert to numpy if needed
    inv_length_sq = np.array(optimizer._gp.flat_samples['kernel_inv_length_sq'])
    
    # Get parameter names if not provided
    if param_names is None:
        param_names = list(optimizer._space.keys)
    
    # Get importance for sorting
    importance = analyze_dimension_importance(optimizer)
    
    # Get indices of top k dimensions
    top_indices = np.argsort(-importance)[:top_k]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot density for each dimension
    for i in range(len(param_names)):
        if i in top_indices:
            sns.kdeplot(
                inv_length_sq[:, i], 
                label=f"{param_names[i]} (Important)",
                ax=ax,
                linewidth=2
            )
        else:
            sns.kdeplot(
                inv_length_sq[:, i], 
                label=f"{param_names[i]}",
                ax=ax,
                alpha=0.3,
                linewidth=1
            )
    
    ax.set_title('Posterior Distribution of Inverse Lengthscales')
    ax.set_xlabel('Inverse Lengthscale')
    ax.set_ylabel('Density')
    
    # Only include top k in legend for clarity
    handles, labels = ax.get_legend_handles_labels()
    important_indices = [i for i, label in enumerate(labels) if "Important" in label]
    ax.legend([handles[i] for i in important_indices], 
              [labels[i] for i in important_indices],
              loc='best')
    
    plt.tight_layout()
    
    return fig, ax


def analyze_optimization_trajectory(
    optimizer,
    figsize: Tuple[int, int] = (12, 8)
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Analyze the optimization trajectory with both function value and sampling locations.
    
    Parameters
    ----------
    optimizer : SAASBayesianOptimization
        The optimizer after running optimization.
    figsize : tuple, default=(12, 8)
        Figure size for the plot.
        
    Returns
    -------
    fig, axes : matplotlib Figure and list of Axes
        The created figure and axes.
    """
    # Get results
    results = optimizer.res
    
    # Extract values and parameters
    targets = np.array([res['target'] for res in results])
    best_targets = np.maximum.accumulate(targets)
    iterations = np.arange(1, len(results) + 1)
    
    # Get importance
    importance = analyze_dimension_importance(optimizer)
    
    # Get top 2 dimensions for visualization
    top_dims = np.argsort(-importance)[:2]
    param_names = list(optimizer._space.keys)
    
    # Extract parameters
    params = np.array([[res['params'][key] for key in param_names] for res in results])
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Function value over iterations
    axes[0, 0].plot(iterations, targets, 'o-', label='Target')
    axes[0, 0].plot(iterations, best_targets, 'r-', label='Best Target')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Target Value')
    axes[0, 0].set_title('Optimization Progress')
    axes[0, 0].legend()
    
    # Plot 2: Sampling locations in top 2 dimensions
    scatter = axes[0, 1].scatter(
        params[:, top_dims[0]], 
        params[:, top_dims[1]], 
        c=targets, 
        cmap='viridis',
        s=50,
        edgecolors='k'
    )
    axes[0, 1].set_xlabel(param_names[top_dims[0]])
    axes[0, 1].set_ylabel(param_names[top_dims[1]])
    axes[0, 1].set_title('Sampling Locations (Top 2 Dimensions)')
    fig.colorbar(scatter, ax=axes[0, 1], label='Target Value')
    
    # Plot 3: Dimension importance
    sorted_idx = np.argsort(-importance)
    axes[1, 0].bar(
        np.arange(len(importance)), 
        importance[sorted_idx],
        tick_label=np.array(param_names)[sorted_idx]
    )
    axes[1, 0].set_xlabel('Parameter')
    axes[1, 0].set_ylabel('Relative Importance')
    axes[1, 0].set_title('Dimension Importance')
    plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
    
    # Plot 4: Sampling sequence in the top dimension
    axes[1, 1].plot(iterations, params[:, top_dims[0]], 'o-', 
                   label=f'Top dimension: {param_names[top_dims[0]]}')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Parameter Value')
    axes[1, 1].set_title('Sampling Sequence (Top Dimension)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    return fig, axes


def extract_optimization_insights(optimizer) -> Dict[str, Any]:
    """
    Extract key insights from the optimization run.
    
    Parameters
    ----------
    optimizer : SAASBayesianOptimization
        The optimizer after running optimization.
        
    Returns
    -------
    dict
        Dictionary with key insights about the optimization.
    """
    # Get dimension importance
    importance = analyze_dimension_importance(optimizer)
    param_names = list(optimizer._space.keys)
    
    # Get results
    results = optimizer.res
    targets = np.array([res['target'] for res in results])
    best_target = np.max(targets)
    best_params = optimizer.max['params']
    
    # Identify important dimensions (above 5% importance)
    important_dims = [(param_names[i], importance[i]) 
                      for i in range(len(param_names)) 
                      if importance[i] > 0.05]
    
    # Compute convergence metrics
    iterations = len(results)
    best_values = np.maximum.accumulate(targets)
    
    # Calculate improvement in last 25% of iterations
    last_quarter = max(1, iterations // 4)
    recent_improvement = (best_values[-1] - best_values[-last_quarter]) / last_quarter
    
    return {
        'best_target': best_target,
        'best_params': best_params,
        'important_dimensions': important_dims,
        'dimension_importance': {param_names[i]: importance[i] for i in range(len(param_names))},
        'iterations': iterations,
        'convergence_metrics': {
            'recent_improvement_rate': recent_improvement,
            'final_improvement_rate': (best_values[-1] - best_values[-2]) if iterations > 1 else 0
        }
    } 