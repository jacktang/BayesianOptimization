"""Bayesian optimization package.

This package provides both standard Bayesian optimization and SAAS-based Bayesian
optimization for high-dimensional problems.
"""

from __future__ import annotations

import importlib.metadata

from bayes_opt import acquisition
from bayes_opt.bayesian_optimization import BayesianOptimization
from bayes_opt.constraint import ConstraintModel
from bayes_opt.domain_reduction import SequentialDomainReductionTransformer
from bayes_opt.logger import ScreenLogger
from bayes_opt.target_space import TargetSpace
from bayes_opt.saas_bo import SAASBayesianOptimization, saas_optimize
from bayes_opt.saasei import SAASExpectedImprovement

# __version__ = importlib.metadata.version("bayesian-optimization")
__version__ = "0.1.0"

__all__ = [
    "acquisition",
    "BayesianOptimization",
    "TargetSpace",
    "ConstraintModel",
    "ScreenLogger",
    "SequentialDomainReductionTransformer",
    "SAASBayesianOptimization",
    "saas_optimize",
    "SAASExpectedImprovement",
]
