"""revlearn: Bayesian vs Rescorla-Wagner models on probabilistic reversal learning."""
from .models import MODELS, BayesianObserver, RescorlaWagner
from .task import Schedule, TaskConfig, make_schedule, schedule_from_best

__all__ = [
    "MODELS", "BayesianObserver", "RescorlaWagner",
    "Schedule", "TaskConfig", "make_schedule", "schedule_from_best",
]
__version__ = "1.0.0"
