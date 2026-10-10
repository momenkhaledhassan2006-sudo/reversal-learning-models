"""Two computational models of choice in a probabilistic reversal-learning task.

Both models share the *same* choice rule -- a softmax (logistic) on a signed value
difference -- so any difference in fit is attributable to the *learning rule*:

* ``RescorlaWagner``  : model-free delta rule with a fixed learning rate; only the
                        chosen option is updated.  theta = (alpha, beta)
* ``BayesianObserver``: model-based hidden-state filter.  The agent believes one
                        option is currently "better" (pays off w.p. ``p``) and that
                        this can flip on any trial with hazard ``h``.  It keeps a
                        belief ``b = P(option 1 is the better one)`` and updates it
                        with Bayes' rule, then diffuses it through the reversal
                        hazard.  theta = (h, p, beta)

Every model exposes the same interface:

    trial_loglik(theta, choices, rewards) -> (T,) log P(choice_t | history, theta)
    simulate(theta, schedule, rng)        -> choices, rewards, p_chosen
    log_prior(theta)                      -> unnormalised log prior (for MAP / MCMC)
    sample_params(rng)                    -> parameters used to simulate agents
"""
from __future__ import annotations

import math

import numpy as np

from .task import Schedule


def _log_sigmoid(z: float) -> float:
    """log(1 / (1 + exp(-z))), stable for the parameter ranges used here."""
    if z >= 0:
        return -math.log1p(math.exp(-z))
    return z - math.log1p(math.exp(z))


def _beta_logpdf_unnorm(x: float, a: float, b: float) -> float:
    return (a - 1.0) * math.log(x) + (b - 1.0) * math.log1p(-x)


def _gamma_logpdf_unnorm(x: float, shape: float, scale: float) -> float:
    return (shape - 1.0) * math.log(x) - x / scale


class RescorlaWagner:
    """Q-learning with a delta rule and softmax choice rule.

    Q_c <- Q_c + alpha * (r - Q_c)   (chosen option c only)
    P(choose 1) = sigmoid(beta * (Q_1 - Q_0))
    """

    name = "RW"
    param_names = ("alpha", "beta")
    bounds = ((0.001, 0.999), (0.01, 20.0))
    start_box = ((0.05, 0.95), (0.5, 10.0))
    n_params = 2

    def sample_params(self, rng: np.random.Generator) -> np.ndarray:
        return np.array([rng.uniform(0.05, 0.9), rng.uniform(1.0, 10.0)])

    def log_prior(self, theta) -> float:
        alpha, beta = theta
        return _beta_logpdf_unnorm(alpha, 1.2, 1.2) + _gamma_logpdf_unnorm(beta, 1.2, 5.0)

    def trial_loglik(self, theta, choices, rewards) -> np.ndarray:
        alpha, beta = float(theta[0]), float(theta[1])
        q0 = q1 = 0.5
        out = np.empty(len(choices))
        for t in range(len(choices)):
            z = beta * (q1 - q0)
            c = choices[t]
            out[t] = _log_sigmoid(z if c == 1 else -z)
            r = rewards[t]
            if c == 1:
                q1 += alpha * (r - q1)
            else:
                q0 += alpha * (r - q0)
        return out

    def simulate(self, theta, schedule: Schedule, rng: np.random.Generator):
        alpha, beta = float(theta[0]), float(theta[1])
        T = schedule.n_trials
        u, v = rng.random(T), rng.random(T)
        choices, rewards, p_chosen = np.empty(T, int), np.empty(T, int), np.empty(T)
        q0 = q1 = 0.5
        for t in range(T):
            p1 = 1.0 / (1.0 + math.exp(-beta * (q1 - q0)))
            c = 1 if u[t] < p1 else 0
            r = 1 if v[t] < schedule.reward_prob[t, c] else 0
            choices[t], rewards[t], p_chosen[t] = c, r, p1 if c == 1 else 1.0 - p1
            if c == 1:
                q1 += alpha * (r - q1)
            else:
                q0 += alpha * (r - q0)
        return choices, rewards, p_chosen


class BayesianObserver:
    """Bayesian hidden-state (change-point) learner with softmax choice rule.

    Hidden state s in {0, 1}: which option is currently better.
    Observation model: the better option pays off w.p. p, the other w.p. 1 - p.
    Transition model : the state flips on each trial with probability h.

        posterior : b+ = b * L1 / (b * L1 + (1 - b) * L0)
        prediction: b' = b+ * (1 - h) + (1 - b+) * h
        value     : EV_1 - EV_0 = (2b - 1)(2p - 1)
        choice    : P(choose 1) = sigmoid(beta * (EV_1 - EV_0))

    The learner (i) updates its belief about *both* options from a single outcome
    (it knows they are anti-correlated) and (ii) has an effective learning rate that
    is large after surprising outcomes and small when outcomes confirm its belief.
    """

    name = "Bayes"
    param_names = ("h", "p", "beta")
    bounds = ((0.001, 0.4), (0.55, 0.99), (0.01, 20.0))
    start_box = ((0.01, 0.2), (0.6, 0.95), (1.0, 10.0))
    n_params = 3

    def sample_params(self, rng: np.random.Generator) -> np.ndarray:
        return np.array([rng.uniform(0.01, 0.2), rng.uniform(0.6, 0.95), rng.uniform(2.0, 15.0)])

    def log_prior(self, theta) -> float:
        h, p, beta = theta
        return (
            _beta_logpdf_unnorm(h / 0.4, 1.5, 4.0)
            + _beta_logpdf_unnorm((p - 0.5) / 0.5, 2.0, 2.0)
            + _gamma_logpdf_unnorm(beta, 1.2, 5.0)
        )

    @staticmethod
    def _likelihoods(c: int, r: int, p: float):
        """(L1, L0): P(r | choice c, state 1) and P(r | choice c, state 0)."""
        if c == 1:
            return (p, 1.0 - p) if r else (1.0 - p, p)
        return (1.0 - p, p) if r else (p, 1.0 - p)

    def trial_loglik(self, theta, choices, rewards) -> np.ndarray:
        h, p, beta = float(theta[0]), float(theta[1]), float(theta[2])
        k = beta * (2.0 * p - 1.0)
        b = 0.5
        out = np.empty(len(choices))
        for t in range(len(choices)):
            z = k * (2.0 * b - 1.0)
            c = choices[t]
            out[t] = _log_sigmoid(z if c == 1 else -z)
            l1, l0 = self._likelihoods(c, rewards[t], p)
            num = b * l1
            b_post = num / (num + (1.0 - b) * l0)
            b = b_post * (1.0 - h) + (1.0 - b_post) * h
        return out

    def simulate(self, theta, schedule: Schedule, rng: np.random.Generator):
        h, p, beta = float(theta[0]), float(theta[1]), float(theta[2])
        k = beta * (2.0 * p - 1.0)
        T = schedule.n_trials
        u, v = rng.random(T), rng.random(T)
        choices, rewards, p_chosen = np.empty(T, int), np.empty(T, int), np.empty(T)
        b = 0.5
        for t in range(T):
            p1 = 1.0 / (1.0 + math.exp(-k * (2.0 * b - 1.0)))
            c = 1 if u[t] < p1 else 0
            r = 1 if v[t] < schedule.reward_prob[t, c] else 0
            choices[t], rewards[t], p_chosen[t] = c, r, p1 if c == 1 else 1.0 - p1
            l1, l0 = self._likelihoods(c, r, p)
            num = b * l1
            b_post = num / (num + (1.0 - b) * l0)
            b = b_post * (1.0 - h) + (1.0 - b_post) * h
        return choices, rewards, p_chosen


class BayesianObserverFixedP:
    """Bayesian observer that is *told* the task's reward probability (p fixed).

    Same filter as ``BayesianObserver`` but p is fixed at the task's true ``p_high``,
    leaving only the reversal hazard h and the inverse temperature beta free
    (theta = (h, beta)).  With p free, p and beta trade off against each other
    (see README, "Identifiability"); fixing p gives a 2-parameter model with the same
    complexity as Rescorla-Wagner.
    """

    name = "Bayes-fixedP"
    param_names = ("h", "beta")
    bounds = ((0.001, 0.4), (0.01, 40.0))
    start_box = ((0.01, 0.2), (1.0, 15.0))
    n_params = 2

    def __init__(self, p_fixed: float = 0.8):
        self.p_fixed = p_fixed
        self._inner = BayesianObserver()

    def _full(self, theta):
        return (float(theta[0]), self.p_fixed, float(theta[1]))

    def sample_params(self, rng: np.random.Generator) -> np.ndarray:
        return np.array([rng.uniform(0.01, 0.2), rng.uniform(2.0, 15.0)])

    def log_prior(self, theta) -> float:
        h, beta = theta
        return _beta_logpdf_unnorm(h / 0.4, 1.5, 4.0) + _gamma_logpdf_unnorm(beta, 1.2, 8.0)

    def trial_loglik(self, theta, choices, rewards) -> np.ndarray:
        return self._inner.trial_loglik(self._full(theta), choices, rewards)

    def simulate(self, theta, schedule: Schedule, rng: np.random.Generator):
        return self._inner.simulate(self._full(theta), schedule, rng)


MODELS = {
    "RW": RescorlaWagner(),
    "Bayes": BayesianObserver(),
    "Bayes-fixedP": BayesianObserverFixedP(0.8),
}
