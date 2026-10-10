import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from revlearn import MODELS, TaskConfig, make_schedule, schedule_from_best  # noqa: E402
from revlearn.analysis import behavior_stats, reversal_curve  # noqa: E402
from revlearn.fitting import cross_validate, fit, sample_posterior  # noqa: E402


@pytest.fixture(scope="module")
def sched():
    return make_schedule(TaskConfig(), np.random.default_rng(0))


def test_schedule_structure(sched):
    assert sched.n_trials == 400
    assert set(np.unique(sched.best)) == {0, 1}
    np.testing.assert_allclose(sched.reward_prob.sum(axis=1), 1.0)
    assert len(sched.reversal_trials) >= 8
    # best option flips exactly at the reversal trials
    for t in sched.reversal_trials:
        assert sched.best[t] != sched.best[t - 1]


def test_schedule_from_best_roundtrip(sched):
    s2 = schedule_from_best(sched.best, 0.8)
    np.testing.assert_array_equal(s2.trial_in_block, sched.trial_in_block)


@pytest.mark.parametrize("name", list(MODELS))
def test_simulate_matches_likelihood(name, sched):
    """The simulator and the likelihood must implement the same model."""
    rng = np.random.default_rng(1)
    m = MODELS[name]
    for _ in range(5):
        theta = m.sample_params(rng)
        c, r, p_chosen = m.simulate(theta, sched, rng)
        np.testing.assert_allclose(np.log(p_chosen), m.trial_loglik(theta, c, r), atol=1e-9)


@pytest.mark.parametrize("name", list(MODELS))
def test_chance_level_when_beta_zero_like(name, sched):
    """With a vanishing inverse temperature the likelihood is log(0.5) per trial."""
    m = MODELS[name]
    theta = m.sample_params(np.random.default_rng(2))
    theta[-1] = 1e-9
    c, r, _ = m.simulate(theta, sched, np.random.default_rng(3))
    np.testing.assert_allclose(m.trial_loglik(theta, c, r), np.log(0.5), atol=1e-6)


def test_rw_delta_rule_hand_computation():
    m = MODELS["RW"]
    # alpha=0.5, beta=2.  Trial 0: Q=(0.5,0.5) -> p=0.5.  Choose 1, reward 1 -> Q1 = 0.75.
    # Trial 1: z = 2*(0.75-0.5) = 0.5 ; choose 1 -> log sigmoid(0.5)
    ll = m.trial_loglik([0.5, 2.0], np.array([1, 1]), np.array([1, 1]))
    assert ll[0] == pytest.approx(np.log(0.5))
    assert ll[1] == pytest.approx(-np.log1p(np.exp(-0.5)))


def test_bayes_belief_update_hand_computation():
    m = MODELS["Bayes"]
    h, p, beta = 0.1, 0.8, 5.0
    ll = m.trial_loglik([h, p, beta], np.array([1, 1]), np.array([1, 1]))
    # b0 = 0.5; choose 1, reward -> L1=0.8, L0=0.2 -> posterior 0.8; then b1 = 0.8*0.9 + 0.2*0.1 = 0.74
    z = beta * (2 * p - 1) * (2 * 0.74 - 1)
    assert ll[0] == pytest.approx(np.log(0.5))
    assert ll[1] == pytest.approx(-np.log1p(np.exp(-z)))


def test_fit_beats_or_matches_true_parameters(sched):
    rng = np.random.default_rng(4)
    for name in MODELS:
        m = MODELS[name]
        theta = m.sample_params(rng)
        c, r, _ = m.simulate(theta, sched, rng)
        res = fit(m, c, r, n_starts=6, rng=rng)
        assert res.loglik >= m.trial_loglik(theta, c, r).sum() - 1e-6   # optimiser finds a point at least as good


def test_rw_parameter_recovery_reasonable():
    rng = np.random.default_rng(5)
    m = MODELS["RW"]
    true, est = [], []
    for _ in range(15):
        s = make_schedule(TaskConfig(), rng)
        th = m.sample_params(rng)
        c, r, _ = m.simulate(th, s, rng)
        true.append(th); est.append(fit(m, c, r, n_starts=6, rng=rng).params)
    true, est = np.array(true), np.array(est)
    assert np.corrcoef(true[:, 0], est[:, 0])[0, 1] > 0.8     # alpha
    assert np.corrcoef(true[:, 1], est[:, 1])[0, 1] > 0.7     # beta


def test_cross_validation_returns_finite_negative_ll(sched):
    rng = np.random.default_rng(6)
    m = MODELS["RW"]
    c, r, _ = m.simulate([0.3, 5.0], sched, rng)
    cv = cross_validate(m, c, r, k_folds=4, n_starts=3, rng=rng)
    assert np.isfinite(cv) and -np.log(2) - 0.2 < cv < 0


def test_mcmc_recovers_neighbourhood_of_truth(sched):
    rng = np.random.default_rng(7)
    m = MODELS["RW"]
    c, r, _ = m.simulate([0.4, 6.0], sched, rng)
    res = fit(m, c, r, n_starts=6, rng=rng)
    samples, acc = sample_posterior(m, c, r, res.params, n_warmup=300, n_samples=500, rng=rng)
    assert 0.05 < acc < 0.9
    assert abs(samples[:, 0].mean() - 0.4) < 0.3


def test_stats_and_curve_shapes(sched):
    rng = np.random.default_rng(8)
    c, r, _ = MODELS["RW"].simulate([0.4, 6.0], sched, rng)
    st = behavior_stats(c, r, sched)
    assert 0 <= st["acc"] <= 1 and 0 <= st["win_stay"] <= 1
    assert reversal_curve(c, sched).shape == (25,)
