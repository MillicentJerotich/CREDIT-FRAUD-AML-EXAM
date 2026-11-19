import numpy as np

from fraud.evaluation import apply_policy, decision_cost
from fraud.monitoring import make_reference, psi


def test_decision_cost_worked_example():
    y = [1, 1, 0, 0, 1]
    flag = [True, False, True, False, False]
    amount = [200, 80, 30, 15, 2]
    # missed frauds 80 + 2, one false alarm at EUR 5
    assert decision_cost(y, flag, amount, c_fp=5.0) == 87.0


def test_amount_aware_rule_threshold():
    p = np.array([0.006, 0.006, 0.4, 0.6])
    amount = np.array([1000.0, 5.0, 5.0, 5.0])
    flags = apply_policy("Amount-aware Bayes rule", p, amount, 5.0, {})
    # 1000 EUR: threshold 5/1005 ~ 0.005 -> flag; 5 EUR: threshold 0.5
    assert flags.tolist() == [True, False, False, True]


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=20000)
    r = make_reference(ref)
    assert psi(r, rng.normal(size=20000)) < 0.01
    assert psi(r, rng.normal(loc=1.0, size=20000)) > 0.25
