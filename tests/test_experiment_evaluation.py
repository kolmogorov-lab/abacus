from collections import Counter
import numpy as np
import pandas as pd

import pytest

from abacus.auto_ab.params import ABTestParams, DataParams, HypothesisParams
from abacus.auto_ab import ABTest


@pytest.fixture()
def checks_dataframe() -> pd.DataFrame:
    """
    DataFrame with checks and sessions information
    """
    df = pd.read_csv("./tests/data/ab_data_check.csv")

    return df


@pytest.fixture()
def continuous_test(checks_dataframe) -> ABTest:
    """
    Creation of simple continuous experiment
    """
    data_params = DataParams(
        id_col="user_id",
        group_col="groups",
        control_name="control",
        treatment_name="treatment",
        target="check_rub_campaign",
    )
    hypothesis_params = HypothesisParams()

    ab_params = ABTestParams(
        data_params=data_params, hypothesis_params=hypothesis_params
    )
    ab_test = ABTest(checks_dataframe, ab_params)

    return ab_test


@pytest.fixture()
def binary_test(checks_dataframe) -> ABTest:
    """
    Creation of simple binary experiment
    """
    data_params = DataParams(
        id_col="user_id",
        group_col="groups",
        control_name="control",
        treatment_name="treatment",
        target="has_transaction"
    )
    hypothesis_params = HypothesisParams(metric_type="binary")

    ab_params = ABTestParams(
        data_params=data_params, hypothesis_params=hypothesis_params
    )
    ab_test = ABTest(checks_dataframe, ab_params)

    return ab_test


@pytest.fixture()
def binary_test_balanced(checks_dataframe) -> ABTest:
    """
    Creation of simple binary experiment with balanced groups
    """
    min_group_size = min(dict(Counter(checks_dataframe["groups"])).values())
    control = (
        checks_dataframe[checks_dataframe["groups"] == "control"]
        .sample(n=min_group_size)
        .reset_index(drop=True)
    )
    treatment = (
        checks_dataframe[checks_dataframe["groups"] == "treatment"]
        .sample(n=min_group_size)
        .reset_index(drop=True)
    )
    checks_dataframe_balanced = pd.concat([control, treatment], axis=0)

    data_params = DataParams(
        id_col="user_id",
        group_col="groups",
        control_name="control",
        treatment_name="treatment",
        target="has_transaction"
    )
    hypothesis_params = HypothesisParams(metric_type="binary")

    ab_params = ABTestParams(
        data_params=data_params, hypothesis_params=hypothesis_params
    )
    ab_test = ABTest(checks_dataframe_balanced, ab_params)

    return ab_test


@pytest.fixture()
def ratio_test(checks_dataframe) -> ABTest:
    """
    Creation of simple ratio experiment
    """
    data_params = DataParams(
        id_col="user_id",
        group_col="groups",
        control_name="control",
        treatment_name="treatment",
        numerator="clicks",
        denominator="session_duration",
    )
    hypothesis_params = HypothesisParams(metric_type="ratio")

    ab_params = ABTestParams(
        data_params=data_params, hypothesis_params=hypothesis_params
    )
    ab_test = ABTest(checks_dataframe, ab_params)

    return ab_test


def test_evaluation_boot_confint_test_continuous(continuous_test):
    """
    Bootstrap confidence interval test evaluation
    """
    np.random.seed(42)
    test_result = continuous_test.test_boot_confint()
    true_test_result = {"stat": None, "p-value": 0, "result": 1}

    assert true_test_result == test_result


def test_evaluation_boot_confint_test_binary(binary_test):
    """
    Bootstrap confidence interval test evaluation
    """
    np.random.seed(42)
    test_result = binary_test.test_boot_confint()
    true_test_result = {"stat": None, "p-value": 0, "result": 1}

    assert true_test_result == test_result


def test_evaluation_boot_fp_test_continuous(continuous_test):
    """
    Bootstrap test evaluation using false positives
    """
    np.random.seed(42)
    test_result = continuous_test.test_boot_fp()
    true_test_result = {"stat": None, "p-value": 0, "result": 1}

    assert true_test_result == test_result


def test_evaluation_boot_fp_test_binary(binary_test):
    """
    Bootstrap test evaluation using false positives
    """
    np.random.seed(42)
    test_result = binary_test.test_boot_fp()
    true_test_result = {"stat": None, "p-value": 0, "result": 1}

    assert true_test_result == test_result


def test_evaluation_boot_ratio_test(ratio_test):
    """
    Bootstrap test evaluation for ratio metric
    """
    np.random.seed(42)
    test_result = ratio_test.test_boot_ratio()
    true_test_result = {"stat": None, "p-value": None, "result": 0}

    assert true_test_result == test_result


def test_evaluation_boot_welch_test(continuous_test):
    """
    Bootstrap test evaluation using Welch's t-test
    """
    np.random.seed(42)
    test_result = continuous_test.test_boot_welch()
    true_test_result = {"stat": None, "p-value": 0, "result": 1}

    assert true_test_result == test_result


def test_evaluation_buckets_test(continuous_test):
    """
    Welch's t-test on buckets of original data
    """
    np.random.seed(42)
    test_result = continuous_test.test_buckets()
    true_test_result = {"stat": None, "p-value": 0, "result": 1}

    assert true_test_result == test_result


def test_evaluation_chisquare_test(binary_test):
    """
    Chi-square test on imbalanced data: number of observations between groups is different
    """
    with pytest.raises(ValueError):
        binary_test.test_chisquare()


def test_evaluation_chisquare_test_balanced(binary_test_balanced):
    """
    Chi-square test on balanced data: number of observations between groups is equal
    """
    test_result = binary_test_balanced.test_chisquare()
    true_test_result = {"stat": 1859.09821, "p-value": 0, "result": 1}

    assert true_test_result == test_result


def test_evaluation_delta_ratio_test(ratio_test):
    """
    Delta method for evaluation of mean and variance of ratio metric and Welch's t-test for evaluation
    """
    test_result = ratio_test.test_delta_ratio()
    true_test_result = {"stat": 0.67055, "p-value": None, "result": 0}

    assert true_test_result == test_result


def test_evaluation_mannwhitney_test(continuous_test):
    """
    Mann-Whitney's U-test
    """
    test_result = continuous_test.test_mannwhitney()
    true_test_result = {
        "stat": 4929788667,
        "p-value": 0,
        "result": 1,
    }

    assert true_test_result == test_result


def test_evaluation_taylor_ratio_test(ratio_test):
    """
    Taylor series expansion for evaluation of mean and variance of ratio metric and Welch's t-test for evaluation
    """
    test_result = ratio_test.test_taylor_ratio()
    true_test_result = {"stat": -20.06346, "p-value": None, "result": 1}

    assert true_test_result == test_result


def test_evaluation_welch_test(continuous_test):
    """
    Welch's t-test
    """
    test_result = continuous_test.test_welch()
    true_test_result = {
        "stat": 4.78451,
        "p-value": 0,
        "result": 1,
    }

    assert true_test_result == test_result
