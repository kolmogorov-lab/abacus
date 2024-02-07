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


@pytest.mark.smoke
def test_experiment_init_default(checks_dataframe):
    """
    Creation of experiment with default parameters
    """
    data_params = DataParams(
        id_col="user_id", group_col="groups", target="check_rub_campaign"
    )
    hypothesis_params = HypothesisParams()

    ab_params = ABTestParams(
        data_params=data_params, hypothesis_params=hypothesis_params
    )
    ab_test = ABTest(checks_dataframe, ab_params)


@pytest.mark.smoke
def test_experiment_init_continuous(checks_dataframe):
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
    hypothesis_params = HypothesisParams(metric_type="continuous")

    ab_params = ABTestParams(
        data_params=data_params, hypothesis_params=hypothesis_params
    )
    ab_test = ABTest(checks_dataframe, ab_params)


@pytest.mark.smoke
def test_experiment_init_binary(checks_dataframe):
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


@pytest.mark.smoke
def test_experiment_init_ratio(checks_dataframe):
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


@pytest.mark.smoke
def test_experiment_init_continuous_custom_metric(checks_dataframe):
    """
    Creation of continuous experiment with custom metric
    """

    def quantile_95(x: np.ndarray) -> float:
        return np.quantile(x, 0.95)

    data_params = DataParams(
        id_col="user_id",
        group_col="groups",
        control_name="control",
        treatment_name="treatment",
        target="check_rub_campaign",
    )
    hypothesis_params = HypothesisParams(
        metric_type="continuous", metric_name="95th quantile", metric=quantile_95
    )

    ab_params = ABTestParams(
        data_params=data_params, hypothesis_params=hypothesis_params
    )
    ab_test = ABTest(checks_dataframe, ab_params)


@pytest.mark.smoke
def test_experiment_init_transformations(checks_dataframe):
    """
    Creation of experiment with metric transformations
    """
    data_params = DataParams(
        id_col="user_id",
        group_col="groups",
        control_name="control",
        treatment_name="treatment",
        target="check_rub_campaign",
        transforms=["bucketing"],
    )
    hypothesis_params = HypothesisParams()

    ab_params = ABTestParams(
        data_params=data_params, hypothesis_params=hypothesis_params
    )
    ab_test = ABTest(checks_dataframe, ab_params)


@pytest.mark.smoke
def test_experiment_init_cuped(checks_dataframe):
    """
    Creation of experiment with variance reduction as CUPED
    """
    data_params = DataParams(
        id_col="user_id",
        group_col="groups",
        control_name="control",
        treatment_name="treatment",
        target="check_rub_campaign",
        covariate="check_rub_pre_campaign",
    )
    hypothesis_params = HypothesisParams()

    ab_params = ABTestParams(
        data_params=data_params, hypothesis_params=hypothesis_params
    )
    ab_test = ABTest(checks_dataframe, ab_params)
