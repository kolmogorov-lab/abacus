from __future__ import annotations
from typing import List, Dict, Any, Callable, Optional, Iterable, Union

from pydantic.dataclasses import dataclass
from pydantic import validator, Field
import numpy as np

from abacus.types import MetricFunctionType, ColumnNameType, ColumnNamesType, \
                        ArrayNumType, ArrayStrType, MetricType, MetricTransformType

class ValidationConfig:
    validate_assignment = True
    arbitrary_types_allowed = True

@dataclass(config=ValidationConfig)
class DataParams:
    """Data description as column names of dataset generated during experiment.

    Parameters:
        id_col (str): ID of observations.
        group_col (str): Group of experiment.
        control_name (str): Name of control group in ``group_col``.
        treatment_name (str): Name of treatment group in ``group_col``.
        is_grouped (bool, Optional): Flag that shows whether observations are grouped.
        strata_col (str, Optional): Name of stratification column. Stratification column must be categorical.
        target (str, Optional): Target column name of continuous metric.
        target_flg (str, Optional): Target flag column name of binary metric.
        numerator (str, Optional): Numerator for ratio metric.
        denominator (str, Optional): Denominator for ratio metric.
        covariate (str, Optional): Covariate column for CUPED.
        target_prev (str, Optional): Target column name for previous period of continuous metric.
        predictors_now (List[str], Optional): List of columns to predict covariate.
        predictors_prev (List[str], Optional): List of columns to create linear model for covariate prediction.
        control (ArrayNumType, Optional): Control group data used for quick access and excluding querying dataset.
        treatment (ArrayNumType, Optional): Treatment group data used for quick access and excluding querying dataset.
        transforms (ArrayStrType, Optional): List of transformations applied to experiment.
    """
    id_col: ColumnNameType = 'id'
    group_col: ColumnNameType = 'groups'
    control_name: str = 'A'
    treatment_name: str = 'B'
    is_grouped: Optional[bool] = True
    strata_col: Optional[ColumnNameType] = ''
    target: Optional[ColumnNameType] = ''
    target_flg: Optional[ColumnNameType] = ''
    numerator: Optional[ColumnNameType] = ''
    denominator: Optional[ColumnNameType] = ''
    covariate: Optional[ColumnNameType] = ''
    target_prev: Optional[ColumnNameType] = ''
    predictors_now: Optional[ColumnNamesType] = Field(default_factory=list)
    predictors_prev: Optional[ColumnNamesType] = Field(default_factory=list)
    control: Optional[ArrayNumType] = Field(default_factory=list)
    treatment: Optional[ArrayNumType] = Field(default_factory=list)
    transforms: Optional[ArrayStrType] = Field(default_factory=list)

@dataclass(config=ValidationConfig)
class HypothesisParams:
    """Description of hypothesis parameters.

    Parameters:
        alpha (float): type I error.
        beta (float): type II error.
        alternative (str): directionality of hypothesis: less, greater, two-sided.
        metric_type (str): metric type: continuous, binary, ratio.
        metric_name (str): metric name: mean, median. If custom metric, then use here appropriate name.
        metric (Callable[[Iterable[float]], np.ndarray], Optional): if metric_name is custom, then you must define metric function.
        metric_transform (Callable[[np.ndarray], np.ndarray], Optional): applied transformations to experiment.
        metric_transform_info (Dict[str, Dict[str, Any]], Optional): information of applied transformations.
        filter_method (str, Optional): method for filtering outliers: top_5, isolation_forest.
        n_boot_samples (int, Optional): number of bootstrap iterations.
        n_buckets (int, Optional): number of buckets.
        strata (str, Optional): stratification column.
        strata_weights (Dict[str, float], Optional): historical strata weights.
    """
    alpha: Optional[float] = 0.05
    beta: Optional[float] = 0.2
    alternative: Optional[str] = 'two-sided'  # less, greater, two-sided
    metric_type: Optional[str] = 'continuous'  # continuous, binary, ratio
    metric_name: Optional[str] = 'mean'  # mean, median
    metric: Optional[MetricType] = np.mean
    metric_transform: Optional[MetricTransformType] = None
    metric_transform_info: Optional[Dict[str, Dict[str, Any]]] = None
    filter_method: Optional[str] = 'top_5'  # top_5, isolation_forest
    n_boot_samples: Optional[int] = 200
    n_buckets: Optional[int] = 100
    strata: Optional[str] = ''
    strata_weights: Optional[Dict[str, float]] = Field(default_factory=dict)

    def __post_init__(self):
        if self.metric_name == 'mean':
            self.metric = np.mean
        if self.metric_name == 'median':
            self.metric = np.median

    @validator("alpha", always=True, allow_reuse=True)
    @classmethod
    def alpha_validator(cls, alpha: float) -> float:
        assert 1 > alpha > 0, 'alpha is not in range [0, 1]'
        return alpha

    @validator("beta", always=True, allow_reuse=True)
    @classmethod
    def beta_validator(cls, beta: float) -> float:
        assert 1 > beta > 0, 'beta is not in range [0, 1]'
        return beta

    @validator("alternative", always=True, allow_reuse=True)
    @classmethod
    def alternative_validator(cls, alternative: str) -> str:
        assert alternative in ['two-sided', 'less', 'greater'], "alternative is not in ['two-sided', 'less', 'greater']"
        return alternative

    @validator("metric_type", always=True, allow_reuse=True)
    @classmethod
    def metric_type_validator(cls, metric_type: str) -> str:
        assert metric_type in ['continuous', 'binary', 'ratio'], "metric_type is not in ['continuous', 'binary', 'ratio']"
        return metric_type

@dataclass
class ABTestParams:
    data_params: DataParams = Field(default=DataParams())
    hypothesis_params: HypothesisParams = Field(default=HypothesisParams())
