from __future__ import annotations
from typing import List, Dict, ClassVar, Optional
from pydantic.dataclasses import dataclass
from pydantic import Field
from pydantic import validator

@dataclass
class SplitBuilderParams:
    """Split experiment parameters class.

    Args:
        map_group_names_to_sizes(Dict): dictionary with group names and sizes. 
            Key with name "control" is obligatory
        main_strata_col (str): nthe name of the column to be used first for splitting
        split_metric_col (str): the name of the column to be binning data for splitting
        id_col (str): the name of the column with id 
        cols: columns for stratification data
        cat_cols: categorical columns that are using for stratification.
            These cols'll be encoded as category features
        n_bins: number of bins to be created based on split_metric_col
        min_cluster_size: min count of samples in HDBSCAN cluster
        strata_outliers_frac: frequency of outlyers in strata
        alpha: significance level for A/A test for split
    """
    min_unique_values_in_col: ClassVar[int] = 3
    control_group_name: ClassVar[str] = "control"
    map_group_names_to_sizes: Dict[str, Optional[int]]
    main_strata_col: str
    split_metric_col: str
    metric_type: str = 'continuous'  # continuous, binary, ratio
    id_col: str = "customer_id"
    cols: List[str] = Field(default_factory=list)
    cat_cols: List[str] = Field(default_factory=list)
    n_bins: int = 3
    min_cluster_size: int = 100 
    strata_outliers_frac: float = 0.01
    alpha: float = 0.05

    def __post_init_post_parse__(self):
        self.cols.extend([self.split_metric_col])
        self.cols = list(set(self.cols))

    @validator("alpha", always=True, allow_reuse=True)
    @classmethod
    def alpha_validator(cls, alpha: float):
        assert 0 < alpha < 1
        return alpha

    @validator("metric_type", always=True, allow_reuse=True)
    @classmethod
    def metric_type_validator(cls, metric_type: str) -> str:
        assert metric_type in ['continuous', 'binary', 'ratio'], "metric_type is not in ['continuous', 'binary', 'ratio']"
        return metric_type
