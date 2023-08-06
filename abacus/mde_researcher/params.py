from typing import List, Any, Dict, Optional, Callable, Union
from pydantic import root_validator, validator
from pydantic.dataclasses import dataclass
from abacus.auto_ab.abtest import ABTest
from fastcore.transform import Pipeline


class ValidationConfig:
    validate_assignment = True
    arbitrary_types_allowed = True


@dataclass(config=ValidationConfig)
class MdeParams:
    """MDE research experiment parameters class.

    Args:
        metrics_names: Metrics which will be compare in experiments.
        injects: Injects represent MDE values.
        min_group_size: Minimal value of groups sizes.
        max_group_size: Maximal value of groups sizes.
        step: Spacing between min_group_size and max_group_size.
        variance_reduction: ABTest methods for variance reduction.
        use_buckets: Use bucketize method.
        transformations: Pipeline of experiment. Will be calulted in __post_init__.
        stat_test: Statistical test type.
        iterations_number: Count of splits for each element in group_sizes.
        max_beta_score: Maximum level of II type error.
        min_beta_score: Minimum level of II type error.
    """
    metrics_names: List[str]
    injects: List[float]
    min_group_size: int
    max_group_size: int
    step: int
    variance_reduction: Optional[Callable[[ABTest], ABTest]] = None
    use_buckets: bool = False
    transformations: Any = None
    stat_test: Callable[[ABTest], Dict[str, Union[int, float] ]] = ABTest.test_boot_confint
    iterations_number: int = 10
    max_beta_score: float = 0.2
    min_beta_score: float = 0.05

    def __post_init__(self):
        if self.use_buckets:
            transformations = [self.variance_reduction, ABTest.bucketing]
        else:
            transformations = [self.variance_reduction]
        transformations = list(filter(None, transformations))
        self.transformations = Pipeline(transformations)

    @validator("variance_reduction", always=True)
    @classmethod
    def variance_reduction_validator(cls, variance_reduction):
        assert variance_reduction in [ABTest.cuped, ABTest.cupac, None], \
                            'variance reduction algorithm is not in list of allowed ones'
        return variance_reduction

    @validator("stat_test", always=True)
    @classmethod
    def stat_test_validator(cls, stat_test):
        assert stat_test in [ABTest.test_boot_confint,
                             ABTest.test_boot_fp,
                             ABTest.test_boot_welch,
                             ABTest.test_boot_ratio,
                             ABTest.test_mannwhitney,
                             ABTest.test_welch,
                             ABTest.test_delta_ratio,
                             ABTest.test_taylor_ratio,
                             ABTest.test_z_proportions,
                            ], 'stat test is not in list of allowed tests'
        return stat_test

    @root_validator
    @classmethod
    def groups_sizes_validator(cls, values):
        min_group_size = values.get("min_group_size")
        max_group_size = values.get("max_group_size")
        assert max_group_size >= min_group_size, \
            "max_group_size should be more than min_group_size"
        return values
