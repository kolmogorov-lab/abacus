from __future__ import annotations
from typing import Union
from pydantic.dataclasses import dataclass


class ValidationConfig:
    validate_assignment = True
    arbitrary_types_allowed = True

@dataclass
class GroupNames():
    test_group_name: Union[str, int] = 'test'
    control_group_name: Union[str, int] = 'control'


@dataclass(config=ValidationConfig)
class ResplitParams:
    """Resplit params class.

    Args:
        group_names (GroupNames): group names
        strata_col (str): name of column with strata
        group_col (str): name of column with groups split
    """
    group_names: GroupNames
    strata_col: str = 'strata'
    group_col: str = 'group_col'
