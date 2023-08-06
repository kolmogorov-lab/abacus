from pydantic.dataclasses import dataclass, Field


@dataclass
class BaseSplitElement():
    """Dataclass with data params for experiment calculations

    Args:
        group_sizes (tuple): tuple with group sizes. Should have control group size on the 0 index position 
            and target group size on the 1 index position
        split_number (int): params with number of split
    """
    group_sizes: tuple
    split_number: int
    control_group_size: int = Field(init=False)
    target_group_size: int = Field(init=False)
    def __post_init__(self):
        self.control_group_size = self.group_sizes[0]
        self.target_group_size = self.group_sizes[1]

@dataclass
class MdeAlphaExperiment(BaseSplitElement):
    """Dataclass for I type error calculations.
    """
    metric_name: str = Field(init=False)

@dataclass
class MdeBetaExperiment(MdeAlphaExperiment):
    """Dataclass for II type error calculations.
    """
    inject: float = Field(init=False)
