from typing import Any, Callable, Dict, List, Tuple, Union
import numpy as np
import numpy.typing as npt
import pandas as pd


ArrayType = Union[List, npt.NDArray, pd.Series]
ArrayNumType = Union[List[float], npt.NDArray, pd.Series]
ArrayStrType = Union[List[str], npt.NDArray, pd.Series]

ColumnNameType = str
ColumnNamesType = ArrayStrType

MetricType = Callable[[ArrayNumType], float]
MetricTransformType = Callable[[np.ndarray], np.ndarray]
MetricFunctionType = Callable[[Any], Union[int, float]]
MetricTransformFunctionType = Callable[[ArrayNumType], ArrayNumType]

StatTestResultType = Dict[str, Union[int, float]]

DataFrameType = pd.DataFrame
