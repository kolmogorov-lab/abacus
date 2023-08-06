from typing import List
import pandas as pd
import statsmodels.api as sm
from category_encoders.target_encoder import TargetEncoder
from abacus.types import ArrayNumType, ColumnNameType, ColumnNamesType, DataFrameType


class VarianceReduction:
    """Implementation of sensitivity increasing approaches.

    As it is easier to apply variance reduction techniques directrly to experiment, all approaches should be called on ``ABTest`` class instance.

    Example:

    .. code-block:: python

        from abacus.auto_ab.abtest import ABTest
        from abacus.auto_ab.params import ABTestParams, DataParams, HypothesisParams

        data_params = DataParams(...)
        hypothesis_params = HypothesisParams(...)
        ab_params = ABTestParams(data_params, hypothesis_params)

        df = pd.read_csv('data.csv')
        ab_test = ABTest(df, ab_params)
        ab_test = ab_test.cuped()
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _target_encoding(x: DataFrameType,
                         encoding_columns: ColumnNamesType,
                         target_column: str
                         ) -> DataFrameType:
        """Encodes target column.
        """
        for col in x[encoding_columns].select_dtypes(include='O').columns:
            te = TargetEncoder()
            x[col] = te.fit_transform(x[col], x[target_column])
        return x

    @staticmethod
    def _predict_target(x: DataFrameType,
                        target_prev_col: ColumnNameType,
                        factors_prev_cols: ColumnNamesType,
                        factors_now_cols: ColumnNamesType
                        ) -> ArrayNumType:
        """Covariate prediction with linear regression model.

        Args:
            x (pandas.DataFrame): Pandas DataFrame.
            target_prev (str): Target on previous period column name.
            factors_prev (List[str]): Factor columns for modelling.
            factors_now (List[str]): Factor columns for prediction on current period.

        Returns:
            pandas.Series: Pandas Series with predicted values
        """
        y = x[target_prev_col]
        x_train = x[factors_prev_cols]
        model = sm.OLS(y, x_train)
        results = model.fit()

        print(results.summary())
        x_predict = x[factors_now_cols]

        return model.predict(x_predict)

    @classmethod
    def cupac(cls,
              x: DataFrameType,
              target_prev_col: ColumnNameType,
              target_now_col: ColumnNameType,
              factors_prev_cols: ColumnNamesType,
              factors_now_cols: ColumnNamesType,
              groups_col: ColumnNameType
              ) -> DataFrameType:
        """ Perform CUPED on target variable with covariate calculated
        as a prediction from a linear regression model.

        Original paper: https://doordash.engineering/2020/06/08/improving-experimental-power-through-control-using-predictions-as-covariate-cupac/.

        Args:
            x (pandas.DataFrame): Pandas DataFrame for analysis.
            target_prev (str): Target on previous period column name.
            target_now (str): Target on current period column name.
            factors_prev (List[str]): Factor columns for modelling.
            factors_now (List[str]): Factor columns for prediction on current period.
            groups (str): Groups column name.

        Returns:
            pandas.DataFrame: Pandas DataFrame with additional columns: target_pred and target_now_cuped
        """
        x = cls._target_encoding(x, list(set(factors_prev_cols + factors_now_cols)), target_prev_col)
        x.loc[:, 'target_pred'] = cls._predict_target(x, target_prev_col, factors_prev_cols, factors_now_cols)
        x_new = cls.cuped(x, target_now_col, groups_col, 'target_pred')
        return x_new

    @classmethod
    def cuped(cls,
              df: DataFrameType,
              target_col: ColumnNameType,
              groups_col: ColumnNameType,
              covariate_col: ColumnNameType
              ) -> DataFrameType:
        """ Perform CUPED on target variable with predefined covariate.

        Covariate has to be chosen with regard to the following restrictions:

        1. Covariate is independent of an experiment.
        2. Covariate is highly correlated with target variable.
        3. Covariate is continuous variable.

        Original paper: https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf.

        Args:
            df (pandas.DataFrame): Pandas DataFrame for analysis.
            target (str): Target column name.
            groups (str): Groups A and B column name.
            covariate (str): Covariate column name. If None, then most correlated column in considered as covariate.

        Returns:
            pandas.DataFrame: Pandas DataFrame with additional target CUPEDed column
        """
        x = df.copy()

        cov = x[[target_col, covariate_col]].cov().loc[target_col, covariate_col]
        var = x[covariate_col].var()
        theta = cov / var

        for group in x[groups_col].unique():
            x_subdf = x[x[groups_col] == group]
            group_y_cuped = x_subdf[target_col] - theta * (x_subdf[covariate_col] - x_subdf[covariate_col].mean())
            x.loc[x[groups_col] == group, target_col] = group_y_cuped

        return x
