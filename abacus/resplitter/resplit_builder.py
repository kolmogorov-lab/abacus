import pandas as pd
from abacus.resplitter.params import ResplitParams


class ResplitBuilder():
    """Builds stratification split for DataFrame.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 resplit_params: ResplitParams):
        """Makes restratification of dataframe 

        Args:
            df (pandas.DataFrame): DataFrame for rebuild split.
            resplit_params (ResplitParams): Params for resplit.
        """
        self.df = df
        self.params = resplit_params
        self._len_df = len(df)

    def collect(self) -> pd.DataFrame:
        """Method recalculate fractions of each strata in dataframe.

        Returns:
            pandas.DataFrame: DataFrame with recalculated strata fractions.
        """
        df_restrata = pd.DataFrame()

        strata_all_count = (self.df.value_counts(self.params.strata_col)
                            .reset_index(name='all_count')
        )
        strata_test_counts = (self.df
                            .value_counts([self.params.strata_col, self.params.group_col])
                            .reset_index(name = 'count')
        )   
        strata_test_counts = (strata_test_counts
                            .merge(strata_all_count,
                                    on = self.params.strata_col, 
                                    how = 'inner')
        )
        strata_test_counts['frequency'] = strata_test_counts['count'] / strata_test_counts['all_count']

        min_group_freq = (strata_test_counts
                    .groupby(self.params.strata_col)
                    .min()['frequency'].reset_index(name='min_freq')
        )
        strata_test_counts = strata_test_counts.merge(min_group_freq, 
                                                        on = self.params.strata_col
                                                    )
        strata_test_counts['disired_count'] = round(strata_test_counts['all_count'] 
                                                    * strata_test_counts['min_freq'], 0
                                                )
        strata_test_counts['disired_count'] = strata_test_counts['disired_count'].astype(int)

        group_names = [self.params.group_names.test_group_name, self.params.group_names.control_group_name]

        for group_name in group_names:  
            for strata in self.df[self.params.strata_col].unique():
                strata_group_count = (strata_test_counts[
                                            (strata_test_counts[self.params.group_col] == group_name) &
                                            (strata_test_counts[self.params.strata_col] == strata)
                    ]['disired_count']
                ).values[0]

                strata_group_values = (self.df[
                                        (self.df[self.params.group_col] == group_name) &
                                        (self.df[self.params.strata_col] == strata)]
                )
                df_strata = strata_group_values.sample(n=strata_group_count)
                df_restrata = pd.concat([df_restrata, df_strata])

        return df_restrata
