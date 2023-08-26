import sys
import logging
from typing import List
import pandas as pd
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline as Pipe
from fastcore.transform import Pipeline
from abacus.splitter.params import SplitBuilderParams
from abacus.auto_ab.abtest import ABTest
from abacus.auto_ab.params import ABTestParams, DataParams, HypothesisParams
pd.options.mode.chained_assignment = None

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class SplitBuilder:
    """Builds stratification split for DataFrame.
    """

    def __init__(self,
                 split_data: pd.DataFrame, 
                 params: SplitBuilderParams):
        """Builds stratification split for DataFrame.

        Args:
            split_data (pandas.DataFrame): dataframe with data building split
            params (SplitBuilderParams): params for stratification and spilt
        """
        self.split_data = split_data.reset_index(drop=True)
        self.params = params

    def _prepare_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """This function converts given categorical features into features suitable for clustering and
        stratification. This functionality is achieved by adding two new features for each categorical
        feature:

        first feature (_encoded): there are two cases:
            (1) If the number of unique values of feature more than the value "min_cluster_size" from config,
                then values with low frequency will be combined into one ("other" with code DEFAULT_CAT_VALUE).
                After encoding feature will contain (min_cluster_size + 1) unique values;
            (2) If the number of unique values of feature less than the value "min_cluster_size" from config,
                the new column will be the same as the original;

        second feature (_freq): frequency with noise of the encoded feature.

        Return:
            pandas.DataFrame: DataFrame with extra columns;
        """
        df_cat = df.copy()
        for col in self.params.cat_cols:
            counts = df[col].value_counts()
            counts.iloc[:self.params.min_cluster_size] = (
                counts.iloc[:self.params.min_cluster_size] 
                + 0.1 * (np.random.uniform(low=0., 
                                            high=1., 
                                            size=len(counts.iloc[:self.params.min_cluster_size])) 
                        )
            )
            counts.iloc[self.params.min_cluster_size:] = sys.maxsize
            counts = counts.to_dict()
            df_cat[col] = (df_cat[col]
                          .map(lambda x, counts=counts: counts[x] / self.split_data.shape[0])
            )
        return df_cat

    def _binnarize(self, df: pd.DataFrame) -> pd.DataFrame:
        stratas_freq = (df[self.params.main_strata_col].value_counts()/len(df))
        stratas_freq = stratas_freq[stratas_freq >= self.params.strata_outliers_frac].index

        clean_df = df[df[self.params.main_strata_col].isin(stratas_freq)]
        main_strata = (clean_df[self.params.main_strata_col].astype(str) +
                    clean_df.groupby(self.params.main_strata_col, group_keys=False)[self.params.split_metric_col]
                        .apply(lambda x: pd.qcut(x,
                                                self.params.n_bins,
                                                labels=range(self.params.n_bins))
                                                ).astype(str)
                    )

        clean_df = clean_df.assign(strata=main_strata)
        if len(self.params.cols)>0:
            additional_strata = (clean_df.groupby("strata", as_index=False)
                                .apply(lambda group: self._clusterize(df=group,
                                                            strata="strata",
                                                            columns=self.params.cols,
                                                            min_cluster_size=self.params.min_cluster_size))
                                ).astype(str).droplevel(0)
            clean_df = clean_df.assign(strata=additional_strata)

        return df.assign(strata=clean_df.strata).fillna("-1")

    @staticmethod
    def _clusterize(
            df: pd.DataFrame, 
            strata: str, 
            columns:list, 
            min_cluster_size:int
        ) -> pd.Series:
        scaler = MinMaxScaler()
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
        pipe = Pipe(steps=[("scaler",scaler),
                    ("clusterer",clusterer)])
        pipe.fit(df[columns])
        labels = pipe['clusterer'].labels_.astype(str)

        return df[strata].astype(str) + labels

    def _assign_strata(self, df) -> pd.DataFrame:
        """Assigns strata for rows.

        Returns:
            pandas.DataFrame: DataFrame with strata columns.
        """
        transform = [self._prepare_categorical, self._binnarize]
        pipeline = Pipeline(transform)
        stratified_data = pipeline(df)
        return stratified_data

    def _map_stratified_samples(self, split_df:pd.DataFrame) -> pd.DataFrame:
        if all(x is None for x in self.params.map_group_names_to_sizes.values()):
            (self.params.map_group_names_to_sizes
                .update((key,len(split_df)//len(self.params.map_group_names_to_sizes)) 
                        for key in self.params.map_group_names_to_sizes
                        )
            )
    
        group_map = pd.DataFrame(columns=[self.params.id_col, "group_name"])
        for group_name, group_size in self.params.map_group_names_to_sizes.items():
            available_id = (split_df.loc[~split_df[self.params.id_col]
                                .isin(group_map[self.params.id_col].values)]
                                .copy()
            )
            
            try:
                group_frac_to_take = min(group_size / len(available_id), 1)
            except ZeroDivisionError:
                group_frac_to_take = 1
                
            groups = (
                available_id
                .groupby("strata", group_keys=False)
                .apply(lambda x, frac=group_frac_to_take: x.sample(frac=frac))[self.params.id_col]
                .to_frame().reset_index(drop=True)
            )
            groups["group_name"] = group_name
            group_map = pd.concat([group_map, groups])

        split_df = split_df.merge(group_map, 
                              on = self.params.id_col,
                              how = "inner"
        )
        return split_df

    def _check_groups(self, 
                    df_with_groups: pd.DataFrame,
                    control_name: str,
                    target_groups_names: List[str],
                    metric_type: str):
        tests_results = {}
        check_flag = 1
        for group in target_groups_names:
            hypothesis_params = HypothesisParams(alpha=self.params.alpha)
            for column in self.params.cols + self.params.cat_cols :
                data_params = DataParams(group_col="group_name",
                                    id_col=self.params.id_col,
                                    control_name=control_name,
                                    treatment_name=group,
                                    target=column
                )
                ab_params = ABTestParams(data_params, hypothesis_params)
                ab_test = ABTest(df_with_groups, ab_params)

                if metric_type == 'continuous':
                    test_result = ab_test.test_welch()
                elif metric_type == 'binary':
                    test_result = ab_test.test_z_proportions()
                tests_results[column] = test_result['p-value'].round(4)

            result = pd.DataFrame(tests_results, index=["1"])
            if(result < hypothesis_params.alpha).any().any():
                check_flag = 0
                log.error(f"Could not split statistically {group} and control")
                return check_flag
        return check_flag


    def _build_split(self, df_with_strata_col: pd.DataFrame) -> pd.DataFrame:
        """Builds stratified split.

        Args:
            df_with_strata_col (pandas.DataFrame): DataFrame with strata column.

        Returns:
            pandas.DataFrame: DataFrame with split.
        """
        max_attempts = 50
        for _ in range(max_attempts):
            groups_maped = self._map_stratified_samples(df_with_strata_col)
            target_groups = groups_maped["group_name"].unique().tolist()
            target_groups.remove(SplitBuilderParams.control_group_name)
            check_flag = self._check_groups(groups_maped,
                                            SplitBuilderParams.control_group_name,
                                            target_groups,
                                            SplitBuilderParams.metric_type)

            if check_flag:
                return groups_maped

        log.error("Split failed!")
        return df_with_strata_col
    

    def collect(self) -> pd.DataFrame:
        """Calculated splits for init dataframe
.
        Returns:
            pandas.DataFrame: DataFrame with split.
        """
        if len(self.split_data) == 0:
            return self.split_data

        transform = [self._assign_strata, self._build_split]
        pipeline = Pipeline(transform)
        return pipeline(self.split_data)
