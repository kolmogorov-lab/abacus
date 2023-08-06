from typing import Union
import itertools
import numpy as np
import pandas as pd
from abacus.splitter.params import SplitBuilderParams
from abacus.mde_researcher._experiment_structures import MdeAlphaExperiment, MdeBetaExperiment
from abacus.mde_researcher._abstract_mde_experiment_builder import AbstractMdeResearchBuilder
from abacus.mde_researcher.multiple_split_builder import MultipleSplitBuilder
from abacus.mde_researcher.params import MdeParams
from abacus.auto_ab.abtest import ABTest
from abacus.auto_ab.params import ABTestParams


class MdeResearchBuilder(AbstractMdeResearchBuilder):
    """Calculates I and II type errors for different group sizes and injects.
    """

    def __init__(self,
                 guests: pd.DataFrame,
                 abtest_params: ABTestParams,
                 experiment_params: MdeParams,
                 stratification_params: SplitBuilderParams):
        """
        Args:
            guests (pandas.DataFrame): Dataframe that collected by PrepilotGuestsCollector.
            abtest_params (ABTestParams): A/B tests params. Using for experiments calculations.
            experiment_params (MdeParams): Parameters for prepilot experiments.
            stratification_params (SplitBuilderParams): Params for groups splits and stratifications.
        """
        super().__init__(guests, abtest_params, experiment_params)
        self.stratification_params = stratification_params
        self._number_of_decimals = 10

    def _calc_experiment_grid_cell(self,
                                   guests_with_splits: pd.DataFrame,
                                   grid_element: Union[MdeBetaExperiment, MdeAlphaExperiment]
    ) -> pd.DataFrame:
        """Calculates stat test for one element of experiment grid.

        Args:
            guests_with_splits (pandas.DataFrame): DataFrame with calculated splits for experiment.
            grid_element (Union[MdeBetaExperiment, MdeAlphaExperiment]): Experiment params.

        Returns:
            pandas.DataFrame: Pandas DataFrame with calculated stat test and experiment parameters.
        """
        row_dict = {
            "metric": [grid_element.metric_name],
            "split_rate": [(grid_element.control_group_size, grid_element.target_group_size)]
            }
        split_column = f"is_control_{grid_element.control_group_size}_{grid_element.target_group_size}_{grid_element.split_number}"
        metric_col = f"{grid_element.metric_name}"
        guests_with_splits[split_column] = (guests_with_splits[split_column]
                                            .map({1: self.abtest_params.data_params.treatment_name, 
                                                  0: self.abtest_params.data_params.control_name})
        )
        if isinstance(grid_element, MdeBetaExperiment):
            guests_with_splits[metric_col].where(guests_with_splits[split_column]==self.abtest_params.data_params.control_name,
                                                 guests_with_splits[metric_col]*grid_element.inject, 
                                                 axis=0,
                                                 inplace=True)
            row_dict["Effect"] = [grid_element.inject]

        self.abtest_params.data_params.group_col = split_column
        self.abtest_params.data_params.target = metric_col

        ab_test = ABTest(guests_with_splits, self.abtest_params)
        ab_test = self.experiment_params.transformations(ab_test)
        row_dict["effect_significance"] = self.experiment_params.stat_test(ab_test)["result"]
        return pd.DataFrame(row_dict)

    def _fill_passed_experiments(self, aggregated_df) -> pd.DataFrame:
        """Fill Nan for passed experiments.

        Args:
            aggregated_df (pandas.DataFrame): Dataframe with calculated experiments.

        Returns:
            pandas.DataFrame: Pandas DataFrame with filled values.
        """
        for metric in self._experiment_params.metrics_names:
            for split in self.group_sizes:
                passed_mde = aggregated_df[(aggregated_df.metric == metric) &
                                           (aggregated_df.split_rate == split)]["Effect"].values
                last_experiment = min(passed_mde)
                passed_injects = np.setdiff1d(self._experiment_params.injects, passed_mde)
                if len(passed_injects) > 0:
                    failed = list(itertools.product([metric], [split], passed_injects[passed_injects < last_experiment],
                                                    [f">={self._experiment_params.max_beta_score}"])
                                 )  # error higher than max_beta_score
                    succes = list(itertools.product([metric], [split], passed_injects[passed_injects > last_experiment],
                                                    [f"<={self._experiment_params.min_beta_score}"])
                                 )
                    succes.extend(failed)
                    df_passed = pd.DataFrame.from_records(succes)
                    df_passed.columns = aggregated_df.columns
                    aggregated_df = pd.concat([aggregated_df, df_passed])
        return aggregated_df

    def _beta_score_calculation(self, df_with_calc: pd.DataFrame) -> pd.DataFrame:
        """Calculates II type error for df with calculated experiments.

        Args:
            df_with_calc (pandas.DataFrame): Dataframe with calculated experiments.

        Returns:
            pandas.DataFrame: DataFrame with II type error.
        """
        res_agg = (df_with_calc
                   .groupby(by=["metric", "split_rate", "Effect"])
                   .agg(sum=("effect_significance", sum),
                        count=("effect_significance", "count"))
                  )
        res_agg["beta"] = round((1.0 - res_agg["sum"]/res_agg["count"]),
                                self._number_of_decimals)
        return res_agg

    @staticmethod
    def _fill_res_with_default(df: pd.DataFrame,
                           column_name: str,
                           min_val: float,
                           max_val: float
    ) -> pd.DataFrame:
        """Fill column with defalt values.

        Args:
            df (pandas.DataFrame): Pandas Dataframe for replace default values.
            column_name (str): DataFrame's column name for replace values.
            min_val (float): Minimum value for replace.
            max_val (float): Maximum value for replace.

        Returns:
            pandas.DataFrame: df with replaced values.
        """
        df[column_name] = np.where(df[column_name] >= max_val,
                                   f">={max_val}",
                                   np.where(df[column_name] <= min_val,
                                             f"<={min_val}",
                                             df[column_name]))
        return df

    def _calc_beta(self, 
                   guests_with_splits,
                   fill_with_default: bool = True
    ) -> pd.DataFrame:
        """Calculates II type error.

        Args:
            guests_with_splits (): Dataframe with precalculated splits.
            fill_with_default (bool): Fill calculated vaules with defaults.

        Returns:
            pandas.DataFrame: Pandas DataFrame with II type error.
        """
        beta_scores = pd.DataFrame()
        res_agg = pd.DataFrame()
        for metric_name in self.experiment_params.metrics_names:
            # index of max inject in self.experiment_params.injects
            max_found_inject_ind = 0
            for group_size in self.group_sizes:
                found_min_inject_flg = False
                found_max_inject_flg = False
                for inject in sorted(self.experiment_params.injects, reverse=True)[max_found_inject_ind:]:
                    if(found_min_inject_flg and found_max_inject_flg):
                        continue
                    else:
                        for split_number in range(1,self.experiment_params.iterations_number + 1):
                            # experiment
                            experiment_params = MdeBetaExperiment(group_sizes=group_size,
                                                                  split_number=split_number,
                                                                  metric_name=metric_name,
                                                                  inject=inject)
                            split_column = f"is_control_{experiment_params.control_group_size}_{experiment_params.target_group_size}_{experiment_params.split_number}"
                            one_split_guests = (guests_with_splits.loc[guests_with_splits[split_column]
                                                                    .isin([0,1])]
                            )
                            experiment_res = self._calc_experiment_grid_cell(one_split_guests, 
                                                                             experiment_params)
                            beta_scores = pd.concat([beta_scores, experiment_res], axis=0)

                        calculated_experiments = ((beta_scores["split_rate"] == group_size) &
                                                       (beta_scores["metric"] == metric_name) &
                                                       (beta_scores["Effect"] == inject))
                        res_inject_agg = self._beta_score_calculation(beta_scores[calculated_experiments])
                        res_agg = pd.concat([res_agg, res_inject_agg], axis=0)
                        # check if beta score higher than min_beta
                        if(res_inject_agg["beta"].values >= round(self.experiment_params.min_beta_score,
                                                           self._number_of_decimals)) and not found_min_inject_flg:
                            max_found_inject_ind = sorted(self.experiment_params.injects, reverse=True).index(inject)
                            found_min_inject_flg = True

                        if(res_inject_agg["beta"].values >= round(self.experiment_params.max_beta_score,
                                                           self._number_of_decimals)) and not found_max_inject_flg:
                            found_max_inject_flg = True

        res_agg.drop(columns=["sum", "count"], inplace=True)
        res_agg = res_agg.reset_index()
        if fill_with_default:
            res_agg = self._fill_res_with_default(res_agg,
                                                "beta",
                                                self.experiment_params.min_beta_score,
                                                self.experiment_params.max_beta_score)
        # append passed experiments
        res_agg = self._fill_passed_experiments(res_agg)
        res_agg["Effect"] = res_agg["Effect"].apply(lambda mde: f"{round((mde//1.0 * 100 + mde%1.0 * 100) - 100, 5)}%")
        res_pivoted = pd.pivot_table(res_agg,
                                     values="beta",
                                     index=["metric", "Effect"],
                                     columns="split_rate",
                                     aggfunc=lambda x: x)
        if fill_with_default:
            res_pivoted.replace(0, f"<={self.experiment_params.min_beta_score}", inplace=True)
        return res_pivoted

    @staticmethod
    def _first_found_mde(df_column: pd.Series) -> int:
        """Calculate max possible MDE for group sizes.

        Args:
            df_column (pandas.Series): DataFrame's column with II type error scores.

        Returns:
            int: Max possible MDE for group sizes.
        """
        if not df_column[(df_column != 0) & (df_column != 1)]:
            return "Effect wasn't detected"
        else:
            return df_column[(df_column != 0) & (df_column != 1)].idxmax()[1]

    def calc_alpha(self, guests: pd.DataFrame,
                   is_splitted: bool = False) -> pd.DataFrame:
        """Calculates I type error.

        Args:
            guests (pandas.DataFrame): Dataframe with guests.
            is_splitted (bool): If False guests must contain splits for calculation. Otherwise splits will be compute for guests.

        Returns:
            pandas.DataFrame: Pandas DataFrame with I type error.
        """
        if not is_splitted:
            prepilot_guests_collector = MultipleSplitBuilder(guests, self.experiment_params.metrics_names,
                                                             self.experiment_params.injects, self.group_sizes,
                                                             self.stratification_params,
                                                             self.experiment_params.iterations_number)
            guests = prepilot_guests_collector.collect(guests)

        alpha_scores = pd.DataFrame()

        for metric_name in self.experiment_params.metrics_names:
            for group_size in self.group_sizes:
                for split_number in range(1,self.experiment_params.iterations_number + 1):
                    experiment_params = MdeAlphaExperiment(group_sizes=group_size,
                                                           split_number=split_number,
                                                           metric_name=metric_name)
                    split_column = f"is_control_{experiment_params.control_group_size}_{experiment_params.target_group_size}_{experiment_params.split_number}"
                    one_split_guests = (guests.loc[guests[split_column]
                                                            .isin([0,1])]
                    )
                    experiment = self._calc_experiment_grid_cell(one_split_guests, 
                                                                 experiment_params)
                    alpha_scores = pd.concat([alpha_scores, experiment], axis=0)

        res_agg = (alpha_scores.groupby(by=["metric", "split_rate"])
                   .agg(sum=("effect_significance", sum),
                        count=("effect_significance", "count")))
        res_agg["alpha"] = res_agg["sum"]/res_agg["count"]
        res_agg.drop(columns=["sum", "count"], inplace=True)
        res_agg = res_agg.reset_index()
        res_pivoted = (pd.pivot_table(res_agg,
                                      values="alpha",
                                      index=["metric"],
                                      columns="split_rate")
                    )
        return res_pivoted

    def collect(self, fill_with_default: bool = True) -> pd.DataFrame:
        """Calculates I and II types error using prepilot parameters.

        Args:
            fill_with_default (bool): Fill calculated vaules with defaults.

        Returns:
            pandas.DataFrame: Pandas DataFrames with aggregated results of experiment.
        """
        prepilot_split_builder = MultipleSplitBuilder(self.guests,
                                                         self.experiment_params.metrics_names,
                                                         self.experiment_params.injects,
                                                         self.group_sizes,
                                                         self.stratification_params,
                                                         self.experiment_params.iterations_number)

        prepilot_guests = prepilot_split_builder.collect()

        beta = self._calc_beta(prepilot_guests, fill_with_default)
        alpha = self.calc_alpha(prepilot_guests, is_splitted = True)
        return beta, alpha
