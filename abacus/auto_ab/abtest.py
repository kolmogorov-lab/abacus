from __future__ import annotations
from typing import Optional, Tuple, Dict
import copy
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy.stats import mannwhitneyu, ttest_ind, shapiro, mode, t, chisquare, norm
from statsmodels.stats.proportion import proportions_ztest
from abacus.auto_ab.graphics import Graphics
from abacus.auto_ab.variance_reduction import VarianceReduction
from abacus.auto_ab.params import ABTestParams
from abacus.resplitter.resplit_builder import ResplitBuilder
from abacus.resplitter.params import ResplitParams
from abacus.types import ArrayNumType, DataFrameType, StatTestResultType

warnings.simplefilter('always')

class ABTest:
    """Performs different calculations of A/B-test.

    - Results evaluation for different metric types (continuous, binary, ratio).
    - Bucketing (decrease number of points, normal distribution of metric of interest)

    Example:

    .. code-block:: python

        from abacus.auto_ab.abtest import ABTest
        from abacus.auto_ab.params import ABTestParams, DataParams, HypothesisParams

        data_params = DataParams(...)
        hypothesis_params = HypothesisParams(...)
        ab_params = ABTestParams(data_params, hypothesis_params)

        df = pd.read_csv('data.csv')
        ab_test = ABTest(df, ab_params)
        ab_test.test_welch()
        # {'stat': 5.172, 'p-value': 0.312, 'result': 0}
    """

    def __init__(self,
                 dataset: Optional[DataFrameType],
                 params: ABTestParams
                 ) -> None:
        self.params = params
        self.__dataset = dataset

        if dataset is not None \
                and len(params.data_params.transforms) == 0\
                and params.hypothesis_params.metric_type in ('continuous', 'binary'):
            self.__check_required_columns(dataset, 'init')
            self.params.data_params.control = self.__get_group(self.params.data_params.control_name, self.dataset)
            self.params.data_params.treatment = self.__get_group(self.params.data_params.treatment_name, self.dataset)

    @property
    def dataset(self) -> DataFrameType:
        return self.__dataset

    def __str__(self) -> str:
        return """
            ABTest(alpha={alpha}, beta={beta}, alternative={alternative},
                   metric type={metric_type}, metric_name={metric_name})
        """.format(alpha=self.params.hypothesis_params.alpha,
                   beta=self.params.hypothesis_params.beta,
                   alternative=self.params.hypothesis_params.alternative,
                   metric_type=self.params.hypothesis_params.metric_type,
                   metric_name=self.params.hypothesis_params.metric_name)

    def __check_applied_transformation(self, method) -> None:
        if method in self.params.data_params.transforms:
            warnings.warn(f'Method `{method}` has already been called before. '
                          f'It will be applied again, but you should check whether it is needed twice.')

    def __check_required_metric_type(self, method: str) -> None:
        available_metric_methods = {
            'continuous': ['report_continuous', 'cuped', 'cupac', 'bucketing', 'filter_outliers',
                           'metric_transform', 'test_boot_fp', 'test_boot_welch', 'test_boot_confint',
                           'test_welch', 'test_mannwhitney', 'test_buckets', 'manual_ttest', 'linearization'],
            'binary': ['report_binary', 'test_boot_ratio', 'test_z_proportions', 'test_chisquare'],
            'ratio': ['report_ratio', 'linearization', 'test_delta_ratio', 'test_taylor_ratio'],
        }

        incorrect_metric_type = ''
        if method in available_metric_methods['continuous']:
            incorrect_metric_type = 'continuous'
        elif method in available_metric_methods['binary']:
            incorrect_metric_type = 'binary'
        elif method in available_metric_methods['ratio']:
            incorrect_metric_type = 'ratio'

        if method not in available_metric_methods[self.params.hypothesis_params.metric_type]:
            raise ValueError("Incorrect metric type: '{incorrect_metric_type}' required, but '{current_metric_type}' provided". \
                             format(incorrect_metric_type=incorrect_metric_type,
                                    current_metric_type=self.params.hypothesis_params.metric_type))

    def __check_required_columns(self, df: DataFrameType, method: str) -> None:
        """Check presence of columns in dataframe.

        Args:
            df (pandas.DataFrame): DataFrame to check.
            method (str): Stage of A/B process which you'd like to test.

        Raises:
            ValueError: If `is_valid_col` is False. Experiment cannot be provided
            if required columns are absent.
        """
        cols: Dict[str, str] = {}
        if method == 'init':
            cols = {
                'id_col': self.params.data_params.id_col,
                'group_col': self.params.data_params.group_col
            }
            if self.params.hypothesis_params.metric_type == 'continuous':
                cols['target'] = self.params.data_params.target
            elif self.params.hypothesis_params.metric_type == 'binary':
                cols['target_flg'] = self.params.data_params.target_flg
            elif self.params.hypothesis_params.metric_type == 'ratio':
                cols['numerator'] = self.params.data_params.numerator
                cols['denominator'] = self.params.data_params.denominator
        elif method == 'cuped':
            cols = {'covariate': self.params.data_params.covariate}
        elif method == 'cupac':
            cols = {'predictors_prev': self.params.data_params.predictors_prev,
                    'predictors_now': self.params.data_params.predictors_now,
                    'target_prev': self.params.data_params.target_prev}
        elif method == 'resplit_df':
            cols = {'strata_col': self.params.data_params.strata_col}

        not_correct_fields = []
        df_cols = df.columns
        for field, value in cols.items():
            if value == '' or value not in df_cols:
                not_correct_fields.append(field)

        if len(not_correct_fields) > 0:
            raise ValueError(f'You did not provide or provide incorrectly following data attributes: {not_correct_fields}')

    def __get_group(self, group_label: str, df: Optional[DataFrameType] = None) -> np.ndarray:
        """Gets target metric column based on desired group label.

        Args:
            group_label (str): Group label, e.g. 'A', 'B'.
            df (DataFrameType, optional): DataFrame to query from.

        Returns:
            numpy.ndarray: Target column for a desired group.
        """
        x = df if df is not None else self.__dataset
        group = np.array([])
        if self.params.hypothesis_params.metric_type == 'continuous':
            group = x.loc[x[self.params.data_params.group_col] == group_label,
            self.params.data_params.target].to_numpy()
        elif self.params.hypothesis_params.metric_type == 'binary':
            group = x.loc[x[self.params.data_params.group_col] == group_label,
            self.params.data_params.target_flg].to_numpy()
        return group

    def __bucketize(self, x: ArrayNumType) -> np.ndarray:
        """Split array ``x`` into N non-overlapping buckets.

        There are two purposes for these actions:

        1. Decrease number of data points of experiment.
        2. Get normal distribution of a metric of interest.

        Procedure:

        1. Shuffle elements of an array.
        2. Split points into N non-overlapping buckets.
        3. On every bucket calculate metric of interest.

        Args:
            x (np.ndarray): Array to split.

        Returns:
            np.ndarray: Splitted into buckets array.
        """
        np.random.shuffle(x)
        x_new = np.array([self.params.hypothesis_params.metric(x_)
                          for x_ in np.array_split(x, self.params.hypothesis_params.n_buckets)])
        return x_new

    def __manual_ttest(self, ctrl_mean: float, ctrl_var: float, ctrl_size: int,
                      treat_mean: float, treat_var: float, treat_size: int) -> StatTestResultType:
        """Performs Welch's t-test based on aggregation of metrics instead of datasets.

        For empirical calculation of T-statistic we need: expectation, variance, array size for each group.

        Args:
            ctrl_mean (float): Mean of control group.
            ctrl_var (float): Variance of control group.
            ctrl_size (int): Size of control group.
            treat_mean (float): Mean of treatment group.
            treat_var (float): Variance of treatment group.
            treat_size (int): Size of treatment group.

        Returns:
            stat_test_typing: Dictionary with following properties: test statistic, p-value, test result. Test result: 1 - significant different, 0 - insignificant difference.
        """
        self.__check_required_metric_type('manual_ttest')

        t_stat_empirical = (treat_mean - ctrl_mean) / (ctrl_var / ctrl_size + treat_var / treat_size) ** (1 / 2)
        df = ((ctrl_var / ctrl_size + treat_var / treat_size) ** 2 /
             (ctrl_var ** 2 / (ctrl_size ** 2 * (ctrl_size - 1)) +
              (treat_var ** 2 / (treat_size ** 2 * (treat_size - 1)))))

        test_result: int = 0
        if self.params.hypothesis_params.alternative == 'two-sided':
            lcv, rcv = t.ppf(self.params.hypothesis_params.alpha / 2, df=df, loc=ctrl_mean, scale=np.sqrt(ctrl_var)), \
                t.ppf(1.0 - self.params.hypothesis_params.alpha / 2, df=df, loc=ctrl_mean, scale=np.sqrt(ctrl_var))
            if not (lcv < t_stat_empirical < rcv):
                test_result = 1
        elif self.params.hypothesis_params.alternative == 'less':
            lcv = t.ppf(self.params.hypothesis_params.alpha, df=df, loc=ctrl_mean, scale=np.sqrt(ctrl_var))
            if t_stat_empirical < lcv:
                test_result = 1
        elif self.params.hypothesis_params.alternative == 'greater':
            rcv = t.ppf(1 - self.params.hypothesis_params.alpha, df=df, loc=ctrl_mean, scale=np.sqrt(ctrl_var))
            if t_stat_empirical > rcv:
                test_result = 1

        result = {
            'stat': t_stat_empirical,
            'p-value': None,
            'result': test_result
        }
        return result

    def __delta_params(self, x: DataFrameType) -> Tuple[float, float]:
        """Calculated expectation and variance for ratio metric using delta approximation.

        Source: https://arxiv.org/pdf/1803.06336.pdf.

        Args:
            x (pandas.DataFrame): Pandas DataFrame of particular group (A, B, etc).

        Returns:
            Tuple[float, float]: Mean and variance of ratio metric.
        """
        num = x[self.params.data_params.numerator]
        den = x[self.params.data_params.denominator]
        num_mean, den_mean = num.mean(), den.mean()
        num_var, den_var = num.var(), den.var()
        cov = x[[self.params.data_params.numerator, self.params.data_params.denominator]].cov().iloc[0, 1]
        n = len(num)

        bias_correction = (den_mean / num_mean ** 3) * (num_var / n) - cov / (n * num_mean ** 2)
        mean = den_mean / num_mean - 1 + bias_correction
        var = den_var / num_mean ** 2 - 2 * (den_mean / num_mean ** 3) * cov + (den_mean ** 2 / num_mean ** 4) * num_var

        return mean, var

    def __taylor_params(self, x: DataFrameType) -> Tuple[float, float]:
        """ Calculated expectation and variance for ratio metric using Taylor expansion approximation.

        Source: https://www.stat.cmu.edu/~hseltman/files/ratio.pdf.

        Args:
            x (pandas.DataFrame): Pandas DataFrame of particular group (A, B, etc).

        Returns:
            Tuple[float, float]: Mean and variance of ratio metric.
        """
        num = x[self.params.data_params.numerator]
        den = x[self.params.data_params.denominator]
        mean = num.mean() / den.mean() - \
               x[[self.params.data_params.numerator, self.params.data_params.denominator]].cov().iloc[0, 1] \
               / (den.mean() ** 2) + den.var() * num.mean() / (den.mean() ** 3)
        var = (num.mean() ** 2) / (den.mean() ** 2) * (num.var() / (num.mean() ** 2) - 2 *
                                                       x[[self.params.data_params.numerator,
                                                          self.params.data_params.denominator]].cov().iloc[0, 1]) / \
              (num.mean() * den.mean() + den.var() / (den.mean() ** 2))

        return mean, var

    def __report_binary(self) -> str:
        self.__check_required_metric_type('report_binary')

        hypothesis = self.params.hypothesis_params
        ctrl = self.params.data_params.control
        trtm = self.params.data_params.treatment

        ztest = self.test_z_proportions()
        ztest_res = 'H0 is not rejected' if ztest['result'] == 0 else 'H0 is rejected'

        try:  # chi-square works well
            chisq = self.test_chisquare()
            chisq_res = 'H0 is not rejected' if chisq['result'] == 0 else 'H0 is rejected'
            chisq_result = f"- Chi-square - test: {chisq['stat']: .2f}, p - value = {chisq['p-value']: .4f}, {chisq_res}."
            test_result = chisq['result'] + ztest['result']
            num_of_tests = 2
        except:
            chisq_result = ''
            test_result = ztest['result']
            num_of_tests = 1

        test_explanation = f'{test_result} out of {num_of_tests} stat.test show that H0 is rejected.'
        transforms: ArrayNumType = self.params.data_params.transforms
        if len(transforms) > 0:
            transforms_str = 'Transformations applied: ' + ' -> '.join(transforms) + '.'
        else:
            transforms_str = 'No transformations applied.'

        params = {
            'ztest_stat': ztest['stat'], 'ztest_pvalue': ztest['p-value'], 'ztest_result': ztest_res,
            'chi_square': chisq_result,
            'ctrl_conv': sum(ctrl) / len(ctrl), 'trtm_conv': sum(trtm) / len(trtm),
            'ctrl_obs': len(ctrl), 'trtm_obs': len(trtm),
            'alpha': hypothesis.alpha, 'beta': hypothesis.beta, 'alternative': hypothesis.alternative,
            'metric_name': hypothesis.metric_name,
            'transforms': transforms_str,
            'test_explanation': test_explanation
        }

        output = '''
Parameters of experiment:
- Metric type: binary.
- Metric: {metric_name}.
- Errors: alpha = {alpha}, beta = {beta}.
- Alternative: {alternative}.

Control group:
- Observations: {ctrl_obs}
- Conversion: {ctrl_conv}

Treatment group:
- Observations: {trtm_obs}
- Conversion: {trtm_conv}

{transforms}

Following statistical tests are used:
- Z-test: {ztest_stat:.2f}, p-value = {ztest_pvalue:.4f}, {ztest_result}.
{chi_square}

{test_explanation}
        '''.format(**params)

        return output

    def __report_continuous(self) -> str:
        self.__check_required_metric_type('report_continuous')

        hypothesis = self.params.hypothesis_params
        ctrl = self.params.data_params.control
        trtm = self.params.data_params.treatment

        welch = self.test_welch()
        welch_res = 'H0 is not rejected' if welch['result'] == 0 else 'H0 is rejected'
        mwu = self.test_mannwhitney()
        mwu_res = 'H0 is not rejected' if mwu['result'] == 0 else 'H0 is rejected'
        boot = self.test_boot_confint()
        boot_res = 'H0 is not rejected' if boot['result'] == 0 else 'H0 is rejected'

        test_result = welch['result'] + mwu['result'] + boot['result']
        test_explanation = ''
        if test_result == 3:
            test_explanation = 'All three stat. tests showed that H0 is rejected.'
        elif test_result == 2:
            test_explanation = 'Two out of three stat. tests showed that H0 is rejected.'
        elif test_result == 1:
            test_explanation = 'Two out of three stat. tests showed that H0 is not rejected.'
        elif test_result == 0:
            test_explanation = 'All three stat. tests showed that H0 is not rejected.'

        bucketing_str = ''
        if 'bucketing' in self.params.data_params.transforms:
            bucketing_str = f'Number of buckets: {hypothesis.n_buckets}.\n'

        metric_transform_str = ''
        if 'metric transform' in self.params.data_params.transforms:
            metric_transform_str = f'Metric transformation applied: {hypothesis.metric_transform.__name__}.\n'

        variance_reduction_str = ''
        filter_outliers_str = ''
        if 'filter outliers' in self.params.data_params.transforms:
            filter_outliers_str = f'Outliers filtering method applied: {hypothesis.filter_method}.\n'

        transforms: ArrayNumType = self.params.data_params.transforms
        if len(transforms) > 0:
            transforms_str = 'Transformations applied: ' + ' -> '.join(transforms) + '.\n'
        else:
            transforms_str = 'No transformations applied.\n'

        params = {
            'welch_stat': welch['stat'], 'welch_pvalue': welch['p-value'], 'welch_result': welch_res,
            'mwu_stat': mwu['stat'], 'mwu_pvalue': mwu['p-value'], 'mwu_result': mwu_res,
            'boot_result': boot_res,
            'ctrl_obs': len(ctrl), 'trtm_obs': len(trtm),
            'ctrl_mean': np.mean(ctrl), 'ctrl_median': np.median(ctrl), 'ctrl_25th': np.quantile(ctrl, 0.25),
            'ctrl_75th': np.quantile(ctrl, 0.75), 'ctrl_min': np.min(ctrl), 'ctrl_max': np.max(ctrl),
            'ctrl_std': np.std(trtm), 'ctrl_var': np.var(trtm),
            'trtm_mean': np.mean(trtm), 'trtm_median': np.median(trtm), 'trtm_25th': np.quantile(trtm, 0.25),
            'trtm_75th': np.quantile(trtm, 0.75), 'trtm_min': np.min(trtm), 'trtm_max': np.max(trtm),
            'trtm_std': np.std(trtm), 'trtm_var': np.var(trtm),
            'alpha': hypothesis.alpha, 'beta': hypothesis.beta, 'alternative': hypothesis.alternative,
            'metric_name': hypothesis.metric_name, 'bucketing_str': bucketing_str,
            'transforms': transforms_str, 'metric_transform_str': metric_transform_str,
            'filter_outliers_str': filter_outliers_str,
            'n_boot_samples': hypothesis.n_boot_samples,
            'test_explanation': test_explanation
        }

        output = '''
Parameters of experiment:
- Metric type: continuous.
- Metric: {metric_name}.
- Errors: alpha = {alpha}, beta = {beta}.
- Alternative: {alternative}.

Control group:
- Observations: {ctrl_obs}
- Mean: {ctrl_mean:.4f}
- Median: {ctrl_median:.4f}
- 25th quantile: {ctrl_25th:.4f}
- 75th quantile: {ctrl_75th:.4f}
- Minimum: {ctrl_min:.4f}
- Maximum: {ctrl_max:.4f}
- St.deviation: {ctrl_std:.4f}
- Variance: {ctrl_var:.4f}

Treatment group:
- Observations: {trtm_obs}
- Mean: {trtm_mean:.4f}
- Median: {trtm_median:.4f}
- 25th quantile: {trtm_25th:.4f}
- 75th quantile: {trtm_75th:.4f}
- Minimum: {trtm_min:.4f}
- Maximum: {trtm_max:.4f}
- St.deviation: {trtm_std:.4f}
- Variance: {trtm_var:.4f}

{transforms}
Number of bootstrap iterations: {n_boot_samples}.\n{bucketing_str}{metric_transform_str}{filter_outliers_str}
Following statistical tests are used:
- Welch's t-test: {welch_stat:.2f}, p-value = {welch_pvalue:.4f}, {welch_result}.
- Mann Whitney's U-test: {mwu_stat:.2f}, p-value = {mwu_pvalue:.4f}, {mwu_result}.
- Bootstrap test: {boot_result}.

{test_explanation}
        '''.format(**params)

        return output

    def __report_ratio(self):
        raise NotImplementedError('Reporting for ratio metric is still in progress..')

    def bucketing(self) -> ABTest:
        """Performs bucketing in order to accelerate results computation.

        Returns:
            ABTest: New instance of ``ABTest`` class with modified control and treatment.
        """
        self.__check_applied_transformation('bucketing')
        self.__check_required_metric_type('bucketing')

        params_new = copy.deepcopy(self.params)
        params_new.data_params.control = self.__bucketize(self.params.data_params.control)
        params_new.data_params.treatment = self.__bucketize(self.params.data_params.treatment)
        params_new.data_params.transforms = np.append(params_new.data_params.transforms, 'bucketing')

        return ABTest(None, params_new)

    def cuped(self) -> ABTest:
        """Performs CUPED for variance reduction.

        Returns:
            ABTest: New instance of ``ABTest`` class with modified control and treatment.
        """
        self.__check_applied_transformation('cuped')
        self.__check_required_metric_type('cuped')
        self.__check_required_columns(self.__dataset, 'cuped')

        result_df = VarianceReduction.cuped(self.__dataset,
                                            target_col=self.params.data_params.target,
                                            groups_col=self.params.data_params.group_col,
                                            covariate_col=self.params.data_params.covariate)

        params_new = copy.deepcopy(self.params)
        params_new.data_params.control = self.__get_group(self.params.data_params.control_name, result_df)
        params_new.data_params.treatment = self.__get_group(self.params.data_params.treatment_name, result_df)
        params_new.data_params.transforms = np.append(params_new.data_params.transforms, 'cuped')

        return ABTest(result_df, params_new)

    def cupac(self) -> ABTest:
        """Performs CUPAC for variance reduction.

        Returns:
            ABTest: New instance of ``ABTest`` class with modified control and treatment.
        """
        self.__check_applied_transformation('cupac')
        self.__check_required_metric_type('cupac')
        self.__check_required_columns(self.__dataset, 'cupac')
        result_df = VarianceReduction.cupac(self.__dataset,
                                            target_prev_col=self.params.data_params.target_prev,
                                            target_now_col=self.params.data_params.target,
                                            factors_prev_cols=self.params.data_params.predictors_prev,
                                            factors_now_cols=self.params.data_params.predictors_now,
                                            groups_col=self.params.data_params.group_col)

        params_new = copy.deepcopy(self.params)
        params_new.data_params.control = self.__get_group(self.params.data_params.control_name, result_df)
        params_new.data_params.treatment = self.__get_group(self.params.data_params.treatment_name, result_df)
        params_new.data_params.transforms = np.append(params_new.data_params.transforms, 'cupac')

        return ABTest(result_df, params_new)

    def filter_outliers(self) -> ABTest:
        self.__check_applied_transformation('filter_outliers')
        self.__check_required_metric_type('filter_outliers')

        target = self.__dataset[[self.params.data_params.target]].values
        dataset_new = self.__dataset.copy()

        if self.params.hypothesis_params.filter_method == 'isolation_forest':
            not_outlier_index = IsolationForest(random_state=0).fit_predict(target) == 1
            dataset_new = self.__dataset.loc[not_outlier_index].reset_index(drop=True)

        if self.params.hypothesis_params.filter_method == 'top_5':
            quantile95 = np.quantile(target, 0.95)
            not_outlier_index = self.__dataset[self.params.data_params.target] <= quantile95
            dataset_new = self.__dataset.loc[not_outlier_index].reset_index(drop=True)

        params_new = copy.deepcopy(self.params)
        params_new.data_params.transforms = np.append(params_new.data_params.transforms, 'filter outliers')
        params_new.data_params.control = self.__get_group(self.params.data_params.control_name, dataset_new)
        params_new.data_params.treatment = self.__get_group(self.params.data_params.treatment_name, dataset_new)

        return ABTest(dataset_new, params_new)

    def linearization(self) -> ABTest:
        """Creates linearized continuous metric based on ratio-metric.
        Important: there is an assumption that all data is already grouped by user
        s.t. numerator for user = sum of numerators for user for different time periods
        and denominator for user = sum of denominators for user for different time periods

        Source: https://research.yandex.com/publications/148.
        """
        self.__check_applied_transformation('linearization')
        self.__check_required_metric_type('linearization')

        if self.params.data_params.is_grouped:
            return ABTest(self.__dataset, self.params)

        dataset_new = copy.deepcopy(self.__dataset)
        params_new = copy.deepcopy(self.params)
        num_col, den_col = 'num', 'den'

        if self.params.hypothesis_params.metric_type == 'ratio':
            numerator_col_name = self.params.data_params.numerator
            denominator_col_name = self.params.data_params.denominator

            df_grouped = self.__dataset.groupby(by=[self.params.data_params.id_col,
                                                    self.params.data_params.group_col]) \
                                        .agg({numerator_col_name: 'sum', denominator_col_name: 'sum'}) \
                                        .rename(columns={numerator_col_name: num_col, denominator_col_name: den_col}) \
                                        .reset_index()
            self.__dataset = df_grouped

        elif self.params.hypothesis_params.metric_type == 'continuous':
            df_grouped = self.__dataset.groupby(by=[self.params.data_params.id_col,
                                                    self.params.data_params.group_col],
                                                 as_index=False)[self.params.data_params.target] \
                                        .agg(['sum', 'count']) \
                                        .rename(columns={'sum': num_col, 'count': den_col}) \
                                        .reset_index()

            self.__dataset = df_grouped


        ctrl = self.__dataset.loc[self.__dataset[self.params.data_params.group_col] == self.params.data_params.control_name]
        k = round(sum(ctrl[num_col]) / sum(ctrl[den_col]), 5)

        new_target_name = 'target_linearized'
        self.__dataset.loc[:, new_target_name] = self.__dataset[num_col] - k * self.__dataset[den_col]

        dataset_new = dataset_new.merge(self.__dataset[[self.params.data_params.id_col,
                                                        new_target_name]],
                                        how='left', on=self.params.data_params.id_col)
        dataset_new = dataset_new.drop_duplicates(subset=[self.params.data_params.id_col])

        params_new.data_params.target = new_target_name
        params_new.data_params.control = dataset_new.loc[
                                            dataset_new[self.params.data_params.group_col] == self.params.data_params.control_name,
                                            new_target_name].to_numpy()
        params_new.data_params.treatment = dataset_new.loc[
                                            dataset_new[self.params.data_params.group_col] == self.params.data_params.treatment_name,
                                            new_target_name].to_numpy()
        params_new.data_params.transforms = np.append(params_new.data_params.transforms, 'linearization')

        params_new.hypothesis_params.metric_type = 'continuous'

        return ABTest(dataset_new, params_new)

    def metric_transform(self) -> ABTest:
        self.__check_applied_transformation('metric_transform')
        self.__check_required_metric_type('metric_transform')

        if self.params.hypothesis_params.metric_transform is None:
            return ABTest(self.__dataset, self.params)

        dataset_new = copy.deepcopy(self.__dataset)
        target = self.params.data_params.target
        group_col = self.params.data_params.group_col

        transform = self.params.hypothesis_params.metric_transform
        transform_name = transform.__name__

        control_name = self.params.data_params.control_name
        control_flg = dataset_new[group_col] == control_name
        dataset_new.loc[control_flg, target] = transform(dataset_new.loc[control_flg, target].to_numpy())

        treatment_name = self.params.data_params.treatment_name
        treatment_flg = dataset_new[group_col] == treatment_name
        dataset_new.loc[treatment_flg, target] = transform(dataset_new.loc[treatment_flg, target].to_numpy())

        params_new = copy.deepcopy(self.params)
        params_new.data_params.transforms = np.append(params_new.data_params.transforms, 'metric transform')
        params_new.data_params.control = transform(dataset_new.loc[control_flg, target].to_numpy())
        params_new.data_params.treatment = transform(dataset_new.loc[treatment_flg, target].to_numpy())

        return ABTest(dataset_new, params_new)

    def plot(self) -> None:
        """Plot experiment.

        Plot figure type depends on the following parameters:

        - hypothesis_params.metric_name
        - hypothesis_params.strategy
        """
        if self.params.hypothesis_params.metric_type == 'continuous':
            Graphics.plot_continuous_experiment(self.params)

        if self.params.hypothesis_params.metric_type == 'binary':
            Graphics.plot_binary_experiment(self.params)

    def report(self) -> None:
        report_output = 'Report for ratio metric currently not supported.'

        if self.params.hypothesis_params.metric_type == 'continuous':
            report_output = self.__report_continuous()

        if self.params.hypothesis_params.metric_type == 'binary':
            report_output = self.__report_binary()

        print(report_output)

    def resplit_df(self) -> ABTest:
        """Resplit dataframe.

        Returns:
            ABTest: Instance of ``ABTest`` class with modified control and treatment.
        """
        resplit_params = ResplitParams(
            group_col=self.params.data_params.group_col,
            strata_col=self.params.data_params.strata_col
        )
        resplitter = ResplitBuilder(self.__dataset, resplit_params)
        new_dataset = resplitter.collect()

        return ABTest(new_dataset, self.params)

    def test_boot_fp(self) -> StatTestResultType:
        """ Performs bootstrap hypothesis testing by calculation of false positives.

        Returns:
            stat_test_typing: Dictionary with following properties: ``test statistic``, ``p-value``, ``test result``. Test result: 1 - significant different, 0 - insignificant difference.
        """
        self.__check_required_metric_type('test_boot_fp')

        x = self.params.data_params.control
        y = self.params.data_params.treatment

        metric_diffs: ArrayNumType = []
        for _ in range(self.params.hypothesis_params.n_boot_samples):
            x_boot = np.random.choice(x, size=x.shape[0], replace=True)
            y_boot = np.random.choice(y, size=y.shape[0], replace=True)
            metric_diffs.append(
                self.params.hypothesis_params.metric(y_boot) - self.params.hypothesis_params.metric(x_boot))
        pd_metric_diffs = pd.DataFrame(metric_diffs)

        left_quant = self.params.hypothesis_params.alpha / 2
        right_quant = 1 - self.params.hypothesis_params.alpha / 2
        ci = pd_metric_diffs.quantile([left_quant, right_quant])
        ci_left, ci_right = float(ci.iloc[0]), float(ci.iloc[1])

        criticals = [0, 0]
        for boot in pd_metric_diffs:
            if boot < 0 and boot < ci_left:
                criticals[0] += 1
            elif boot > 0 and boot > ci_right:
                criticals[1] += 1
        false_positive = min(criticals) / pd_metric_diffs.shape[0]

        test_result: int = 0  # 0 - cannot reject H0, 1 - reject H0
        if false_positive <= self.params.hypothesis_params.alpha:
            test_result = 1

        result = {
            'stat': None,
            'p-value': false_positive,
            'result': test_result
        }
        return result

    def test_boot_confint(self) -> StatTestResultType:
        """ Performs bootstrap confidence interval and zero
        statistical significance.

        Returns:
            stat_test_typing: Dictionary with following properties: ``test statistic``, ``p-value``, ``test result``. Test result: 1 - significant different, 0 - insignificant difference.
        """
        self.__check_required_metric_type('test_boot_confint')

        x = self.params.data_params.control
        y = self.params.data_params.treatment

        metric_diffs: ArrayNumType = []
        for _ in range(self.params.hypothesis_params.n_boot_samples):
            x_boot = np.random.choice(x, size=x.shape[0], replace=True)
            y_boot = np.random.choice(y, size=y.shape[0], replace=True)
            metric_diffs.append(self.params.hypothesis_params.metric(y_boot) -
                                self.params.hypothesis_params.metric(x_boot))
        pd_metric_diffs = pd.DataFrame(metric_diffs)

        boot_mean = pd_metric_diffs.mean()
        boot_std = pd_metric_diffs.std()
        zero_pvalue = norm.sf(0, loc=boot_mean, scale=boot_std)[0]

        test_result: int = 0  # 0 - cannot reject H0, 1 - reject H0
        if self.params.hypothesis_params.alternative == 'two-sided':
            left_quant = self.params.hypothesis_params.alpha / 2
            right_quant = 1 - self.params.hypothesis_params.alpha / 2
            ci = pd_metric_diffs.quantile([left_quant, right_quant])
            ci_left, ci_right = float(ci.iloc[0]), float(ci.iloc[1])

            if ci_left > 0 or ci_right < 0:  # 0 is not in critical area
                test_result = 1
        elif self.params.hypothesis_params.alternative == 'less':
            left_quant = self.params.hypothesis_params.alpha
            ci = pd_metric_diffs.quantile([left_quant])
            ci_left = float(ci.iloc[0])
            if ci_left < 0:  # 0 is not is critical area
                test_result = 1
        elif self.params.hypothesis_params.alternative == 'greater':
            right_quant = self.params.hypothesis_params.alpha
            ci = pd_metric_diffs.quantile([right_quant])
            ci_right = float(ci.iloc[0])
            if 0 < ci_right:  # 0 is not in critical area
                test_result = 1

        result = {
            'stat': None,
            'p-value': zero_pvalue,
            'result': test_result
        }
        return result

    def test_boot_ratio(self) -> StatTestResultType:
        """Performs bootstrap for ratio-metric.

        Returns:
            stat_test_typing: Dictionary with following properties: ``test statistic``, ``p-value``, ``test result``. Test result: 1 - significant different, 0 - insignificant difference.
        """
        self.__check_required_metric_type('test_boot_ratio')

        x = self.__dataset[self.__dataset[self.params.data_params.group_col] == self.params.data_params.control_name]
        y = self.__dataset[self.__dataset[self.params.data_params.group_col] == self.params.data_params.treatment_name]

        a_metric_total = sum(x[self.params.data_params.numerator]) / sum(x[self.params.data_params.denominator])
        b_metric_total = sum(y[self.params.data_params.numerator]) / sum(y[self.params.data_params.denominator])
        origin_mean = b_metric_total - a_metric_total
        boot_diffs = []
        boot_a_metric = []
        boot_b_metric = []

        for _ in range(self.params.hypothesis_params.n_boot_samples):
            a_ids = x[self.params.data_params.id_col].sample(x[self.params.data_params.id_col].nunique(), replace=True)
            b_ids = y[self.params.data_params.id_col].sample(y[self.params.data_params.id_col].nunique(), replace=True)

            a_boot = x[x[self.params.data_params.id_col].isin(a_ids)]
            b_boot = y[y[self.params.data_params.id_col].isin(b_ids)]
            a_boot_metric = sum(a_boot[self.params.data_params.numerator]) / sum(
                a_boot[self.params.data_params.denominator])
            b_boot_metric = sum(b_boot[self.params.data_params.numerator]) / sum(
                b_boot[self.params.data_params.denominator])
            boot_a_metric.append(a_boot_metric)
            boot_b_metric.append(b_boot_metric)
            boot_diffs.append(b_boot_metric - a_boot_metric)

        # correction
        boot_mean = np.mean(boot_diffs)
        delta = abs(origin_mean - boot_mean)
        boot_diffs = [boot_diff + delta for boot_diff in boot_diffs]
        pd_metric_diffs = pd.DataFrame(boot_diffs)

        left_quant = self.params.hypothesis_params.alpha / 2
        right_quant = 1 - self.params.hypothesis_params.alpha / 2
        ci = pd_metric_diffs.quantile([left_quant, right_quant])
        ci_left, ci_right = float(ci.iloc[0]), float(ci.iloc[1])

        test_result: int = 0  # 0 - cannot reject H0, 1 - reject H0
        if ci_left > 0 or ci_right < 0:  # left border of ci > 0 or right border of ci < 0
            test_result = 1

        result = {
            'stat': None,
            'p-value': None,
            'result': test_result
        }
        return result

    def test_boot_welch(self) -> StatTestResultType:
        r""" Performs Welch's t-test for independent samples with unequal number of observations and variance.

        Welch's t-test is used as a wider approaches with fewer restrictions on samples size as in Student's t-test.

        Statistic of the test:

        .. math::
            t = \frac{\hat{X}_1 - \hat{X}_2}{\sqrt{\frac{s_1}{\sqrt{N_1}} + \frac{s_2}{\sqrt{N_2}}}}.

        Returns:
            stat_test_typing: Dictionary with following properties: ``test statistic``, ``p-value``, ``test result``. Test result: 1 - significant different, 0 - insignificant difference.
        """
        self.__check_required_metric_type('test_boot_welch')

        x = self.params.data_params.control
        y = self.params.data_params.treatment

        t_calc: int = 0
        for _ in range(self.params.hypothesis_params.n_boot_samples):
            x_boot = np.random.choice(x, size=x.shape[0], replace=True)
            y_boot = np.random.choice(y, size=y.shape[0], replace=True)

            t_boot = (np.mean(x_boot) - np.mean(y_boot)) / (
                        np.var(x_boot) / x_boot.shape[0] + np.var(y_boot) / y_boot.shape[0])
            test_res = ttest_ind(y_boot, x_boot, equal_var=False, alternative=self.params.hypothesis_params.alternative)

            if t_boot >= test_res[1]:
                t_calc += 1

        pvalue = t_calc / self.params.hypothesis_params.n_boot_samples

        test_result: int = 0  # 0 - cannot reject H0, 1 - reject H0
        if pvalue <= self.params.hypothesis_params.alpha:
            test_result = 1

        result = {
            'stat': None,
            'p-value': pvalue,
            'result': test_result
        }
        return result

    def test_buckets(self) -> StatTestResultType:
        """ Performs buckets hypothesis testing.

        Returns:
            stat_test_typing: Dictionary with following properties: ``test statistic``, ``p-value``, ``test result``. Test result: 1 - significant different, 0 - insignificant difference.
        """
        self.__check_required_metric_type('test_buckets')

        x = self.params.data_params.control
        y = self.params.data_params.treatment

        np.random.shuffle(x)
        np.random.shuffle(y)
        x_new = np.array([self.params.hypothesis_params.metric(x)
                          for x in np.array_split(x, self.params.hypothesis_params.n_buckets)])
        y_new = np.array([self.params.hypothesis_params.metric(y)
                          for y in np.array_split(y, self.params.hypothesis_params.n_buckets)])

        test_result: int = 0
        if (shapiro(x_new).pvalue >= self.params.hypothesis_params.alpha) \
                and (shapiro(y_new).pvalue >= self.params.hypothesis_params.alpha):
            stat, pvalue = ttest_ind(y_new, x_new, equal_var=False,
                                     alternative=self.params.hypothesis_params.alternative)
            if pvalue <= self.params.hypothesis_params.alpha:
                test_result = 1
        else:
            def metric(arr: np.array):
                modes, _ = mode(arr)
                return sum(modes) / len(modes)

            self.params.hypothesis_params.metric = metric
            _, pvalue, test_result = self.test_boot_confint()

        result = {
            'stat': None,
            'p-value': pvalue,
            'result': test_result
        }
        return result

    def test_chisquare(self) -> StatTestResultType:
        """Performs Chi-Square test.

        Returns:
            stat_test_typing: Dictionary with following properties: ``test statistic``, ``p-value``, ``test result``. Test result: 1 - significant different, 0 - insignificant difference.
        """
        self.__check_required_metric_type('test_chisquare')

        x = self.__get_group(self.params.data_params.control_name, self.dataset)
        y = self.__get_group(self.params.data_params.treatment_name, self.dataset)

        if len(x) == len(y):
            observed = np.array([sum(y), len(y) - sum(y)])
            expected = np.array([sum(x), len(x) - sum(x)])
            stat, pvalue = chisquare(observed, expected)

            test_result: int = 0
            if pvalue <= self.params.hypothesis_params.alpha:
                test_result = 1

            result = {
                'stat': stat,
                'p-value': pvalue,
                'result': test_result
            }
            return result
        else:
            raise ValueError('Both groups have different lengths')


    def test_delta_ratio(self) -> StatTestResultType:
        """ Delta method with bias correction for ratios.

        Source: https://arxiv.org/pdf/1803.06336.pdf.

        Returns:
            stat_test_typing: Dictionary with following properties: ``test statistic``, ``p-value``, ``test result``. Test result: 1 - significant different, 0 - insignificant difference.
        """
        self.__check_required_metric_type('test_delta_ratio')

        x = self.__dataset[self.__dataset[self.params.data_params.group_col] == self.params.data_params.control_name]
        y = self.__dataset[self.__dataset[self.params.data_params.group_col] == self.params.data_params.treatment_name]

        ctrl_mean, ctrl_var = self.__delta_params(x)
        treat_mean, treat_var = self.__delta_params(y)

        return self.__manual_ttest(ctrl_mean, ctrl_var, x.shape[0], treat_mean, treat_var, y.shape[0])

    def test_mannwhitney(self) -> StatTestResultType:
        r"""Performs Mann-Whitney U test.

        Test works on continues metrics and their ranks.

        Assumptions of Mann-Whitney test:

        1. Independence of observations.
        2. Same shape of metric distributions.

        Statistic of the test:

        .. math::
            U = \sum_{i=1}^{n} \sum_{j=1}^{m} S(X_i, Y_j).

        Returns:
            stat_test_typing: Dictionary with following properties: ``test statistic``, ``p-value``, ``test result``. Test result: 1 - significant different, 0 - insignificant difference.
        """
        self.__check_required_metric_type('test_mannwhitney')

        x = self.params.data_params.control
        y = self.params.data_params.treatment

        test_result: int = 0
        stat, pvalue = mannwhitneyu(x, y, alternative=self.params.hypothesis_params.alternative)

        if pvalue <= self.params.hypothesis_params.alpha:
            test_result = 1

        result = {
            'stat': stat,
            'p-value': pvalue,
            'result': test_result
        }
        return result

    def test_taylor_ratio(self) -> StatTestResultType:
        """ Calculate expectation and variance of ratio for each group and then use t-test for hypothesis testing.

        Source: http://www.stat.cmu.edu/~hseltman/files/ratio.pdf.

        Returns:
            stat_test_typing: Dictionary with following properties: ``test statistic``, ``p-value``, ``test result``. Test result: 1 - significant different, 0 - insignificant difference.
        """
        self.__check_required_metric_type('test_taylor_ratio')

        x = self.__dataset[self.__dataset[self.params.data_params.group_col] == self.params.data_params.control_name]
        y = self.__dataset[self.__dataset[self.params.data_params.group_col] == self.params.data_params.treatment_name]

        ctrl_mean, ctrl_var = self.__taylor_params(x)
        treat_mean, treat_var = self.__taylor_params(y)

        return self.__manual_ttest(ctrl_mean, ctrl_var, x.shape[0], treat_mean, treat_var, y.shape[0])

    def test_welch(self) -> StatTestResultType:
        """Performs Welch's t-test.

        Returns:
            stat_test_typing: Dictionary with following properties: ``test statistic``, ``p-value``, ``test result``. Test result: 1 - significant different, 0 - insignificant difference.
        """
        self.__check_required_metric_type('test_welch')

        x = self.params.data_params.control
        y = self.params.data_params.treatment

        test_result: int = 0
        stat, pvalue = ttest_ind(y, x, equal_var=False, alternative=self.params.hypothesis_params.alternative)

        if pvalue <= self.params.hypothesis_params.alpha:
            test_result = 1

        result = {
            'stat': stat,
            'p-value': pvalue,
            'result': test_result
        }
        return result

    def test_z_proportions(self) -> StatTestResultType:
        r"""Performs z-test for proportions.

        The two-proportions z-test is used to compare two observed proportions.

        Statistic of the test:

        .. math::
            Z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})(\frac{1}{n_1} + \frac{1}{n_2})}}.

        Returns:
            stat_test_typing: Dictionary with following properties: ``test statistic``, ``p-value``, ``test result``. Test result: 1 - significant different, 0 - insignificant difference.
        """
        self.__check_required_metric_type('test_z_proportions')

        x = self.__get_group(self.params.data_params.control_name, self.dataset)
        y = self.__get_group(self.params.data_params.treatment_name, self.dataset)

        count = np.array([sum(y), sum(x)])
        nobs = np.array([len(y), len(x)])

        alternative = self.params.hypothesis_params.alternative
        if alternative != 'two-sided':
            alternative = 'smaller' if alternative == 'less' else 'larger'
        stat, pvalue = proportions_ztest(count, nobs, alternative=alternative)

        test_result: int = 0
        if pvalue <= self.params.hypothesis_params.alpha:
            test_result = 1

        result = {
            'stat': stat,
            'p-value': pvalue,
            'result': test_result
        }
        return result
