import numpy as np
import pandas as pd
from abacus.splitter.params import SplitBuilderParams
from abacus.mde_researcher.params import MdeParams
from abacus.mde_researcher.mde_research_builder import MdeResearchBuilder
from abacus.auto_ab.abtest import ABTest
from abacus.auto_ab.params import ABTestParams, DataParams, HypothesisParams


POSSIBLE_TESTS = [
    ABTest.test_boot_confint,
    ABTest.test_boot_fp,
    ABTest.test_delta_ratio,
    ABTest.test_mannwhitney,
    ABTest.test_taylor_ratio,
    ABTest.test_welch,
]


if __name__ == "__main__":
    df = pd.read_csv("examples/data/ab_data.csv")

    df["moda_city"] = np.random.randint(1, 5, df.shape[0])
    df["moda_city"] = df["moda_city"].astype(str)
    df["country"] = np.random.randint(1, 3, df.shape[0])
    df["id"] = df.index

    df["numerator"] = np.random.randint(1, 5, df.shape[0])
    df["denominator"] = np.random.randint(1, 5, df.shape[0])
    df["country"] = np.random.randint(1, 3, df.shape[0])

    data_params = DataParams(
        id_col="id",
        group_col="groups",
        control_name="control",
        treatment_name="target",
        strata_col="country",
        target="height_now",
        predictors=["weight_now"],
        numerator="clicks",
        denominator="sessions",
        covariate="height_prev",
        target_prev="height_prev",
        predictors_prev=["weight_prev"],
        is_grouped=True,
    )

    hypothesis_params = HypothesisParams(
        alpha=0.05,
        beta=0.2,
        alternative="two-sided",
        metric_type="continuous",
        metric_name="mean",
        n_boot_samples=1000,
        n_buckets=50,
    )

    ab_params = ABTestParams(data_params, hypothesis_params)

    ab_params = ABTestParams()
    ab_params.data_params.numerator = "numerator"
    ab_params.data_params.denominator = "denominator"

    split_builder_params = SplitBuilderParams(
        map_group_names_to_sizes={"control": None, "target": None},
        main_strata_col="moda_city",
        split_metric_col="height_now",
        id_col="id",
        cols=["height_prev"],
        cat_cols=["country"],
        pvalue=0.05,
        n_bins=6,
        min_cluster_size=500,
    )

    for test in POSSIBLE_TESTS:
        print(test)
        prepilot_params = MdeParams(
            metrics_names=["height_now"],
            injects=[1.0001, 1.0002, 1.0003],
            min_group_size=50000,
            max_group_size=52000,
            step=10000,
            variance_reduction=None,
            use_buckets=False,
            stat_test=test,
            iterations_number=3,
            max_beta_score=2.0,
            min_beta_score=0.0,
        )

        prepilot = MdeResearchBuilder(
            df, ab_params, prepilot_params, split_builder_params
        )
        beta, alpha = prepilot.collect()
