Experiment Initialization
=========================

Before actual analysis, you have to define your experiment.
Here is how you can do it:

.. code-block:: python

    from abacus.auto_ab.abtest import ABTest
    from abacus.auto_ab.params import ABTestParams, DataParams, HypothesisParams

    df = pd.read_csv('./data/ab_data.csv')

    data_params = DataParams(
        id_col='user_id',
        group_col='groups',
        control_name='control',
        treatment_name='treatment',
        target='check_rub_campaign',
    )

    hypothesis_params = HypothesisParams(
        alpha=0.01,
        beta=0.2,
        alternative='greater',
        metric_type='solid',
        metric_name='95th quantile',
        metric=lambda x: np.quantile(x, 0.95)
    )

    ab_params = ABTestParams(data_params, hypothesis_params)
    ab_test = ABTest(df, ab_params)

As you can see, you just need to describe data and your hypothesis.

For data, you have to define columns and their purposes. Required attributes are:

- ``id_col`` is observation id. It can be user_id or any other id for your rows. Note that if your observations are somehow dependent (e.g. several checks per user), they must have the same id_col.
- ``group_col`` contains group names. If your data have two groups, then there mush be only two unique values in this column.
- ``control_name`` and ``treatment_name`` are group names e.g. 'control', 'treatment', 'A', 'B', 'control group', 'send sms', 'do not send sms', etc.
- ``target`` is obviously target column containing metric of interest.

Hypothesis is described with:

- ``alpha`` — type I error.
- ``beta`` — type II error.
- ``alternative`` — alternative of hypothesis (two-sided, less, or greater.
- ``metric_type`` — metric type. There are three of them: continuous, binary, and ratio.
- ``metric_name`` — metric name, either default ('mean' or 'median') or customer (e.g. '95th percentile').
- ``metric`` — function for metric calculation if ``metric_name`` is not default.
