Auto A/B
========

.. currentmodule:: abacus.auto_ab

ABTest
------

.. autoclass:: abacus.auto_ab.ABTest

.. autosummary::
    :nosignatures:

    ABTest
    ABTest.bucketing
    ABTest.cupac
    ABTest.cuped
    ABTest.linearization
    ABTest.plot
    ABTest.resplit_df
    ABTest.test_boot_confint
    ABTest.test_boot_fp
    ABTest.test_boot_ratio
    ABTest.test_boot_welch
    ABTest.test_buckets
    ABTest.test_chisquare
    ABTest.test_delta_ratio
    ABTest.test_mannwhitney
    ABTest.test_taylor_ratio
    ABTest.test_welch
    ABTest.test_z_proportions

.. autofunction:: abacus.auto_ab.ABTest.__bucketize
.. autofunction:: abacus.auto_ab.ABTest.__check_required_columns
.. autofunction:: abacus.auto_ab.ABTest.__get_group
.. autofunction:: abacus.auto_ab.ABTest.__delta_params
.. autofunction:: abacus.auto_ab.ABTest.__manual_ttest
.. autofunction:: abacus.auto_ab.ABTest.__taylor_params

.. autofunction:: abacus.auto_ab.ABTest.bucketing
.. autofunction:: abacus.auto_ab.ABTest.cupac
.. autofunction:: abacus.auto_ab.ABTest.cuped
.. autofunction:: abacus.auto_ab.ABTest.linearization
.. autofunction:: abacus.auto_ab.ABTest.plot
.. autofunction:: abacus.auto_ab.ABTest.resplit_df
.. autofunction:: abacus.auto_ab.ABTest.test_boot_confint
.. autofunction:: abacus.auto_ab.ABTest.test_boot_fp
.. autofunction:: abacus.auto_ab.ABTest.test_boot_ratio
.. autofunction:: abacus.auto_ab.ABTest.test_boot_welch
.. autofunction:: abacus.auto_ab.ABTest.test_buckets
.. autofunction:: abacus.auto_ab.ABTest.test_chisquare
.. autofunction:: abacus.auto_ab.ABTest.test_delta_ratio
.. autofunction:: abacus.auto_ab.ABTest.test_mannwhitney
.. autofunction:: abacus.auto_ab.ABTest.test_taylor_ratio
.. autofunction:: abacus.auto_ab.ABTest.test_welch
.. autofunction:: abacus.auto_ab.ABTest.test_z_proportions

|

VarianceReduction
-----------------

.. autoclass:: abacus.auto_ab.VarianceReduction

.. autofunction:: abacus.auto_ab.VarianceReduction._target_encoding
.. autofunction:: abacus.auto_ab.VarianceReduction._predict_target
.. autofunction:: abacus.auto_ab.VarianceReduction.cuped
.. autofunction:: abacus.auto_ab.VarianceReduction.cupac

|

Graphics
--------

.. autoclass:: abacus.auto_ab.Graphics

.. autofunction:: abacus.auto_ab.Graphics.plot_continuous_experiment
.. autofunction:: abacus.auto_ab.Graphics.plot_binary_experiment
.. autofunction:: abacus.auto_ab.Graphics.plot_bootstrap_confint

|

Params
------

.. autoclass:: abacus.auto_ab.DataParams
.. autoclass:: abacus.auto_ab.HypothesisParams
.. autoclass:: abacus.auto_ab.ABTestParams
