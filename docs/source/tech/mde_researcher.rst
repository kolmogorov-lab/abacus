MDE Researcher
==============

.. currentmodule:: abacus.mde_researcher

.. autosummary::
    :nosignatures:

    AbstractMdeResearchBuilder
    BaseSplitElement
    MdeAlphaExperiment
    MdeBetaExperiment
    MdeResearchBuilder
    MultipleSplitBuilder
    MdeParams

Abstract MDE Experiment
-----------------------


.. autoclass:: abacus.mde_researcher.AbstractMdeResearchBuilder

.. autofunction:: abacus.mde_researcher.AbstractMdeResearchBuilder._build_group_sizes

Experiment Structures
---------------------

.. autoclass:: abacus.mde_researcher.BaseSplitElement
.. autoclass:: abacus.mde_researcher.MdeAlphaExperiment
.. autoclass:: abacus.mde_researcher.MdeBetaExperiment

MDE Research Builder
--------------------

.. autoclass:: abacus.mde_researcher.MdeResearchBuilder

.. autofunction:: abacus.mde_researcher.MdeResearchBuilder.calc_alpha
.. autofunction:: abacus.mde_researcher.MdeResearchBuilder.collect

Multiple Split Builder
----------------------

.. autoclass:: abacus.mde_researcher.MultipleSplitBuilder

.. autofunction:: abacus.mde_researcher.MultipleSplitBuilder._build_splits_grid
.. autofunction:: abacus.mde_researcher.MultipleSplitBuilder._update_strat_params
.. autofunction:: abacus.mde_researcher.MultipleSplitBuilder._build_split
.. autofunction:: abacus.mde_researcher.MultipleSplitBuilder.calc_injected_metrics
.. autofunction:: abacus.mde_researcher.MultipleSplitBuilder.collect

Params
------

.. autoclass:: abacus.mde_researcher.MdeParams
