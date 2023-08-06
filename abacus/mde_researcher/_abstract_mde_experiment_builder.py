import logging
import itertools
import numpy as np
import pandas as pd
from abacus.auto_ab.params import ABTestParams
from abacus.mde_researcher.params import MdeParams

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class AbstractMdeResearchBuilder:
    """Base class for Experiment Builders.
    """
    def __init__(self,
                 guests: pd.DataFrame,
                 abtest_params: ABTestParams,
                 experiment_params: MdeParams):
        """
        Args:
            guests (pandas.DataFrame): Pandas dataframe that collected by PrepilotGuestsCollector.
            abtest_params (ABTestParams): A/B tests params. Using for experiments calculations..
            experiment_params (MdeParams): Parameters for experiments.
        """
        self.guests = guests
        self.abtest_params = abtest_params
        self.experiment_params = experiment_params
        self._group_sizes = self._build_group_sizes()

    @property
    def experiment_params(self):
        return self._experiment_params

    @experiment_params.setter
    def experiment_params(self, new_experiment_params):
        self._experiment_params = new_experiment_params
        self._group_sizes = self._build_group_sizes()

    @property
    def group_sizes(self):
        return self._group_sizes

    def _build_group_sizes(self):
        """Build list of groups sizes tuples.

        Returns:
            List[int]: List of groups sizes pairs.
        """
        control = np.sort(np.arange(self.experiment_params.min_group_size,
                                    self.experiment_params.max_group_size+1,
                                    self.experiment_params.step))
        groups_split = list()
        for el in control:
            groups_split.extend(list(itertools.product([el], [el])))
        return groups_split
