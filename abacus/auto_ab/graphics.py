from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abacus.auto_ab.params import ABTestParams
from abacus.types import ArrayNumType


class Graphics:
    """Illustration of an experiment.

    - As it is easier to apply plotting directly to experiment, all methods should be called on ``ABTest`` class instance.
    - Experiment's plot is based on metric type.

    Example:

    .. code-block:: python

        from abacus.auto_ab.abtest import ABTest
        from abacus.auto_ab.params import ABTestParams, DataParams, HypothesisParams

        data_params = DataParams(...)
        hypothesis_params = HypothesisParams(...)
        ab_params = ABTestParams(data_params, hypothesis_params)

        df = pd.read_csv('data.csv')
        ab_test = ABTest(df, ab_params)
        ab_test.plot()
    """

    def __init__(self) -> None:
        pass

    @classmethod
    def plot_continuous_experiment(
        cls, params: ABTestParams, save_path: Optional[str]
    ) -> None:
        """Plot distributions of continuous metric and actual experiment metric.

        Args:
            params (ABTestParams): Parameters of the experiment.
            save_path (str, optional): Path where to save image.
        """
        bins = 300
        ctrl_mean: float = params.hypothesis_params.metric(params.data_params.control)
        trtm_mean: float = params.hypothesis_params.metric(params.data_params.treatment)
        metric_name = params.hypothesis_params.metric_name

        control_dist_height, _ = np.histogram(params.data_params.control, bins=bins)
        treatment_dist_height, _ = np.histogram(params.data_params.treatment, bins=bins)

        x_label = params.data_params.target
        if "metric transform" in params.data_params.transforms:
            x_label = (
                params.hypothesis_params.metric_transform.__name__
                + "("
                + params.data_params.target
                + ")"
            )

        plt.rcParams.update({"font.size": 14})
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.hist(
            params.data_params.control, bins, alpha=0.5, label="Control", color="Red"
        )
        ax.hist(
            params.data_params.treatment,
            bins,
            alpha=0.5,
            label="Treatment",
            color="Blue",
        )
        ax.axvline(
            x=ctrl_mean, linewidth=2, color="Red", label=f"Control {metric_name}"
        )
        ax.axvline(
            x=trtm_mean, linewidth=2, color="Blue", label=f"Treatment {metric_name}"
        )
        ax.legend()
        ax.set_xlabel(x_label)

        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
        plt.close()

    @classmethod
    def plot_bootstrap_confint(
        cls, params: ABTestParams, save_path: Optional[str]
    ) -> None:
        """Plot bootstrapped metric of an experiment with its confidence
        interval and zero value.

        Args:
            x (np.ndarray): Bootstrap metric.
            params (ABTestParams): Parameters of the experiment.
        """
        bootstrap_diffs = []
        control_len = len(params.data_params.control)
        treatment_len = len(params.data_params.treatment)
        metric = params.hypothesis_params.metric
        for _ in range(params.hypothesis_params.n_boot_samples):
            boot_control = np.random.choice(
                params.data_params.control, size=control_len, replace=True
            )
            boot_treatment = np.random.choice(
                params.data_params.treatment, size=treatment_len, replace=True
            )
            boot_diff = metric(boot_treatment) - metric(boot_control)
            bootstrap_diffs.append(boot_diff)

        x_label = params.data_params.target
        bins: int = params.hypothesis_params.n_boot_samples // 10
        max_height = max(np.histogram(bootstrap_diffs, bins)[0])
        ci_left, ci_right = np.quantile(
            bootstrap_diffs, params.hypothesis_params.alpha / 2
        ), np.quantile(bootstrap_diffs, 1 - params.hypothesis_params.alpha / 2)
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.hist(
            bootstrap_diffs, bins, alpha=0.5, label="Differences in metric", color="Red"
        )
        ax.axvline(x=0, color="Red", label="No difference")
        ax.vlines(
            [ci_left, ci_right],
            ymin=0,
            ymax=max_height,
            linestyle="--",
            label="Confidence interval",
        )
        ax.legend()
        ax.set_xlabel("effect on " + x_label)

        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
        plt.close()

    @classmethod
    def plot_binary_experiment(
        cls, params: ABTestParams, save_path: Optional[str]
    ) -> None:
        """Plot experiment with binary outcome.

        Args:
            params (ABTestParams): Parameters of the experiment.
            save_path (str, optional): Path where to save image.
        """
        x = params.data_params.control
        y = params.data_params.treatment

        ctrl_total = len(x)
        ctrl_conv = sum(x)
        ctrl_not_conv = ctrl_total - ctrl_conv
        ctrl_conv_share = round(ctrl_conv / ctrl_total * 100, 2)
        ctrl_not_conv_share = round(ctrl_not_conv / ctrl_total * 100, 2)
        treat_total = len(y)
        treat_conv = sum(y)
        treat_not_conv = treat_total - treat_conv
        treat_conv_share = round(treat_conv / treat_total * 100, 2)
        treat_not_conv_share = round(treat_not_conv / treat_total * 100, 2)

        df = pd.DataFrame(
            data={
                "Experiment group": ["control", "control", "treatment", "treatment"],
                "is converted": ["no", "yes", "no", "yes"],
                "Number of observations": [
                    ctrl_not_conv,
                    ctrl_conv,
                    treat_not_conv,
                    treat_conv,
                ],
                "Share": [
                    ctrl_not_conv_share,
                    ctrl_conv_share,
                    treat_not_conv_share,
                    treat_conv_share,
                ],
            }
        )
        shares = [
            ctrl_not_conv_share,
            treat_not_conv_share,
            ctrl_conv_share,
            treat_conv_share,
        ]

        plt.figure(figsize=(20, 12), dpi=300)
        ax = sns.barplot(
            data=df,
            x="Experiment group",
            y="Number of observations",
            hue="is converted",
        )
        patches = ax.patches
        for i in range(len(patches)):
            x = patches[i].get_x() + patches[i].get_width() / 2
            y = patches[i].get_height() + 0.05
            ax.annotate("{:.1f}%".format(shares[i]), (x, y), ha="center")

        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
        plt.close()
