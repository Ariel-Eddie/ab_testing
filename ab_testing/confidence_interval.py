import attr

import numpy as np
import pandas as pd

from typing import Union
from scipy.stats import norm, t


@attr.s(auto_attribs=True)
class ConfidenceInterval:

    @staticmethod
    def z_confidence_interval(x: Union[np.ndarray, pd.Series]):  #  -> Tuple[float, float, float]:

        mu_hat = np.mean(x)
        # unbiased std
        sigma_hat = np.std(x, ddof=1)
        z_left = norm.ppf(0.025)
        z_right = norm.ppf(0.975)
        lower = mu_hat + z_left * sigma_hat / np.sqrt(len(x))
        upper = mu_hat + z_right * sigma_hat / np.sqrt(len(x))
        return mu_hat, lower, upper

    ## If we don't really know the standard deviation of the sample
    @staticmethod
    def t_confidence_interval(x: Union[np.ndarray, pd.Series]):  #  -> Tuple[float, float, float]:

        mu_hat = np.mean(x)
        # unbiased std
        sigma_hat = np.std(x, ddof=1)
        t_left = t.ppf(0.025, df=len(x) - 1)
        t_right = t.ppf(0.975, df=len(x) - 1)
        lower = mu_hat + t_left * sigma_hat / np.sqrt(len(x))
        upper = mu_hat + t_right * sigma_hat / np.sqrt(len(x))
        return mu_hat, lower, upper
