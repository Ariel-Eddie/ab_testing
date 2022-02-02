import attr
import numpy as np
import pandas as pd
import statsmodels.stats.api as sms
from enum import Enum
from typing import Union
from scipy.stats import norm
from statsmodels.stats.weightstats import ztest
from ab_testing.confidence_interval import ConfidenceInterval


class Method(str, Enum):
    hand_made = "hand_made"
    auto = 'auto'


class TestAlternative(str, Enum):
    larger = "larger"
    smaller = "smaller"


@attr.s(auto_attribs=True)
class TraditionalAbTesting:

    """
    ALL TESTS IN THIS CLASS ARE MADE WITH A SIGNIFICANCE THRESHOLD OF 5%
    MEANING IN 5% OF CASES WE WILL REJECT THE NULL HYPOTHESIS EVEN IF IT IS REALLY TRUE

    WE ASSUME THAT THE OBSERVATIONS ARE v.a.i.i.d
    ELSE USE CHI-SQUARE TEST

    THIS TRADITIONAL ab_testing IS NOT SUITABLE FOR ONLINE SITUATION
    USE BAYESIAN APPROACH INSTEAD

    H0 : No difference between the two mean estimators u0 = u1
    H1 : Statistically significant difference between the two mean : u0 # u1

    Goal : Reject H0 meaning the z statistic we observed is very unlikely and the test is statistically valid
    5% of false alarm

    CONSIDERATIONS :
    size effect
    variance effect

    """

    default_N1 = 100
    default_mu1 = .2
    default_sigma1 = 1
    default_x1 = np.random.randn(default_N1)*default_sigma1 + default_mu1

    default_N2 = 100
    default_mu2 = .5
    default_sigma2 = 1
    default_x2 = np.random.randn(default_N2)*default_sigma2 + default_mu2

    @staticmethod
    def test_power(current_win_rate: float,
                   expected_win_rate: float,
                   power: float = 0.95) -> int:

        """
        :param current_win_rate:
        :param expected_win_rate:
        :param power: assuming the difference is the one we estimated (current_win_rate vs.  expected_win_rate),
        we have about power% chance to detect it as statistically significant in our test with the sample size
        :return: sample size
        """

        effect_size = sms.proportion_effectsize(current_win_rate, expected_win_rate)
        required_size = sms.NormalIndPower().solve_power(
            effect_size,
            power=power,
            alpha=0.05,  # SIGNIFICANCE THRESHOLD OF 5%
            ratio=1
        )
        return int(required_size)

    @staticmethod
    def one_sample_two_sided_z_test(x: Union[np.ndarray, pd.Series],
                                    ref_value: float = 0,
                                    method: Method = Method.hand_made):

        # auto test with statsmodels
        if method == Method.auto:
            return ztest(x, value=ref_value)

        elif method == Method.hand_made:
            # estimators
            mu_hat = x.mean()
            sigma_hat = x.std(ddof=1)
            # Standardized mean estimator under the null hypothesis
            z = (mu_hat - ref_value) / (sigma_hat / np.sqrt(len(x)))  # our mu0 = 0
            p_right = 1 - norm.cdf(np.abs(z))
            p_left = norm.cdf(-np.abs(z))
            p = p_right + p_left
            return z, p

    @staticmethod
    def one_sample_one_sided_z_test(x: Union[np.ndarray, pd.Series],
                                    alternative: TestAlternative,
                                    ref_value: float = 0,
                                    method: Method = Method.hand_made):

        # auto test with statsmodels
        if method == Method.auto:
            return ztest(x, alternative=alternative, value=ref_value)

        elif method == Method.hand_made:
            # estimators
            mu_hat = x.mean()
            sigma_hat = x.std(ddof=1)
            # Standardized mean estimator under the null hypothesis
            z = (mu_hat - ref_value) / (sigma_hat / np.sqrt(len(x)))  # our mu0 = 0
            if alternative == TestAlternative.larger:
                p = 1 - norm.cdf(z)
            else:
                p = norm.cdf(z)
            return z, p

    @staticmethod
    def two_sample_two_sided_z_test(x1: Union[np.ndarray, pd.Series],
                                    x2: Union[np.ndarray, pd.Series],
                                    method: Method = Method.hand_made):

        # auto test with statsmodels
        if method == Method.auto:
            return ztest(x1, x2)

        elif method == Method.hand_made:
            mu_hat1 = x1.mean()
            mu_hat2 = x2.mean()
            dmu_hat = mu_hat2 - mu_hat1
            # Variance not std because the var is directly needed in the next step
            s2_hat1 = x1.var(ddof=1)
            s2_hat2 = x2.var(ddof=1)
            s_hat = np.sqrt(s2_hat1 / len(x1) + s2_hat2 / len(x2))
            z = dmu_hat / s_hat  # reference value is 0
            p_right = 1 - norm.cdf(np.abs(z))
            p_left = norm.cdf(-np.abs(z))
            p = p_right + p_left
            return z, p

    def test_and_interpret(self, x1: Union[np.ndarray, pd.Series], x2: Union[np.ndarray, pd.Series]):

        z, p = self.two_sample_two_sided_z_test(x1, x2, method=Method.auto)
        test = p < 0.05
        if test:
            description = "Null hypothesis rejected, there is a significant different between the two groups."
        else:
            description = "Not enough evidence to reject the null hypothesis."

        mu_hat1, lci1, rci1 = ConfidenceInterval().z_confidence_interval(x1)
        mu_hat2, lci2, rci2 = ConfidenceInterval().z_confidence_interval(x2)

        return {
            "z": z,
            "p": p,
            "description": description,
            "confidence_interval_group1": [round(lci1, 6), round(rci1, 6)],
            "confidence_interval_group2": [round(lci2, 6), round(rci2, 6)]
        }

    def test_experiments(self):
        res = {
            "two_sided": self.one_sample_two_sided_z_test(self.default_x1, method=Method.auto),
            "two_sided_hm": self.one_sample_two_sided_z_test(self.default_x1),
            "one_sided": self.one_sample_one_sided_z_test(self.default_x1, alternative=TestAlternative.smaller, method=Method.auto),
            "one_sided_hm":  self.one_sample_one_sided_z_test(self.default_x1, alternative=TestAlternative.smaller),
            "two_sample_two_sided": self.two_sample_two_sided_z_test(self.default_x1, self.default_x2, method=Method.auto),
            "two_sample_two_sided_hm": self.two_sample_two_sided_z_test(self.default_x1, self.default_x2)
        }

        data1 = pd.read_csv("../data/ab_data.csv")
        data1 = data1.drop_duplicates(subset=['user_id'])

        required_n = self.test_power(.11, .13)
        x1 = data1[data1['group'] == 'control']['converted']
        x2 = data1[data1['group'] == 'treatment']['converted']

        x1 = x1.sample(n=required_n, random_state=123)
        x2 = x2.sample(n=required_n, random_state=123)
        rr1 = self.test_and_interpret(x1, x2)

        data2 = pd.read_csv("../data/advertisement_clicks.csv")
        x3 = data2[data2['advertisement_id'] == 'A']['action']
        x4 = data2[data2['advertisement_id'] == 'B']['action']
        rr2 = self.test_and_interpret(x3, x4)

        return res, rr1, rr2


if __name__ == '__main__':

    tester = TraditionalAbTesting()
    r, rr, rrr = tester.test_experiments()
    exit()
