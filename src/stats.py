from typing import Callable, List, Tuple
import numpy as np
import pandas as pd
from functools import cache


class Stats:
    """
    A class to perform statistical analysis on a DataFrame of financial data.

    Attributes:
        df (pd.DataFrame): The input DataFrame containing financial data.
        df_pct_changes (pd.DataFrame): The percentage changes of the assets.
        final_return (pd.Series): The final return of each asset in the DataFrame.
        avg_return (float): The average return of the assets.

    Methods:
        ratios: Computes the squared ratios of non-zero final returns.
        return_matrix: Generates a matrix of returns based on rolling periods.
        weighted_return_series: Computes a weighted return series.
        weighted_mean_std: Calculates the weighted mean and standard deviation of returns.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the Stats class with a DataFrame and computes initial statistics.

        Args:
            df (pd.DataFrame): The input DataFrame containing financial data.
        """
        self.df_values = df
        self.df_pct_changes = df.iloc[:, 1:].fillna(1).pct_change(fill_method=None).dropna()
        self.final_return = df.iloc[-1, 1:] - 1
        self.df_values.fillna(1, inplace=True)
        with pd.option_context("future.no_silent_downcasting", True):
            self.final_return = self.final_return.fillna(0).infer_objects(copy=False)
        self.avg_return = self.final_return.mean()

    def ratios(self, transformation: Callable[[float], float] = lambda x: np.exp(x)) -> List[float]:
        """
        Computes the squared ratios of non-zero final returns.

        Args:
            transformation (Callable[[float], float]): A function to transform the returns.
                                                     Defaults to exponential function.

        Returns:
            List[float]: A list of squared ratios.
        """
        ss = self.final_return.apply(transformation).abs().sum()
        ratios = [transformation(x) / ss for x in self.final_return]
        return ratios
