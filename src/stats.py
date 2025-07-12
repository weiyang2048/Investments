from typing import List, Tuple
import numpy as np
import pandas as pd
from functools import cache


class Stats:
    """
    A class to perform statistical analysis on a DataFrame of financial data.

    Attributes:
        df (pd.DataFrame): The input DataFrame containing financial data.
        final_return (pd.Series): The final return of each asset in the DataFrame.
        avg_return (float): The average return of the assets.

    Methods:
        ratios: Computes the squared ratios of non-zero final returns.
        alternative_return: Calculates the alternative return using the computed ratios.
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
        self.df = df
        self.final_return = df.iloc[-1, 1:] - 1
        self.df.fillna(1, inplace=True)
        with pd.option_context("future.no_silent_downcasting", True):
            self.final_return = self.final_return.fillna(0).infer_objects(copy=False)
        self.avg_return = self.final_return.mean()

    @property
    def ratios(self) -> List[float]:
        """
        Computes the squared ratios of non-zero final returns.

        Returns:
            List[float]: A list of squared ratios.
        """
        none_zero_ss = self.final_return[self.final_return > 0].apply(lambda x: x**2).sum()
        ratios = [x**2 / none_zero_ss if x > 0 else 0 for x in self.final_return]
        return ratios
    
    def alternative_return(self, ratios: List[float] = None) -> float:
        """
        Calculate the alternative return using the computed ratios.

        This method computes the alternative return by taking the dot product of the final returns
        and the provided ratios. If no ratios are provided, it defaults to using the precomputed
        ratios from the `ratios` property.

        Args:
            ratios (List[float], optional): A list of ratios to use for the calculation. If not provided,
                                            the method will use the default ratios.

        Returns:
            float: The calculated alternative return value.
        """
        if ratios is None:
            ratios = self.ratios
        return np.dot(self.final_return, ratios)

    @property
    def return_matrix(self) -> pd.DataFrame:
        """
        Generates a matrix of returns based on rolling periods.

        Returns:
            pd.DataFrame: A DataFrame representing the return matrix.
        """
        return (
            self.df.iloc[:, 1:]
            .rolling(2)
            .apply(lambda x: x.iloc[1] / x.iloc[0] - 1)
            .dropna()
            .reset_index(drop=True)
        )

    def weighted_return_series(self, weights: List[float] = None) -> pd.Series:
        """
        Computes a weighted return series.

        Args:
            weights (List[float], optional): A list of weights for the assets. Defaults to equal weights.

        Returns:
            pd.Series: A series representing the weighted returns.
        """
        if weights is None:
            weights = np.ones(self.df.shape[1] - 1) / (self.df.shape[1] - 1)
        return self.return_matrix.dot(weights)

    def weighted_mean_std(self, weights: List[float] = None) -> Tuple[float, float]:
        """
        Calculates the weighted mean and standard deviation of returns.

        Args:
            weights (List[float], optional): A list of weights for the assets. Defaults to equal weights.

        Returns:
            Tuple[float, float]: A tuple containing the weighted mean and standard deviation.
        """
        return (
            self.weighted_return_series(weights).mean(numeric_only=True, skipna=True),
            self.weighted_return_series(weights).std(numeric_only=True, skipna=True),
        )
