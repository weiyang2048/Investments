from typing import List, Tuple
import numpy as np
import pandas as pd
from functools import cache


class Stats:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.final_return = df.iloc[-1, 1:] - 1
        self.df.fillna(1, inplace=True)
        self.final_return = self.final_return.replace(np.inf, 0)
        self.final_return = self.final_return.replace(np.nan, 0)
        self.avg_return = self.final_return.mean()

    @property
    def ratios(self) -> List[float]:
        none_zero_ss = self.final_return[self.final_return > 0].apply(lambda x: x**2).sum()
        ratios = [x**2 / none_zero_ss if x > 0 else 0 for x in self.final_return]
        return ratios

    @property
    def alternative_return(self) -> float:
        return np.dot(self.final_return, self.ratios)

    @property
    def return_matrix(self) -> pd.DataFrame:
        return (
            self.df.iloc[:, 1:]
            .rolling(2)
            .apply(lambda x: x.iloc[1] / x.iloc[0] - 1)
            .dropna()
            .reset_index(drop=True)
        )

    def weighted_return_series(self, weights: List[float] = None) -> pd.Series:
        if weights is None:
            weights = np.ones(self.df.shape[1] - 1) / (self.df.shape[1] - 1)
        return self.return_matrix.dot(weights)

    def weighted_mean_std(self, weights: List[float] = None) -> Tuple[float, float]:
        return (
            self.weighted_return_series(weights).mean(numeric_only=True, skipna=True),
            self.weighted_return_series(weights).std(numeric_only=True, skipna=True),
        )
