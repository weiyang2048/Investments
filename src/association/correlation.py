import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st


def marchenko_pastur(n: int, p: int) -> float:
    return np.sqrt((1 + (2 * (1 / n)) - np.sqrt((1 + (4 * (1 / n)) ** 2 - 16 * (1 / n) * (p / n)))) / (2 * (1 / n)))


def denoise_corr_marchenko_pastur(corr: pd.DataFrame, q: float = None) -> pd.DataFrame:
    """
    Denoise a correlation matrix using the Marchenko-Pastur method.

    Args:
        corr (pd.DataFrame): The raw correlation matrix.
        q (float, optional): The ratio T/N, where T is the number of observations and N is the number of variables.
                             If None, will attempt to infer from the matrix shape.

    Returns:
        pd.DataFrame: The denoised correlation matrix.
    """

    eigvals, eigvecs = np.linalg.eigh(corr.values)
    n = corr.shape[0]
    if q is None:
        q = 2.0

    lambda_plus = (1 + np.sqrt(1.0 / q)) ** 2
    eigvals_denoised = np.where(eigvals > lambda_plus, eigvals, np.mean(eigvals[eigvals <= lambda_plus]))
    corr_denoised = eigvecs @ np.diag(eigvals_denoised) @ eigvecs.T
    corr_denoised = (corr_denoised + corr_denoised.T) / 2
    np.fill_diagonal(corr_denoised, 1.0)
    return pd.DataFrame(corr_denoised, index=corr.index, columns=corr.columns)


def pivoted_to_corr(df: pd.DataFrame, plot: bool = False, streamlit: bool = False, marchenko_pastur: bool = True) -> pd.DataFrame:
    df_pivot = df.copy()
    df_pivot.drop(columns=["Date"], inplace=True)
    df_pivot.dropna(axis=0, inplace=True)
    df_pivot = df_pivot.pct_change()
    corr_matrix = df_pivot.corr()
    if marchenko_pastur:
        corr_matrix = denoise_corr_marchenko_pastur(corr_matrix)
    corr_matrix = corr_matrix * 10
    if plot:
        mask = np.eye(corr_matrix.shape[0], dtype=bool)
        annot_matrix = corr_matrix.round(0).astype(int).astype(str)
        annot_matrix = annot_matrix.where(~mask, "")
        fig = sns.clustermap(
            corr_matrix,
            center=corr_matrix.median().median(),
            cmap="RdYlGn",
            linewidths=0.5,
            figsize=(7, 7),
            annot=annot_matrix,
            fmt="s",
            cbar_kws={"label": "Correlation"},
        )
        if streamlit:
            rows = st.columns([1, 10, 1])
            with rows[1]:
                st.pyplot(fig, use_container_width=False)
    return corr_matrix
