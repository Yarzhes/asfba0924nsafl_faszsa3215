"""Realized volatility estimators: Garman-Klass, Rogers-Satchell, Parkinson"""
from typing import Iterable
import numpy as np
import pandas as pd


def parkinson(df: pd.DataFrame, high_col: str = "high", low_col: str = "low", window: int = 30) -> pd.Series:
    H = df[high_col]
    L = df[low_col]
    rs = (np.log(H / L) ** 2) / (4 * np.log(2))
    return rs.rolling(window).mean().pipe(np.sqrt)


def garman_klass(df: pd.DataFrame, open_col: str = "open", high_col: str = "high", low_col: str = "low", close_col: str = "close", window: int = 30) -> pd.Series:
    O = df[open_col]
    H = df[high_col]
    L = df[low_col]
    C = df[close_col]
    term1 = 0.5 * (np.log(H / L) ** 2)
    term2 = (2 * np.log(2) - 1) * (np.log(C / O) ** 2)
    rk = (term1 - term2).rolling(window).mean()
    return rk.pipe(lambda s: np.sqrt(np.abs(s)))


def rogers_satchell(df: pd.DataFrame, open_col: str = "open", high_col: str = "high", low_col: str = "low", close_col: str = "close", window: int = 30) -> pd.Series:
    O = df[open_col]
    H = df[high_col]
    L = df[low_col]
    C = df[close_col]
    rs_term = (np.log(H / C) * np.log(H / O)) + (np.log(L / C) * np.log(L / O))
    rs = rs_term.rolling(window).mean()
    return rs.pipe(lambda s: np.sqrt(np.abs(s)))
