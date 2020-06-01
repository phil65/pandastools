# -*- coding: utf-8 -*-
"""
@author: Philipp Temminghoff
"""

import logging
import time
from typing import Union

import numba
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_transition_info(ds, colname, threshold=None, extra_rows=0):
    """
    calculate splits and add "secs" and "process_num" column to dataframe
    """
    logger.info("Start splitting process")
    now = time.time()
    # setze ersten Wert auf 0. (evtl auch noch den letzten?)
    # Bisschen unschön, aber nötig, damit start und end indexes in sync sind.
    col = ds[colname]
    threshold if threshold else col.max() / 2

    start_indexes = get_transition_indices(col.to_numpy(), threshold)
    start_indexes = np.maximum(start_indexes - extra_rows, 0)
    end_indexes = get_transition_indices(col.to_numpy(), threshold, True)
    end_indexes = np.minimum(end_indexes + extra_rows, len(ds.index))
    if len(start_indexes) > len(end_indexes):
        start_indexes = start_indexes[:-1]
    if len(end_indexes) > len(start_indexes):
        end_indexes = start_indexes[1:]
    if not colname or len(start_indexes) == 0 or len(end_indexes) == 0:
        start_indexes = [0]
        end_indexes = [len(ds.index)]  # perhaps -1 ?

    logger.info("Found indices. Applying new columns to dataset....")
    ds = ds.drop("secs", errors="ignore", axis=1)
    proc = np.full((len(ds.index),), np.nan)
    for i, (start, end) in enumerate(zip(start_indexes, end_indexes), start=1):
        proc[start:end] = i
    categories = [i + 1 for i in range(len(start_indexes))]
    ds["process_num"] = pd.Categorical(proc, categories=categories)
    ds["secs"] = ds.groupby("process_num").apply(calc_secs)["secs"]
    logger.info(f"Splitting took {(time.time() - now):.2f} seconds")
    return ds


def calc_secs(ds):
    idx = ds.index.astype(int).to_numpy() / 1_000_000_000
    ds["secs"] = idx - idx[0] if len(ds.index) > 0 else np.nan
    return ds


@numba.jit(nopython=True, parallel=True)
def get_transition_indices(y: np.ndarray,
                           threshold: Union[int, float],
                           falling_edge: bool = False) -> np.ndarray:
    """
    return indices where a transition occurs (default: detect rising edges)
    """
    # Find where y crosses a threshold in a specific direction.
    lower = y < threshold
    higher = y >= threshold
    if falling_edge:
        return np.where(higher[:-1] & lower[1:])[0]
    else:
        return np.where(lower[:-1] & higher[1:])[0]


@numba.jit(nopython=True, parallel=True)
def find_transition_times(t: np.ndarray,
                          y: np.ndarray,
                          threshold: Union[int, float],
                          falling_edge: bool = False) -> np.ndarray:
    """
    Given the input signal `y` with samples at times `t`,
    find the times where `y` increases through the value `threshold`.

    `t` and `y` must be 1-D numpy arrays.

    Linear interpolation is used to estimate the time `t` between
    samples at which the transitions occur.
    """

    transition_indices = get_transition_indices(y, threshold, falling_edge)

    t0 = t[transition_indices]
    t1 = t[transition_indices + 1]
    y0 = y[transition_indices]
    y1 = y[transition_indices + 1]
    slope = (y1 - y0) / (t1 - t0)
    transition_times = t0 + (threshold - y0) / slope

    return transition_times


@numba.jit(nopython=True)
def periods(t: np.ndarray, y: np.ndarray, threshold):
    """
    Given the input signal `y` with samples at times `t`,
    find the time periods between the times at which the
    signal `y` increases through the value `threshold`.

    `t` and `y` must be 1-D numpy arrays.
    periods(df.index.astype(int).to_numpy(), df.InductorDown.to_numpy(), 0.5)
    """
    transition_times = find_transition_times(t, y, threshold)
    deltas = np.diff(transition_times)
    return deltas


def sliced(df, column_name, threshold):
    t_indices = get_transition_indices(df[column_name], threshold)
    return np.split(df, t_indices)
