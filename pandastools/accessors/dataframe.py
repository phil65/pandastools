# -*- coding: utf-8 -*-
"""
@author: Philipp Temminghoff
"""

import io
import logging

import numpy as np
import pandas as pd

from pandastools.utils import dataimport, helpers

logger = logging.getLogger(__name__)


@pd.api.extensions.register_dataframe_accessor("pt")
class DataFrameAccessor(object):
    def __init__(self, parent):
        self._obj = parent

    def iat(self, start=None, stop=None, step=1):
        return self._obj.iloc[start:stop:step]

    def get_info(self, null_counts: bool = True) -> str:
        buf = io.StringIO()
        self._obj.info(buf=buf, null_counts=null_counts)
        return buf.getvalue()

    def uniquify_columns(self):
        diff = len(self._obj.columns) - len(set(self._obj.columns))  # type: ignore
        if diff == 0:
            return self._obj
        seen = set()
        new = list()
        for item in self._obj.columns:  # type: ignore
            fudge = 1
            newitem = str(item)
            while newitem in seen:
                fudge += 1
                newitem = "{}_{}".format(item, fudge)
            seen.add(newitem)
            new.append(newitem)
        self._obj.columns = new
        return self._obj

    def convert_dtypes(self, old_type, new_dtype):
        cols = self._obj.select_dtypes([old_type]).columns
        self._obj[cols] = (self._obj.select_dtypes([old_type])
                           .apply(lambda x: x.astype(new_dtype)))
        return self._obj

    def split(self, thresh, colname, extra_rows: int = 0):
        """
        split dataframe into separate chunks based on supplied criteria
        """
        # split processes based on bool value (just process when no bool split)
        ds = self._obj.drop("secs", errors="ignore", axis=1)
        array = np.full((len(ds.index),), np.nan)
        ds["process_num"] = pd.Categorical(array)
        ds = dataimport.add_transition_info(ds=ds,
                                            colname=colname,
                                            threshold=thresh,
                                            extra_rows=extra_rows)
        return ds

    def index_to_secs(self):
        if self._obj.empty:
            logger.debug("index_to_secs failed. Dataframe empty")
            return self._obj
        elif isinstance(self._obj.index, pd.DatetimeIndex):
            secs = self._obj.index.astype(int) / 1_000_000_000
            ds = self._obj.assign(secs=secs - secs[0]).set_index("secs", drop=True)
            return ds
        else:
            logger.debug("index_to_secs failed. No DateTimeIndex")
            return self._obj

    def cleanup(self):
        ds = self._obj.infer_objects()
        cols = ds.select_dtypes(["object"]).columns
        ds[cols] = ds.select_dtypes(["object"]).apply(lambda x: x.astype("category"))
        return ds

    def tolerance_bands(self, window, pct):
        df = self._obj
        rolling = df.rolling(window=window, center=True, min_periods=1)
        u_band = np.maximum(rolling.max(), df * (pct + 1))
        l_band = np.minimum(rolling.min(), df * (1 - pct))
        result = pd.concat([df, l_band, u_band], axis=1)
        cols = [f"{x}_{y}" for x in ["real", "lower", "upper"] for y in df.columns]
        result.columns = cols
        return result

    def merge_columns(self, columns, divider=" "):

        def split(x):
            return divider.join(str(i) for i in x)

        return self._obj[columns].apply(split, axis=1)

    def duplicate_features(self, columns):
        df = self._obj
        new_names = helpers.uniquify_names(columns, df.columns)
        new_cols = df[columns]
        new_cols.columns = new_names
        df = pd.concat([df, new_cols], axis=1)
        return df


if __name__ == "__main__":
    test = pd.DataFrame()
    test.pt.uniquify_columns()
