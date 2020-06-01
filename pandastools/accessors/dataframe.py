# -*- coding: utf-8 -*-
"""
@author: Philipp Temminghoff
"""

import io

import numpy as np
import pandas as pd

from pandastools.utils import dataimport


@pd.api.extensions.register_dataframe_accessor("pt")
class DataFrameAccessor(object):
    def __init__(self, parent):
        self._obj = parent

    def iat(self, start=None, stop=None, step=1):
        return self._obj.iloc[start:stop:step]

    def get_info(self, null_counts=True) -> str:
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

    def split(self, thresh, colname, extra_rows):
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


if __name__ == "__main__":
    df = pd.DataFrame()
    df.pp.uniquify_columns()
