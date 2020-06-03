# -*- coding: utf-8 -*-
"""
@author: Philipp Temminghoff
"""

import pandas as pd


@pd.api.extensions.register_index_accessor("pt")
class IndexAccessor(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def index_to_secs(self):
        if self._obj.size == 0:
            logger.debug("index_to_secs failed. Index empty")
            return self._obj
        elif isinstance(self._obj, pd.DatetimeIndex):
            secs = self._obj.astype(int) / 1_000_000_000
            secs = secs - secs[0]
            return secs
        else:
            logger.debug("index_to_secs failed. No DateTimeIndex")
            return self._obj


if __name__ == "__main__":
    df = pd.DataFrame(dict(a=[1, 2, 3, 4, 5]))
    print(df["a"].pt.tolerance_bands(5, 0.01))