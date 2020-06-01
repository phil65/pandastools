# -*- coding: utf-8 -*-
"""
@author: Philipp Temminghoff
"""

import numpy as np
import pandas as pd


@pd.api.extensions.register_series_accessor("pt")
class SeriesAccessor(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def downcast(self):
        s = self._obj
        prev_dtype = s.dtype
        if s.dtype == "float64":
            all_int = np.all(np.mod(s.to_numpy(), 1) == 0)
            dc_type = "integer" if all_int else "float"
        elif s.dtype == "int64":
            dc_type = "integer"
        else:
            return self._obj
        s = pd.to_numeric(s, downcast=dc_type)
        return self._obj
