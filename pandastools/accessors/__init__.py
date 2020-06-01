# -*- coding: utf-8 -*-

"""accessors module

contains accessors for pandas
"""

from .dataframe import DataFrameAccessor
from .series import SeriesAccessor

__all__ = ["DataFrameAccessor", "SeriesAccessor"]
