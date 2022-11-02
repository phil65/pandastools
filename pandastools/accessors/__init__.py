"""contains accessors for pandas."""

from .dataframe import DataFrameAccessor
from .index import IndexAccessor
from .series import SeriesAccessor

__all__ = ["DataFrameAccessor", "SeriesAccessor", "IndexAccessor", "scipydataframe"]
