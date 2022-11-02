from __future__ import annotations

import logging
from typing import Literal

import pandas as pd


logger = logging.getLogger(__name__)


@pd.api.extensions.register_index_accessor("pt")
class IndexAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def index_to_secs(self):
        if self._obj.size == 0:
            logger.debug("index_to_secs failed. Index empty")
            return self._obj
        elif isinstance(self._obj, pd.DatetimeIndex):
            secs = self._obj.view(int) / 1_000_000_000
            secs = secs - secs[0]
            return secs
        else:
            logger.debug("index_to_secs failed. No DateTimeIndex")
            return self._obj

    def detect_gaps(self, max_diff: float | str, level=None):
        if level is None:
            index = self._obj
        else:
            index = self._obj.get_level_values(level)
        return (index.to_series().diff() > max_diff).cumsum() + 1

    def to_datetime(
        self,
        level=None,
        fmt: str | None = None,
        errors: Literal["ignore", "raise"] = "ignore",
    ):
        if isinstance(self._obj, pd.MultiIndex):
            values = self._obj.get_level_values(level).to_series()
            values = pd.to_datetime(
                values, format=fmt, errors=errors, infer_datetime_format=not bool(fmt)
            )
            return self._obj.set_levels(values, level)
        else:
            return pd.to_datetime(
                self._obj, format=fmt, errors=errors, infer_datetime_format=not bool(fmt)
            )


if __name__ == "__main__":
    df = pd.DataFrame(dict(a=[1, 2, 3, 4, 5]))
    print(df["a"].pt.tolerance_bands(5, 0.01))
