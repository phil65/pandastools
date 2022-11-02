from __future__ import annotations

import numpy as np
import pandas as pd


SECONDS_PER_YEAR = 31_557_600


@pd.api.extensions.register_series_accessor("pt")
class SeriesAccessor:
    def __init__(self, pandas_obj: pd.Series):
        self._obj = pandas_obj

    def downcast(self):
        s = self._obj
        if s.dtype == "float64":
            all_int = np.all(np.mod(s.to_numpy(), 1) == 0)
            dc_type = "integer" if all_int else "float"
        elif s.dtype == "int64":
            dc_type = "integer"
        else:
            return self._obj
        s = pd.to_numeric(s, downcast=dc_type)
        return self._obj

    def tolerance_bands(self, window, pct):
        series = self._obj
        rolling = series.rolling(window=window, center=True, min_periods=1)
        u_band = np.maximum(rolling.max(), series * (pct + 1))
        l_band = np.minimum(rolling.min(), series * (1 - pct))
        result = pd.concat([series, l_band, u_band], axis=1)
        cols = [f"{series.name}_{x}" for x in ["real", "lower", "upper"]]
        result.columns = cols
        return result

    def float_year_to_datetime(self):
        year = self._obj.astype(int)
        secs = (self._obj - year) * SECONDS_PER_YEAR
        return pd.to_datetime(year, format="%Y") + pd.to_timedelta(secs, unit="s")

    def cut(
        self,
        bins,
        right=True,
        labels=None,
        retbins=False,
        precision=3,
        include_lowest=False,
        duplicates="raise",
        ordered=True,
    ):
        return pd.cut(
            self._obj,
            bins=bins,
            right=right,
            labels=labels,
            retbins=retbins,
            precision=precision,
            include_lowest=include_lowest,
            duplicates=duplicates,
            ordered=ordered,
        )


if __name__ == "__main__":
    df = pd.DataFrame(dict(a=[1, 2, 3, 4, 5]))
    print(df["a"].pt.tolerance_bands(5, 0.01))
