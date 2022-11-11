"""Tests for `pandastools` package."""

import pandas as pd

from pandastools import accessors  # noqa


def test_uniquify_cols():
    df = pd.DataFrame(dict(a=[1, 2, 5]))
    df2 = df.pt.uniquify_columns()
    assert df.equals(df2)
    df3 = pd.concat([df, df], axis=1)
    copy = df3.copy()
    result = df3.pt.uniquify_columns()
    assert result.columns.to_list() == ["a", "a_2"]
    assert df3.equals(copy)


def test_get_info():
    df = pd.DataFrame(dict(a=[1, 2, 5]))
    df.pt.get_info()


def test_split():
    df = pd.DataFrame(dict(a=[1, 2, 5, 5, 1, 6]))
    df2 = df.pt.split(thresh=3, colname="a", extra_rows=0)
    print(df2)


def test_index_to_secs():
    date_rng = pd.date_range(start="1/1/2018", end="1/08/2018", freq="H")
    result = pd.DataFrame(index=date_rng)
    result.index.pt.index_to_secs()


def test_cleanup():
    df = pd.DataFrame(dict(a=[1, 2, 5, 5, 1, 6]))
    df.pt.cleanup()


def test_tolerance_bands():
    df = pd.DataFrame(dict(a=[1, 2, 5, 5, 1, 6]))
    df["a"].pt.tolerance_bands(window=2, pct=0.01)
    df.pt.tolerance_bands(window=2, pct=0.01)


def test_merge_columns():
    df = pd.DataFrame(dict(a=["a", "b", "c"], b=["d", "e", "f"]))
    df["c"] = df.pt.merge_columns(columns=["b", "a"], divider="/")
    print(df["c"])
