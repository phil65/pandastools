#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `pandastools` package."""

import pandas as pd
import pytest

from pandastools import accessors


def test_uniquify_cols():
    df = pd.DataFrame(dict(a=[1, 2, 5]))
    df2 = df.pt.uniquify_columns()
    assert df.equals(df2)
    df3 = pd.concat([df, df], axis=1)
    df3.pt.uniquify_columns()
    assert df3.columns.to_list() == ["a", "a_2"]


def test_get_info():
    df = pd.DataFrame(dict(a=[1, 2, 5]))
    result = df.pt.get_info()


def test_split():
    df = pd.DataFrame(dict(a=[1, 2, 5, 5, 1, 6]))
    df2 = df.pt.split(thresh=3, colname="a", extra_rows=0)
    print(df2)


def test_index_to_secs():
    df = pd.DataFrame(dict(a=[1, 2, 5, 5, 1, 6]))


def test_cleanup():
    df = pd.DataFrame(dict(a=[1, 2, 5, 5, 1, 6]))
    df.pt.cleanup()
