# -*- coding: utf-8 -*-
"""
@author: Philipp Temminghoff
"""

import itertools


def uniquify_names(new_cols, old_cols):
    cols = []
    for col in new_cols:
        if col not in old_cols:
            cols.append(col)
            continue
        for i in itertools.count():
            name = f"{col}_{i}"
            if name not in old_cols:
                cols.append(name)
                break
    return cols
