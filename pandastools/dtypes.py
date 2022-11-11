from typing import Any

from bidict import bidict
import numpy as np
import pandas as pd


class DType:
    name: str
    pandas_str: str
    dtype: Any
    is_index_dtype = True

    @classmethod
    def get_dtypes(cls) -> bidict[str, str]:
        dct = {i.pandas_str: i.name for i in cls.__subclasses__()}
        return bidict(dct)

    @classmethod
    def get_dtypes_for_index(cls) -> bidict[str, str]:
        dct = {i.pandas_str: i.name for i in cls.__subclasses__() if i.is_index_dtype}
        return bidict(dct)


class Int8(DType):
    name = "Integer (8 bit)"
    pandas_str = "Int8"
    dtype = pd.Int8Dtype()
    # np_dtype = np.int8


class Int16(DType):
    name = "Integer (16 bit)"
    pandas_str = "Int16"
    dtype = pd.Int16Dtype()
    # np_dtype = np.int16


class Int64(DType):
    name = "Integer (64 bit)"
    pandas_str = "Int64"
    dtype = pd.Int64Dtype()
    # np_dtype = np.int64


class Float32(DType):
    name = "Float (32 bit)"
    pandas_str = "float32"
    dtype = np.float32
    # np_dtype = np.float32


class Float64(DType):
    name = "Float (64 bit)"
    pandas_str = "float64"
    dtype = np.float64
    # np_dtype = np.float64


class Category(DType):
    name = "Category"
    pandas_str = "category"
    dtype = pd.CategoricalDtype()


class DateTime(DType):
    name = "DateTime"
    pandas_str = "datetime64[ns, UTC]"
    dtype = pd.DatetimeTZDtype(tz="UTC")


class String(DType):
    name = "String"
    pandas_str = "string"
    dtype = pd.StringDtype()


class Bool(DType):
    name = "Bool"
    pandas_str = "bool"
    dtype = bool
    is_index_dtype = False


class Boolean(DType):
    name = "Boolean"
    pandas_str = "boolean"
    dtype = pd.BooleanDtype()
    is_index_dtype = False


class Object(DType):
    name = "Object"
    pandas_str = "object"
    dtype = object
    is_index_dtype = False
