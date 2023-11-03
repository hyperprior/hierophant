from functools import wraps
from numbers import Number
from typing import Dict, List, TypedDict, Union

import numpy as np
import ray
from beartype import beartype

Statistic = Dict[str, Number]


class NumericColumn(TypedDict):
    name: str
    values: Dict[str, Union[np.ndarray, List[Number]]]


class ColumnStatistic(TypedDict):
    column: str
    statistic: str
    value: Number


@beartype
@ray.remote
def number_rows(

@beartype
@ray.remote
def column_mean(column: NumericColumn, sigdig=3) -> ColumnStatistic:
    return {
        "column": column["name"],
        "statistic": "mean",
        "value": round(column["values"].mean(), sigdig),
    }


@beartype
@ray.remote
def column_min(column: NumericColumn, sigdig=3) -> ColumnStatistic:
    return {
        "column": column["name"],
        "statistic": "min",
        "value": round(column["values"].min(), sigdig),
    }


@beartype
@ray.remote
def column_max(column: NumericColumn, sigdig=3) -> ColumnStatistic:
    return {
        "column": column["name"],
        "statistic": "max",
        "value": round(column["values"].max(), sigdig),
    }


@beartype
@ray.remote
def column_variance(column: NumericColumn, sigdig=3) -> ColumnStatistic:
    return {
        "column": column["name"],
        "statistic": "variance",
        "value": round(column["values"].var(), sigdig),
    }


@beartype
@ray.remote
def column_std(column: NumericColumn, sigdig=3) -> ColumnStatistic:
    return {
        "column": column["name"],
        "statistic": "standard deviation",
        "value": round(column["values"].std(), sigdig),
    }


@beartype
@ray.remote
def column_quantiles(
    column: NumericColumn,
    quantiles=[0.01, 0.05, 0.1, 0.25, 0.33, 0.5, 0.67, 0.75, 0.9, 0.95, 0.99],
    sigdig=3,
) -> ColumnStatistic:
    return {
        "column": column["name"],
        "statistic": "quantiles",
        "value": round(column["values"].quantile(quantiles), sigdig).to_dict(),
    }


@beartype
@ray.remote
def column_dtype(column: NumericColumn) -> ColumnStatistic:
    return {
        "column": column["name"],
        "statistic": "dtype",
        "value": str(column["values"].dtype),
    }


@beartype
@ray.remote
def column_number_negative(column: NumericColumn) -> ColumnStatistic:
    return {
        "column": column["name"],
        "statistic": "number negative",
        "value": int((column["values"] < 0).sum()),
    }


@beartype
@ray.remote
def column_proportion_negative(column: NumericColumn, sigdig=3) -> ColumnStatistic:
    return {
        "column": column["name"],
        "statistic": "proportion negative",
        "value": round((column["values"] < 0).mean(), sigdig),
    }


@beartype
@ray.remote
def column_number_zeros(column: NumericColumn) -> ColumnStatistic:
    return {
        "column": column["name"],
        "statistic": "number zeros",
        "value": int((column["values"] == 0).sum()),
    }


@beartype
@ray.remote
def column_proportion_zeros(column: NumericColumn, sigdig=3) -> ColumnStatistic:
    return {
        "column": column["name"],
        "statistic": "proportion negative",
        "value": round((column["values"] == 0).mean(), sigdig),
    }


@beartype
@ray.remote
def column_number_positive(column: NumericColumn) -> ColumnStatistic:
    return {
        "column": column["name"],
        "statistic": "number positive",
        "value": int((column["values"] > 0).sum()),
    }


@beartype
@ray.remote
def column_proportion_positive(column: NumericColumn, sigdig=3) -> ColumnStatistic:
    return {
        "column": column["name"],
        "statistic": "proportion positive",
        "value": round((column["values"] > 0).mean(), sigdig),
    }
