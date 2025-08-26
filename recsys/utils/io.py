import polars as pl
from typing import Iterable, Union

def scan_parquet(paths: Union[str, Iterable[str]]) -> pl.LazyFrame:
    return pl.scan_parquet(paths)

def write_parquet(df: pl.DataFrame, path: str):
    df.write_parquet(path, compression="zstd")
