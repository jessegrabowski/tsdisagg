from importlib.metadata import version

from tsdisagg.ts_disagg import disaggregate_series

__all__ = ["disaggregate_series"]

__version__ = version("tsdisagg")
