# tsdisagg
Tools for converting low time series data to high frequency, based on the R package `tempdisagg`, and espeically the accompanying paper by [Sax and Steiner 2013](https://journal.r-project.org/archive/2013-2/sax-steiner.pdf).

`tsdisagg` allows the user to convert low frequency time series data (e.g., yearly or quarterly) to a higher frequency (e.g., quarterly or monthly) in a way that preserves desired aggregate statistics in the high frequency data. It should, for example, sum back to the original low-frequency data.

In addition, regression-based methods are also implemented that allow the user to supply "indicator series", allowing variation from correlated high-frequency time series to be imputed into the low frequency data.

If you have any questions or issues, please open a thread. Pull requests to add features or fix bugs are welcome. Please clone the repository locally to have access to the testing suite.

## Installation
To install, use
`pip install tsdisagg`

## Current Features
Currently, only conversion between yearly, quarterly, and monthly data is supported. Conversion to lower frequencies is non-trivial due to the calendar math that needs to be added, but this is on my to-do list.

The following interpolation methods have been implemented:

Single series, non-parametric methods:
- Denton
- Denton-Cholette

Multiseries, regression-based methods:
- Chow-Lin
- Litterman


## Examples

Disaggregate a timeseries using the univariate Denton-Cholette method:
```python
import pandas as pd
from tsdisagg import disaggregate_series
from tsdisagg.datasets import load_data

# Load example data
sales_a = load_data("annual_sales")


# Disaggregate from annual to quarterly using Denton-Cholette method
sales_q_dc = disaggregate_series(
    sales_a.resample("YS").last(), # Use `.resample` to ensure the frequency is set correctly
    target_freq="QS", # Desired output frequency
    method="denton-cholette", # Disaggregation method
    agg_func="sum", # Sales are flow data, so we want the quarters to sum back to the annual data
    h=1, # Differencing order (1 in this case to preserve the trend)
)
```

Disaggregate a timeseries using the multivariate Chow-Lin method with an indicator series:
```python
import pandas as pd
from tsdisagg import disaggregate_series
from tsdisagg.datasets import load_data

# Load example data
sales_a = load_data("annual_sales")
exports_q = load_data("quarterly_exports")

# Disaggregate from annual to quarterly using Chow-Lin method with quarterly sales as indicator
sales_q_chow_lin = disaggregate_series(
    sales_a.resample("YS").last(), # Target series, annual frequency
    exports_q.assign(intercept=1), # Indicator matrix. We can have as many series as we want here; so we use
                                   # Exports at quarterly frequency, plus a deterministic intercept term.
    method="chow-lin", # Disaggregation method
    agg_func="sum", # Sales are flow data, so we want the quarters to sum back to the annual data
    optimizer_kwargs={"method": "powell"}, # Additional arguments to the optimizer
)
```

# Citing `tsdisagg`
If you use `tsdisagg` in your research, please use the following citation:

```bibtex
@software{tsdisagg,
author = {Jesse Grabowski},
title = {tsdisagg: Temporal Disaggregation of Time Series Data in Python},
version = {0.1.0},
url = {https://github.com/jessegrabowski/tsdisagg},
howpublished = {GitHub},
year = {2025},
}
```
