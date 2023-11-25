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
For example usage, please see the `examples.ipynb` notebook. `tsdisagg` depends heavily on `pandas` to handle time reindexing, so the user is advised to read the associated Pandas documentation, especially as it relates to setting frequencies.


## To-do:
1. Refactor codebase to use `statsmodels` model and results objects, as well as `.fit()` api
2. Add missing interpolation methods relative to `timedisagg` (Fernandez, min RSS objective functions)
3. Add support for finer time frequencies (weekly, daily, hourly)
