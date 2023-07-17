"""
This script takes the CMS sample paths generated for series measurements, separates
sets of scan file paths, and outputs an xr.DataSet for the ex situ measurements or outputs a xr.DataArray(s) for in situ time-resolved measurements.
"""

### Imports:

### Imports:
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import xarray as xr
import PyHyperScattering as phs
import pygix
from tqdm.auto import tqdm  # progress bar loader!

# Load custom pygix functions
pg_convert = phs.util.GIWAXS.pg_convert  
pg_convert_series = phs.util.GIWAXS.pg_convert_series
