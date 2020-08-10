#!/usr/bin/env python
"""
Blah


"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (10.08.2020)"
__email__ = "mdekauwe@gmail.com"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import math
import xarray as xr

sys.path.append('src')

from weather_generator import WeatherGenerator
import constants as c
from farq import FarquharC3
from penman_monteith_leaf import PenmanMonteith


def main(met):

    days = met.doy
    hod = met.hod
    ndays = int(len(days) / 24.)
    nhours = len(met)
    print(ndays, nhours)

    out = setup_output_dataframe(nhours)



def setup_output_dataframe(ndays):

    zero = np.zeros(ndays)
    out = pd.DataFrame({'year':zero, 'doy':zero,
                        'An_can':zero, 'An_sun':zero, 'An_sha':zero,
                        'E_can':zero, 'E_sun':zero, 'E_sha':zero})
    return (out)


def read_met_file(fname):
    """ Build a dataframe from the netcdf outputs """

    ds = xr.open_dataset(fname)
    lat = ds.latitude.values[0][0]
    lon = ds.longitude.values[0][0]

    vars_to_keep = ['SWdown','Tair','Wind','Psurf',\
                    'VPD','CO2air','Wind']
    df = ds[vars_to_keep].squeeze(dim=["y","x"],
                                  drop=True).to_dataframe()

    time_idx = df.index

    df = df.reindex(time_idx)
    df['year'] = df.index.year
    df['hod'] = df.index.hour
    df['doy'] = df.index.dayofyear
    df["par"] = df.SWdown * c.SW_2_PAR
    df["Tair"] -= c.DEG_2_KELVIN
    df = df.drop('SWdown', axis=1)
    df = df.rename(columns={'PAR': 'par', 'Tair': 'tair', 'Wind': 'wind',
                            'VPD': 'vpd', 'CO2air': 'co2', 'Psurf': 'press'})

    # Make sure there is no bad data...
    df.vpd = np.where(df.vpd < 0.0, 0.0, df.vpd)

    return df, lat, lon


if __name__ == "__main__":


    #
    ##  Just use TUMBA nc for now
    #
    year_to_run = 2003
    met_fn = "met/AU-Tum_2002-2017_OzFlux_Met.nc"
    (met, lat, lon) = read_met_file(met_fn)
    met = met[met.index.year == year_to_run]

    #plt.plot(met.press)
    #plt.show()




    main(met)
