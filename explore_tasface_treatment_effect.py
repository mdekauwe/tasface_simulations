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

import constants as c
import parameters as p
from radiation import calculate_absorbed_radiation
from two_leaf import Canopy as TwoLeaf


def main(met, lai, soil_volume, theta_sat):

    T = TwoLeaf(p, gs_model="medlyn")

    days = met.doy
    hod = met.hod
    ndays = int(len(days) / 24.)
    nhours = len(met)
    hours_in_day = int(nhours / float(ndays))

    out, store = setup_output_dataframe(ndays, hours_in_day)

    i = 0
    hour_cnt = 1 # hour count
    day_cnt = 0
    while i < len(met):
        year = met.index.year[i]
        doy = met.doy[i]
        hod = met.hod[i] + 1

        if day_cnt-1 == -1:
            beta = calc_beta(theta_sat)
        else:
            beta = calc_beta(out.sw[day_cnt-1])

        (An, et, Tcan,
         apar, lai_leaf) = T.main(met.tair[i], met.par[i], met.vpd[i],
                                  met.wind[i], met.press[i], met.ca[i],
                                  doy, hod, lai[i], Vcmax25=p.Vcmax25,
                                  Jmax25=p.Jmax25, beta=beta)

        if hour_cnt == hours_in_day: # End of the day
            store_daily(year, doy, day_cnt, store, soil_volume, theta_sat, beta,
                        out)

            hour_cnt = 1
            day_cnt += 1
        else:
            store_hourly(hour_cnt-1, An, et, lai_leaf, met.precip[i], store,
                         hours_in_day)

            hour_cnt += 1

        i += 1

    return (out)

def calc_beta(theta, theta_fc=0.4, theta_wp=0.1):

    beta = (theta - theta_wp) / (theta_fc - theta_wp)
    beta = max(0.0, beta)
    beta = min(1.0, beta)

    return beta

def setup_output_dataframe(ndays, nhours):

    zero = np.zeros(ndays)
    out = pd.DataFrame({'year':zero, 'doy':zero,
                        'An_can':zero, 'E_can':zero, 'LAI':zero, 'sw':zero,
                        'beta': zero})

    zero = np.zeros(nhours)
    hour_store = pd.DataFrame({'An_can':zero, 'E_can':zero, 'LAI_can':zero,
                               'delta_sw':zero})

    return (out, hour_store)

def store_hourly(idx, An, et, lai_leaf, precip, store, hours_in_day):

    if hours_in_day == 24:
        an_conv = c.UMOL_TO_MOL * c.MOL_C_TO_GRAMS_C * c.SEC_TO_HR
        et_conv = c.MOL_WATER_2_G_WATER * c.G_TO_KG * c.SEC_TO_HR
        precip_conv = c.SEC_TO_HR
    else:
        an_conv = c.UMOL_TO_MOL * c.MOL_C_TO_GRAMS_C * c.SEC_TO_HLFHR
        et_conv = c.MOL_WATER_2_G_WATER * c.G_TO_KG * c.SEC_TO_HLFHR
        precip_conv = c.SEC_TO_HLFHR

    #sun_frac = lai_leaf[c.SUNLIT] / np.sum(lai_leaf)
    #sha_frac = lai_leaf[c.SHADED] / np.sum(lai_leaf)
    store.An_can[idx] = np.sum(An * an_conv)
    store.E_can[idx] = np.sum(et * et_conv)
    store.LAI_can[idx] = np.sum(lai_leaf)
    store.delta_sw[idx] = (precip * precip_conv) - store.E_can[idx]

def store_daily(year, doy, idx, store, soil_volume, theta_sat, beta, out):

    out.year[idx] = year
    out.doy[idx] = doy
    out.An_can[idx] = np.sum(store.An_can)
    out.E_can[idx] = np.sum(store.E_can)
    out.LAI[idx] = np.mean(store.LAI_can)
    out.beta[idx] = beta

    if idx-1 == -1:
        prev_sw = theta_sat
    else:
        prev_sw = out.sw[idx-1]
    out.sw[idx] = update_sw_bucket(np.sum(store.delta_sw), prev_sw,
                                   soil_volume, theta_sat)

def update_sw_bucket(delta_sw, sw_prev, soil_volume, theta_sat):
    """
    Update the simple bucket soil water balance

    Parameters:
    -----------
    precip : float
        precipitation (kg m-2 s-1)
    water_loss : float
        flux of water out of the soil (transpiration (kg m-2 timestep-1))
    sw_prev : float
        volumetric soil water from the previous timestep (m3 m-3)
    soil_volume : float
        volume soil water bucket (m3)

    Returns:
    -------
    sw : float
        new volumetric soil water (m3 m-3)
    """

    sw = min(theta_sat, \
             sw_prev + delta_sw / (soil_volume * c.M_2_MM))
    sw = max(0.0, sw)

    return sw



def read_met_file(fname):
    """ Build a dataframe from the netcdf outputs """

    ds = xr.open_dataset(fname)
    lat = ds.latitude.values[0][0]
    lon = ds.longitude.values[0][0]

    vars_to_keep = ['SWdown','Tair','Wind','Psurf',\
                    'VPD','CO2air','Precip']
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
                            'VPD': 'vpd', 'CO2air': 'ca', 'Psurf': 'press',
                            'Precip': 'precip'})

    # Make sure there is no bad data...
    df.vpd = np.where(df.vpd < 0.0, 0.0, df.vpd)

    return df, lat, lon


if __name__ == "__main__":


    #
    ##  Just use TUMBA nc for now
    #
    met_fn = "met/AU-Tum_2002-2017_OzFlux_Met.nc"
    (met, lat, lon) = read_met_file(met_fn)

    #plt.plot(met.press)
    #plt.show()

    # Just keep ~ a spring/summer
    met = met[(met.index.year == 2003) | (met.index.year == 2004)]
    met = met[ ((met.index.year == 2003) & (met.doy >= 260)) |
               ((met.index.year == 2004) & (met.doy <= 90)) ]


    # Need to create an LAI harvest timeseries, for now fix it.
    lai = np.ones(len(met)) * 1.0


    soil_depth = 0.2 # depth of soil bucket, m
    ground_area = 1.0 # m
    soil_volume = ground_area * soil_depth # m3
    theta_sat = 0.5
    out = main(met, lai, soil_volume, theta_sat)

    #plt.plot(out.An_can)
    #plt.plot(out.sw, out.beta, "ko")
    plt.plot(out.sw)
    plt.show()
