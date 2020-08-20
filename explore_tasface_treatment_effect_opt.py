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
from two_leaf_opt import Canopy as TwoLeaf


def main(p, met, lai):

    days = met.doy
    hod = met.hod
    ndays = int(len(days) / 24.)
    nhours = len(met)
    hours_in_day = int(nhours / float(ndays))

    if hours_in_day == 24:
        met_timestep = 60.
    else:
        met_timestep = 30.
    timestep_sec = 60. * met_timestep

    # kg timestep (e.g. 30 min-1)-1 MPa-1 m-2
    p.Kmax *= c.MMOL_2_MOL * c.MOL_WATER_2_G_WATER * c.G_TO_KG * timestep_sec

    T = TwoLeaf(params=p, met_timestep=met_timestep)

    out, store = setup_output_dataframe(ndays, hours_in_day, p)

    i = 0
    hour_cnt = 1 # hour count
    day_cnt = 0
    while i < len(met):
        print("%d:%d" % (i, len(met)))
        year = met.index.year[i]
        doy = met.doy[i]
        hod = met.hod[i] + 1

        if day_cnt-1 == -1:
            beta = calc_beta(p.theta_sat)
            psi_soil = out.psi_soil[day_cnt]
        else:
            beta = calc_beta(out.sw[day_cnt-1])
            psi_soil = out.psi_soil[day_cnt-1]



        (An, et, Tcan,
         apar, lai_leaf) = T.main(met.tair[i], met.par[i], met.vpd[i],
                                  met.wind[i], met.press[i], met.ca[i],
                                  doy, hod, lai[i], psi_soil)

        if hour_cnt == hours_in_day: # End of the day
            store_daily(year, doy, day_cnt, store, beta, out, p)

            hour_cnt = 1
            day_cnt += 1
        else:
            store_hourly(hour_cnt-1, An, et, lai_leaf, met.precip[i], store,
                         timestep_sec)

            hour_cnt += 1

        i += 1

    return (out)

def calc_beta(theta, theta_fc=0.35, theta_wp=0.1):

    beta = (theta - theta_wp) / (theta_fc - theta_wp)
    beta = max(0.0, beta)
    beta = min(1.0, beta)

    return beta

def calc_swp(p, sw):
    """
    Calculate the soil water potential (MPa). The params The parameters b
    and psi_e are estimated from a typical soil moisture release function.

    Parameters:
    -----------
    sw : object
        volumetric soil water content, m3 m-3

    Returns:
    -----------
    psi_soil : float
        soil water potential, MPa

    References:
    -----------
    * Duursma et al. (2008) Tree Physiology 28, 265276, eqn 10
    """
    psi_soil_min = -20.0
    theta_min = 1E-03

    if sw < theta_min:
        psi_soil = psi_soil_min
    else:
        psi_soil = p.psi_e * (sw / p.theta_sat)**-p.b
        if psi_soil < -20:
            psi_soil = psi_soil_min

    return psi_soil   # MPa

def setup_output_dataframe(ndays, nhours, p):

    zero = np.zeros(ndays)
    out = pd.DataFrame({'year':zero, 'doy':zero,
                        'An_can':zero, 'E_can':zero, 'LAI':zero, 'sw':zero,
                        'beta': zero, 'psi_soil': zero})

    out.sw[0] = p.theta_sat
    out.psi_soil[0] = calc_swp(p, out.sw[0])

    zero = np.zeros(nhours)
    hour_store = pd.DataFrame({'An_can':zero, 'E_can':zero, 'LAI_can':zero,
                               'delta_sw':zero})

    return (out, hour_store)

def store_hourly(idx, An, et, lai_leaf, precip, store, timestep_sec):

    an_conv = c.UMOL_TO_MOL * c.MOL_C_TO_GRAMS_C * timestep_sec
    et_conv = c.MOL_WATER_2_G_WATER * c.G_TO_KG * timestep_sec
    precip_conv = timestep_sec

    #sun_frac = lai_leaf[c.SUNLIT] / np.sum(lai_leaf)
    #sha_frac = lai_leaf[c.SHADED] / np.sum(lai_leaf)
    store.An_can[idx] = np.sum(An * an_conv)
    store.E_can[idx] = np.sum(et * et_conv)
    store.LAI_can[idx] = np.sum(lai_leaf)

    precip *= precip_conv
    precip_max = 1.0    # hack for runoff
    if precip > precip_max:
        precip = precip_max

    store.delta_sw[idx] = precip - store.E_can[idx]

def store_daily(year, doy, idx, store, beta, out, p):

    out.year[idx] = year
    out.doy[idx] = doy
    out.An_can[idx] = np.sum(store.An_can)
    out.E_can[idx] = np.sum(store.E_can)
    out.LAI[idx] = np.mean(store.LAI_can)
    out.beta[idx] = beta

    if idx-1 == -1:
        prev_sw = p.theta_sat
    else:
        prev_sw = out.sw[idx-1]

    esoil = 3.0 # mm day-1
    delta = np.sum(store.delta_sw) - esoil
    delta_max = 3.0  # hacky, fix
    if delta > delta_max:
        delta = delta_max

    out.sw[idx] = update_sw_bucket(p, np.sum(store.delta_sw), prev_sw)
    out.psi_soil[idx] = calc_swp(p, out.sw[idx])


def update_sw_bucket(p, delta_sw, sw_prev):
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


    Returns:
    -------
    sw : float
        new volumetric soil water (m3 m-3)
    """

    sw = min(p.theta_sat, \
             sw_prev + delta_sw / (p.soil_volume * c.M_2_MM))
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

    time = met.copy()
    time_day = time.resample('D').mean()
    time_day = time_day.index

    aCa = met.ca.values.copy()
    eCa = aCa * 1.6
    #print(np.mean(aCa), np.mean(eCa))

    #print( np.sum(met.precip * 3600.0) )
    #sys.exit()

    # Need to create an LAI harvest timeseries
    lai = np.zeros(len(met))
    lai_max = 4.0
    days_to_max = 30
    cnt = 0
    for i in range(len(lai)):

        if cnt == 0.0:
            laix = 0.0
        else:
            laix += lai_max / float(days_to_max)

        lai[i] = laix

        #print(i, cnt, lai[i])

        cnt += 1
        if cnt > days_to_max:
            cnt = 0

    #lai = np.ones(len(met)) * 1.0


    out_aCa = main(p, met, lai)

    met.ca *= 1.5
    out_eCa = main(p, met, lai)


    #lai *= 1.2
    #out_eCa_eL = main(p, met, lai)

    fig = plt.figure(figsize=(9,16))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.2)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(time_day, out_aCa.An_can, "b-")
    ax1.plot(time_day, out_eCa.An_can, "r-", alpha=0.5)
    ax1.set_ylabel("An (g C m$^{-2}$ d$^{-1}$)")
    ax1.legend(numpoints=1, loc="best", frameon=False)
    ax1.set_ylim(0, 20)

    ax2.plot(time_day, out_aCa.E_can, "b-")
    ax2.plot(time_day, out_eCa.E_can, "r-", alpha=0.5)
    ax2.set_ylabel("E (mm d$^{-1}$)")
    ax2.set_ylim(0, 5)

    plt.setp(ax1.get_xticklabels(), visible=False)

    fig.autofmt_xdate()
    fig.savefig("opt_A_E.png", bbox_inches='tight', pad_inches=0.1)


    fig = plt.figure(figsize=(9,16))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.2)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)




    ax1.plot(time_day, out_aCa.sw, "b-", label="aC$_a$")
    ax1.plot(time_day, out_eCa.sw, "r-", label="eC$_a$")
    ax1.set_ylim(0.05, 0.35)
    #ax1.plot(time_day, out_eCa_eL.sw, "g-", label="eC$_a$ + e$_{LAI}$")
    ax1.legend(numpoints=1, loc="best", frameon=False)

    ax1.set_ylabel("SWC (m$^{3}$ m$^{-3}$)")

    #ax2.plot(out_aCa.sw, out_aCa.beta, "bo")
    #ax2.plot(out_eCa.sw, out_eCa.beta, "ro")
    #ax2.set_ylabel("An (g C m$^{-2}$ d$^{-1}$)")

    rr = np.log(out_eCa.An_can / out_aCa.An_can)
    response_eca = (np.exp(rr)-1.0)*100.0

    #rr = np.log(out_eCa_eL.An_can / out_aCa.An_can)
    #response_eca_ela = (np.exp(rr)-1.0)*100.0


    #response = ((out_eCa.An_can/out_aCa.An_can)-1.0)*100.
    #response_eca_ela = np.where(response_eca > 100, np.nan, response_eca_ela)
    response_eca = np.where(response_eca > 100, np.nan, response_eca)
    #print(np.nanmean(response_eca), np.nanmean(response_eca_ela))

    ax2.plot(time_day, response_eca, "r-", label="eC$_a$")
    #ax2.plot(time_day, response_eca_ela, "g-", label="eC$_a$ + e$_{LAI}$")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Response of A to CO$_2$ (%)")

    rr = np.log(out_eCa.E_can / out_aCa.E_can)
    response_eca = (np.exp(rr)-1.0)*100.0

    #rr = np.log(out_eCa_eL.E_can / out_aCa.E_can)
    #response_eca_ela = (np.exp(rr)-1.0)*100.0


    #response_eca_ela = np.where(response_eca > 50, np.nan, response_eca_ela)
    response_eca = np.where(response_eca > 50, np.nan, response_eca)
    #print(np.nanmean(response_eca), np.nanmean(response_eca_ela))

    ax3.plot(time_day, response_eca, "r-", label="eC$_a$")
    #ax3.plot(time_day, response_eca_ela, "g-", label="eC$_a$ + e$_{LAI}$")
    ax3.set_ylim(-50, 50)
    ax3.set_ylabel("Response of E to CO$_2$ (%)")


    from matplotlib.ticker import MaxNLocator
    ax1.yaxis.set_major_locator(MaxNLocator(5))

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    fig.autofmt_xdate()
    fig.savefig("blah_opt.png", bbox_inches='tight', pad_inches=0.1)
