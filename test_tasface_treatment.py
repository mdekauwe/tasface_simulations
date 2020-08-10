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

sys.path.append('src')

from weather_generator import WeatherGenerator
import constants as c
from farq import FarquharC3
from penman_monteith_leaf import PenmanMonteith


class CoupledModel(object):
    """Iteratively solve leaf temp, Ci, gs and An."""

    def __init__(self, g0, g1, D0, gamma, Vcmax25, Jmax25, Rd25, Eaj, Eav,
                 deltaSj, deltaSv, Hdv, Hdj, Q10, leaf_width, SW_abs,
                 gs_model, sw0=0.5, psi_e=-0.8*c.KPA_2_MPA, theta_sat=0.5, b=6.,
                 soil_depth=1.0, ground_area=1.0, alpha=None, iter_max=100,
                 met_timestep=30):

        # set params
        self.g0 = g0
        self.g1 = g1
        self.D0 = D0
        self.gamma = gamma
        self.Vcmax25 = Vcmax25
        self.Jmax25 = Jmax25
        self.Rd25 = Rd25
        self.Eaj = Eaj
        self.Eav = Eav
        self.deltaSj = deltaSj
        self.deltaSv = deltaSv
        self.Hdv = Hdv
        self.Hdj = Hdj
        self.Q10 = Q10
        self.leaf_width = leaf_width
        self.alpha = alpha
        self.SW_abs = SW_abs # leaf abs of solar rad [0,1]
        self.gs_model = gs_model
        self.iter_max = iter_max
        self.sw0 = sw0
        self.psi_e = psi_e
        self.theta_sat = theta_sat
        self.b = b
        self.timestep_sec = 60. * met_timestep
        self.soil_depth = soil_depth # depth of soil bucket, m
        self.ground_area = ground_area # m
        self.soil_volume = self.ground_area * self.soil_depth # m3

        self.emissivity_leaf = 0.99   # emissivity of leaf (-)

    def initialise_model(self, met):
        """
        Set everything up: set initial values, build an output dataframe to
        save things

        Parameters:
        -----------
        met : object
            met forcing variables: day; Ca; par; precip; press; tair; vpd

        Returns:
        -------
        n : int
            number of timesteps in the met file
        out : object
            output dataframe to store things as we go along

        """
        n = len(met)

        out = self.setup_out_df(met)
        out.sw[0] = self.sw0
        out.gsw[0] = 0.0
        out.anleaf[0] = 0.0
        out.eleaf[0] = 0.0
        out.hod[0] = 0
        out.doy[0] = 0
        out.year[0] = met.year.iloc[0]

        return n, out

    def setup_out_df(self, met):
        """
        Create and output dataframe to save things

        Parameters:
        -----------
        met : object
            met forcing variables: day; Ca; par; precip; press; tair; vpd

        Returns:
        -------
        out : object
            output dataframe to store things as we go along.
        """
        dummy = np.ones(len(met)) * np.nan
        out = pd.DataFrame({'year':dummy,
                            'doy':dummy,
                            'hod':dummy,
                            'eleaf':dummy,
                            'sw':dummy,
                            'anleaf':dummy,
                            'gsw':dummy})

        return out


    def main(self, met):
        """
        Parameters:
        ----------
        tair : float
            air temperature (deg C)
        par : float
            Photosynthetically active radiation (umol m-2 s-1)
        vpd : float
            Vapour pressure deficit (kPa, needs to be in Pa, see conversion
            below)
        wind : float
            wind speed (m s-1)
        pressure : float
            air pressure (using constant) (Pa)
        Ca : float
            ambient CO2 concentration

        Returns:
        --------
        An : float
            net leaf assimilation (umol m-2 s-1)
        gs : float
            stomatal conductance (mol m-2 s-1)
        et : float
            transpiration (mol H2O m-2 s-1)
        """

        F = FarquharC3(theta_J=0.85, peaked_Jmax=True, peaked_Vcmax=True,
                       model_Q10=True, gs_model=self.gs_model,
                       gamma=self.gamma, g0=self.g0,
                       g1=self.g1, D0=self.D0, alpha=self.alpha)
        P = PenmanMonteith(self.leaf_width, self.SW_abs)

        (n, out) = self.initialise_model(met)

        an_day = []
        astore = 0.0
        e_day = []
        estore = 0.0
        daylen = 0.0

        hod = 1
        doy = 1
        for i in range(1, len(met)):

            if met.par[i] > 50.0:
                out = self.run_timestep(F, P, i, met, out)
                daylen += 1.0
            else:
                out.gsw[i] = 0.0
                out.anleaf[i] = 0.0
                out.eleaf[i] = 0.0

            lai = 2.0
            water_losss = out.eleaf[i] * lai
            out.sw[i] = self.update_sw_bucket(met.precip[i], out.eleaf[i],
                                              out.sw[i-1])

            astore += out.anleaf[i]
            estore += out.eleaf[i]

            out.hod[i] = hod
            out.doy[i] = doy
            out.year[i] = met.year[i]
            hod += 1
            if hod > 47:
                hod = 0
                doy += 1
                #print(daylen/2.0)
                an_day.append(astore/(daylen/2.0))
                astore = 0.0
                e_day.append(estore/(daylen/2.0))
                estore = 0.0
                daylen = 0.0


        return out, an_day, e_day

    def run_timestep(self, F, P, i, met, out, rnet=None):


        # set initialise values
        dleaf = met.vpd[i]
        dair = met.tair[i]
        tair = met.tair[i]
        vpd = met.vpd[i]
        Ca = met.ca[i]
        Cs = met.ca[i]
        pressure = met.press[i]
        wind = met.wind[i]
        Tleaf = tair
        Tleaf_K = Tleaf + c.DEG_2_KELVIN
        par = met.par[i]

        #print("Start: %.3f %.3f %.3f" % (Cs, Tleaf, dleaf))



        iter = 0
        while True:

            (An, gsc, Ci) = F.calc_photosynthesis(Cs=Cs, Tleaf=Tleaf_K, Par=par,
                                                  Jmax25=self.Jmax25,
                                                  Vcmax25=self.Vcmax25,
                                                  Q10=self.Q10, Eaj=self.Eaj,
                                                  Eav=self.Eav,
                                                  deltaSj=self.deltaSj,
                                                  deltaSv=self.deltaSv,
                                                  Rd25=self.Rd25, Hdv=self.Hdv,
                                                  Hdj=self.Hdj, vpd=dleaf)

            # Calculate new Tleaf, dleaf, Cs

            (new_tleaf, et,
             le_et, gbH, gw) = self.calc_leaf_temp(P, Tleaf, tair, gsc,
                                                   par, vpd, pressure, wind,
                                                   rnet=rnet)

            gbc = gbH * c.GBH_2_GBC
            if gbc > 0.0 and An > 0.0:
                Cs = Ca - An / gbc # boundary layer of leaf
            else:
                Cs = Ca

            if math.isclose(et, 0.0) or math.isclose(gw, 0.0):
                dleaf = dair
            else:
                dleaf = (et * pressure / gw) * c.PA_2_KPA # kPa

            # Check for convergence...?
            if math.fabs(Tleaf - new_tleaf) < 0.02:
                break

            if iter > self.iter_max:
                raise Exception('No convergence: %d' % (iter))

            # Update temperature & do another iteration
            Tleaf = new_tleaf
            Tleaf_K = Tleaf + c.DEG_2_KELVIN

            iter += 1

        out.gsw[i] = gsc * c.GSC_2_GSW
        out.anleaf[i] = An
        out.eleaf[i] = et

        if et < 0.0:
            raise Exception("ET shouldn't be negative, issue in energy balance")


        return (out)

    def calc_leaf_temp(self, P=None, tleaf=None, tair=None, gsc=None, par=None,
                       vpd=None, pressure=None, wind=None, rnet=None):
        """
        Resolve leaf temp

        Parameters:
        ----------
        P : object
            Penman-Montheith class instance
        tleaf : float
            leaf temperature (deg C)
        tair : float
            air temperature (deg C)
        gs : float
            stomatal conductance (mol m-2 s-1)
        par : float
            Photosynthetically active radiation (umol m-2 s-1)
        vpd : float
            Vapour pressure deficit (kPa, needs to be in Pa, see conversion
            below)
        pressure : float
            air pressure (using constant) (Pa)
        wind : float
            wind speed (m s-1)

        Returns:
        --------
        new_Tleaf : float
            new leaf temperature (deg C)
        et : float
            transpiration (mol H2O m-2 s-1)
        gbH : float
            total boundary layer conductance to heat for one side of the leaf
        gw : float
            total leaf conductance to water vapour (mol m-2 s-1)
        """
        tleaf_k = tleaf + c.DEG_2_KELVIN
        tair_k = tair + c.DEG_2_KELVIN

        air_density = pressure / (c.RSPECIFC_DRY_AIR * tair_k)

        # convert from mm s-1 to mol m-2 s-1
        cmolar = pressure / (c.RGAS * tair_k)

        # W m-2 = J m-2 s-1
        if rnet is None:
            rnet = P.calc_rnet(par, tair, tair_k, tleaf_k, vpd, pressure)

        (grn, gh, gbH, gw) = P.calc_conductances(tair_k, tleaf, tair,
                                                 wind, gsc, cmolar)
        if math.isclose(gsc, 0.0):
            et = 0.0
            le_et = 0.0
        else:
            (et, le_et) = P.calc_et(tleaf, tair, vpd, pressure, wind, par,
                                    gh, gw, rnet)

        # D6 in Leuning. NB I'm doubling conductances, see note below E5.
        # Leuning isn't explicit about grn but I think this is right
        # NB the units or grn and gbH are mol m-2 s-1 and not m s-1, but it
        # cancels.
        Y = 1.0 / (1.0 + (2.0 * grn) / (2.0 * gbH))

        # sensible heat exchanged between leaf and surroundings
        H = Y * (rnet - le_et)

        # leaf-air temperature difference recalculated from energy balance.
        # NB. I'm using gh here to include grn and the doubling of conductances
        new_Tleaf = tair + H / (c.CP * air_density * (gh / cmolar))

        return (new_Tleaf, et, le_et, gbH, gw)

    def update_sw_bucket(self, precip, water_loss, sw_prev):
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
        loss = water_loss * c.MOL_WATER_2_G_WATER * \
                c.G_TO_KG * self.timestep_sec
        delta_sw = (precip * self.timestep_sec) - loss

        sw = min(self.theta_sat, \
                 sw_prev + delta_sw / (self.soil_volume * c.M_2_MM))
        #print(sw * (self.soil_volume * c.M_2_MM))
        return sw


if __name__ == "__main__":

    #
    ## Hack together a year's worth of data just to get this running
    #

    # NB need to check why, but only works for Aus if this is positive,
    # otherwise peak is N.H summer
    lat = 41.7 # roughly old TASFACE loc -
    lon = 147.3 # roughly old TASFACE loc


    #
    ##  Just use TUMBA nc for now
    #



    # Parameters

    # gs stuff
    g0 = 0.001
    g1 = 9.0
    D0 = 1.5 # kpa

    # A stuff
    Vcmax25 = 30.0
    Jmax25 = Vcmax25 * 2.0
    Rd25 = 2.0
    Eaj = 30000.0
    Eav = 60000.0
    deltaSj = 650.0
    deltaSv = 650.0
    Hdv = 200000.0
    Hdj = 200000.0
    Q10 = 2.0
    gamma = 0.0

    # Misc stuff
    leaf_width = 0.02

    # Cambell & Norman, 11.5, pg 178
    # The solar absorptivities of leaves (-0.5) from Table 11.4 (Gates, 1980)
    # with canopies (~0.8) from Table 11.2 reveals a surprising difference.
    # The higher absorptivityof canopies arises because of multiple reflections
    # among leaves in a canopy and depends on the architecture of the canopy.
    SW_abs = 0.8 # use canopy absorptance of solar radiation


    CM = CoupledModel(g0, g1, D0, gamma, Vcmax25, Jmax25, Rd25, Eaj, Eav,
                     deltaSj,
                     deltaSv, Hdv, Hdj, Q10, leaf_width, SW_abs,
                     gs_model="leuning")
    out, an_day, e_day = CM.main(met)

    print("THAT'S WRONG - FIX")
    plt.plot(an_day)
    plt.show()

    plt.plot(e_day)
    plt.show()

    plt.plot(out.sw)
    plt.show()
