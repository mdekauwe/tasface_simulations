#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Solve 30-minute coupled A-gs(E) using a two-leaf approximation roughly following
Wang and Leuning.

References:
----------
* Wang & Leuning (1998) Agricultural & Forest Meterorology, 91, 89-111.
* Dai et al. (2004) Journal of Climate, 17, 2281-2299.
* De Pury & Farquhar (1997) PCE, 20, 537-557.

"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, sin, exp, sqrt, acos, asin
import random
import math

import constants as c
from farq import FarquharC3
from radiation import spitters
from radiation import calculate_absorbed_radiation
from radiation import calculate_cos_zenith, calc_leaf_to_canopy_scalar
from sperry_optimisation import ProfitMax
from utils import calc_esat
from penman_monteith_leaf import PenmanMonteith

__author__  = "Martin De Kauwe"
__version__ = "1.0 (09.11.2018)"
__email__   = "mdekauwe@gmail.com"


class Canopy(object):
    """Iteratively solve leaf temp, Ci, gs and An."""

    def __init__(self, params=None, peaked_Jmax=True, peaked_Vcmax=True,
                 met_timestep=30):

        self.p = params
        self.S = ProfitMax(params=params, met_timestep=met_timestep,
                           derive_weibull_params=False)
        self.F = FarquharC3(peaked_Jmax=peaked_Jmax,
                            peaked_Vcmax=peaked_Vcmax)
        self.PM = PenmanMonteith()


    def main(self, tair, par, vpd, wind, pressure, Ca, doy, hod,
             lai, psi_soil):
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
        doy : float
            day of day
        hod : float
            hour of day
        lat : float
            latitude
        lon : float
            longitude
        lai : floar
            leaf area index

        Returns:
        --------
        An : float
            net leaf assimilation (umol m-2 s-1)
        gs : float
            stomatal conductance (mol m-2 s-1)
        et : float
            transpiration (mol H2O m-2 s-1)
        """

        An = np.zeros(2) # sunlit, shaded
        gsc = np.zeros(2)  # sunlit, shaded
        gsw = np.zeros(2)  # sunlit, shaded
        Evap = np.zeros(2) # sunlit, shaded
        Tcan = np.zeros(2) # sunlit, shaded
        lai_leaf = np.zeros(2)
        sw_rad = np.zeros(2) # VIS, NIR
        tcanopy = np.zeros(2)

        opt_a = np.zeros(2)
        opt_gsw = np.zeros(2)
        opt_gsc = np.zeros(2)
        opt_e = np.zeros(2)
        opt_p = np.zeros(2)


        (cos_zenith, elevation) = calculate_cos_zenith(doy, self.p.lat, hod)

        sw_rad[c.VIS] = 0.5 * (par * c.PAR_2_SW) # W m-2
        sw_rad[c.NIR] = 0.5 * (par * c.PAR_2_SW) # W m-2

        # get diffuse/beam frac, just use VIS as the answer is the same for NIR
        (diffuse_frac, direct_frac) = spitters(doy, sw_rad[0], cos_zenith)

        (qcan, apar,
         lai_leaf, kb) = calculate_absorbed_radiation(self.p, par, cos_zenith,
                                                      lai, direct_frac,
                                                      diffuse_frac, doy, sw_rad,
                                                      tair)

        # Calculate scaling term to go from a single leaf to canopy,
        # see Wang & Leuning 1998 appendix C
        scalex = calc_leaf_to_canopy_scalar(lai, kb=kb, kn=self.p.kn)

        if lai_leaf[0] < 1.e-3: # to match line 336 of CABLE radiation
            scalex[0] = 0.

        # Is the sun up?
        if elevation > 0.0 and par > 50.:

            # sunlit / shaded loop
            for ileaf in range(2):

                # initialise values of Tleaf, Cs, dleaf at the leaf surface
                dleaf = vpd
                if dleaf < 0.05:
                    dleaf = 0.05
                #Cs = Ca * c.umol_to_mol * pressure
                Cs = Ca

                press = pressure * c.PA_2_KPA
                Tleaf = tair
                Tleaf_K = Tleaf + c.DEG_2_KELVIN


                iter = 0
                while True:

                    if scalex[ileaf] > 0.:


                        (opt_a[ileaf],
                         opt_gsw[ileaf],
                         opt_gsc[ileaf],
                         opt_e[ileaf],
                         opt_p[ileaf]) = self.S.optimisation(self.p, self.F,
                                                             psi_soil,
                                                             dleaf, Cs,
                                                             Tleaf_K,
                                                             apar[ileaf],
                                                             press,
                                                             lai_leaf[ileaf],
                                                             scalex[ileaf])


                        An[ileaf] = opt_a[ileaf]
                        gsw[ileaf] = opt_gsw[ileaf]
                        gsc[ileaf] = opt_gsc[ileaf]
                        Evap[ileaf] = opt_e[ileaf]

                        # Put Manon's funcs in and iterate
                        # Calc leaf temp

                    else:
                        An[ileaf] = 0.0
                        gsw[ileaf] = 0.0
                        gsc[ileaf] = 0.0
                        Evap[ileaf] = 0.0

                    # Calculate new Tleaf, dleaf, Cs ... need to fix
                    (new_tleaf) = self.calc_leaf_temp(self.p, Tleaf,
                                                      tair, gsc[ileaf],
                                                      vpd, pressure, wind,
                                                      rnet=qcan[ileaf],
                                                      lai=lai_leaf[ileaf],
                                                      trans=Evap[ileaf] )

                    if np.isclose(Evap[ileaf], 0.0) or \
                       np.isclose(gsw[ileaf], 0.0):
                        dleaf = vpd
                    else:
                        dleaf = (Evap[ileaf] * \
                                 pressure / gsw[ileaf]) * c.PA_2_KPA # kPa


                    # Check for convergence...?
                    if math.fabs(Tleaf - new_tleaf) < 0.02:
                        Tcan[ileaf] = Tleaf
                        break

                    if iter > self.iter_max:
                        #raise Exception('No convergence: %d' % (iter))
                        An[ileaf] = 0.0
                        gsw_c[ileaf] = 0.0
                        Evap[ileaf] = 0.0
                        break

                    # Update temperature & do another iteration
                    Tleaf = new_tleaf
                    Tleaf_K = Tleaf + c.DEG_2_KELVIN
                    Tcan[ileaf] = Tleaf

                    iter += 1

        return (An, Evap, Tcan, apar, lai_leaf)




    def calc_leaf_temp(self, p, tleaf=None, tair=None, gsc=None,
                       vpd=None, pressure=None, wind=None, rnet=None,
                       lai=None, trans=None):
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
        vpd : float
            Vapour pressure deficit (kPa, needs to be in Pa, see conversion
            below)
        pressure : float
            air pressure (using constant) (Pa)
        wind : float
            wind speed (m s-1)
        trans : float
            transpiration (mol m-2 s-1)

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

        (grn, gh, gbH, gw, gb) = self.PM.calc_conductances(p, tair_k, tleaf,
                                                           tair, wind, gsc,
                                                           cmolar, lai)

        # latent heat of water vapour at air temperature (J mol-1)
        lambda_et = (c.H2OLV0 - 2.365E3 * tair) * c.H2OMW

        # slope of sat. water vapour pressure (e_sat) to temperature curve
        # (Pa K-1), note kelvin conversion in func
        slope = (calc_esat(tair + 0.1) - calc_esat(tair)) / 0.1

        # D6 in Leuning. NB I'm doubling conductances, see note below E5.
        # Leuning isn't explicit about grn but I think this is right
        # NB the units or grn and gbH are mol m-2 s-1 and not m s-1, but it
        # cancels.
        Y = 1.0 / (1.0 + (2.0 * grn) / (2.0 * gbH))

        # sensible heat exchanged between leaf and surroundings
        H = rnet - lambda_et * trans # W m-2

        num = H
        den = c.CP * air_density * gh + lambda_et * slope * (gb / cmolar)
        new_Tleaf = tair + num / den

        # Update leaf temperature:
        new_tleaf_k = tleaf_k + (new_Tleaf + c.DEG_2_KELVIN)

        # Update net radiation for canopy
        rnet -= c.CP * c.AIR_MASS * (new_tleaf_k - tair_k) * grn

        return (new_Tleaf)
