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
from photosynthesis import get_a_ci
from penman_monteith_leaf import PenmanMonteith
from radiation import spitters
from radiation import calculate_absorbed_radiation
from radiation import calculate_cos_zenith, calc_leaf_to_canopy_scalar

__author__  = "Martin De Kauwe"
__version__ = "1.0 (09.11.2018)"
__email__   = "mdekauwe@gmail.com"


class Canopy(object):
    """Iteratively solve leaf temp, Ci, gs and An."""

    def __init__(self, p, peaked_Jmax=True, peaked_Vcmax=True, model_Q10=True,
                 iter_max=100):

        self.p = p

        self.peaked_Jmax = peaked_Jmax
        self.peaked_Vcmax = peaked_Vcmax
        self.model_Q10 = model_Q10
        self.iter_max = iter_max


    def main(self, tair, par, vpd, wind, pressure, Ca, doy, hod,
             lai, psi_soil, kmax, b_plant, c_plant, rnet=None, Vcmax25=None,
             Jmax25=None, beta=None):
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

        PM = PenmanMonteith()

        An = np.zeros(2) # sunlit, shaded
        gsc = np.zeros(2)  # sunlit, shaded
        et = np.zeros(2) # sunlit, shaded
        Tcan = np.zeros(2) # sunlit, shaded
        lai_leaf = np.zeros(2)
        sw_rad = np.zeros(2) # VIS, NIR
        tcanopy = np.zeros(2)

        opt_a = np.zeros(2)
        opt_g = np.zeros(2)
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
                Cs = Ca * c.umol_to_mol * pressure
                press = pressure * c.PA_2_KPA
                Tleaf = tair
                Tleaf_K = Tleaf + c.DEG_2_KELVIN


                iter = 0
                while True:

                    if scalex[ileaf] > 0.:

                        (opt_a[ileaf],
                         opt_g[ileaf],
                         opt_e[ileaf],
                         opt_p[ileaf]) = self.sperry_optimisation(psi_soil,
                                                                  dleaf, Cs,
                                                                  Tleaf,
                                                                  apar[ileaf],
                                                                  press,
                                                                  lai_leaf[ileaf],
                                                                  Vcmax25,
                                                                  Jmax25, kmax,
                                                                  b_plant, c_plant,
                                                                  scalex[ileaf])

                        print("here")
                        print(opt_a[ileaf], opt_g[ileaf])

                        # Put Manon's funcs in
                        # Calc leaf temp




                        sys.exit()

                        #(An[ileaf],
                        # gsc[ileaf]) = F.photosynthesis(self.p, Cs=Cs,
                        #                                Tleaf=Tleaf_K,
                        #                                Par=apar[ileaf],
                        #                                vpd=dleaf,
                        #                                scalex=scalex[ileaf],
                        #                                Vcmax25=Vcmax25,
                        ##                                Jmax25=Jmax25,
                        #                                beta=beta)
                    else:
                        An[ileaf], gsc[ileaf] = 0., 0.

                    # Calculate new Tleaf, dleaf, Cs
                    (new_tleaf, et[ileaf],
                     le_et, gbH, gw) = self.calc_leaf_temp(self.p, PM, Tleaf,
                                                           tair, gsc[ileaf],
                                                           None, vpd,
                                                           pressure, wind,
                                                           rnet=qcan[ileaf],
                                                           lai=lai_leaf[ileaf])

                    gbc = gbH * c.GBH_2_GBC
                    if gbc > 0.0 and An[ileaf] > 0.0:
                        Cs = Ca - An[ileaf] / gbc # boundary layer of leaf
                    else:
                        Cs = Ca

                    if np.isclose(et[ileaf], 0.0) or np.isclose(gw, 0.0):
                        dleaf = vpd
                    else:
                        dleaf = (et[ileaf] * pressure / gw) * c.PA_2_KPA # kPa

                    # Check for convergence...?
                    if math.fabs(Tleaf - new_tleaf) < 0.02:
                        Tcan[ileaf] = Tleaf
                        break

                    if iter > self.iter_max:
                        #raise Exception('No convergence: %d' % (iter))
                        An[ileaf] = 0.0
                        gsc[ileaf] = 0.0
                        et[ileaf] = 0.0
                        break

                    # Update temperature & do another iteration
                    Tleaf = new_tleaf
                    Tleaf_K = Tleaf + c.DEG_2_KELVIN
                    Tcan[ileaf] = Tleaf

                    iter += 1

        return (An, et, Tcan, apar, lai_leaf)

    def sperry_optimisation(self, psi_soil, vpd, ca, tair, par, press, lai,
                            Vcmax25, Jmax25, kmax, b_plant, c_plant, scalex):
        """
        Optimise A, g assuming a single whole-plant vulnerability. Note e to gs
        assumes perfect coupling and no energy balance.

        Parameters:
        -----------
        psi_soil : float
            soil water potential, MPa
        vpd : float
            vapour pressure deficit, kPa
        tair : float
            air temperature, deg C
        par : float
            photosynthetically active radiation, umol m-2 s-1
        press : float
            air pressure, kPa

        Returns:
        -----------
        opt_a : float
            optimised A, umol m-2 s-1
        opt_gw : float
            optimised gw, mol H2O m-2 s-1
        opt_e : float
            optimised E, mmol H2O m-2 s-1
        opt_p : float
            optimised p_leaf (really total plant), MPa
        """

        # kg H2O 30 min-1 m-2 (leaf area)
        e_crit = self.get_e_crit(psi_soil, kmax, b_plant, c_plant)

        de = 1.0

        all_e = np.zeros(0)
        all_k = np.zeros(0)
        all_a = np.zeros(0)
        all_p = np.zeros(0)
        all_g = np.zeros(0)

        laba = 1000.

        for i in range(101):

            # Vary e from 0 to e_crit (0.01 is just partioning step)
            e = i * 0.01 * e_crit
            p = self.get_p_leaf(e, psi_soil, kmax, b_plant, c_plant)

            #sys.exit()
            # Convert e (kg m-2 30min-1) leaf to mol H2O m-2 s-1
            if e > 0.0:

                #emol = e * (c.KG_TO_G * c.G_WATER_2_MOL_WATER * c.HLFHR_2_SEC / laba)
                emol = e * (c.KG_TO_G * c.G_WATER_2_MOL_WATER * c.HR_2_SEC / laba)

                # assume perfect coupling

                gh = emol / vpd * press  # mol H20 m-2 s-1
                #print(emol, gh, vpd, press)

                gc = gh * c.GSW_2_GSC
                g = gc * (1.0 / press * c.KPA_2_PA) # convert to Pa
            else:
                emol = 0.0
                gh = 0.0
                gc = 0.0
                g  = 0.0



            ci,a = get_a_ci(Vcmax25, Jmax25, 2.5, g, ca, tair, par, scalex)

            e_de = e + de
            p_de = self.get_p_leaf(e_de, psi_soil, kmax, b_plant, c_plant)
            k = de / (p_de - p)

            all_k = np.append(all_k, k)
            all_a = np.append(all_a, a)
            all_p = np.append(all_p, p)
            all_e = np.append(all_e, emol * c.mol_2_mmol)
            all_g = np.append(all_g, gc * c.GSC_2_GSW)

        # Locate maximum profit
        gain = all_a / np.max(all_a)
        risk = 1.0 - all_k / np.max(all_k)
        profit = gain - risk
        idx = np.argmax(profit)
        opt_a = all_a[idx]
        opt_gw = all_g[idx]
        opt_e = all_e[idx]
        opt_p = all_p[idx]

        return opt_a, opt_gw, opt_e, opt_p

    def get_p_leaf(self, transpiration, psi_soil, kmax, b_plant, c_plant):
        """
        Integrate vulnerability curve across soil–plant system. This is a
        simplification, as opposed to splitting between root-zone, stem & leaf.

        Parameters:
        -----------
        transpiration : float
            kg H2O 30 min-1 m-2 (leaf area)
        psi_soil : float
            soil water potential, MPa
        Returns:
        -----------
        p_leaf : float
            integrated vulnerability curve, MPa
        """
        dp = 0.0
        p = psi_soil # MPa
        N = 20 # P range
        for i in range(N): # iterate through the P range

            # Vulnerability to cavitation
            weibull = np.exp(-1.0 * (p / b_plant)**c_plant)

            # Hydraulic conductance of the element, including vulnerability to
            # cavitation
            k = max(1E-12, kmax * weibull * float(N))
            dp += transpiration / k # should have a gravity, height addition
            p = psi_soil + dp

        p_leaf = p

        # MPa
        return p_leaf

    def get_e_crit(self, psi_soil, kmax, b_plant, c_plant):
        """
        Calculate the maximal E beyond which the tree desiccates due to
        hydraulic failure (e_crit)
        Parameters:
        -----------
        psi_soil : float
            soil water potential, MPa
        Returns:
        -----------
        e_crit : float
            kg H2O 30 min-1 m-2 (basal area)
        """

        # P at Ecrit, beyond which tree desiccates
        p_crit = b_plant * np.log(1000.0) ** (1.0 / c_plant) # MPa
        e_min = 0.0
        e_max = 100.0
        e_crit = 50.0

        while True:
            p = self.get_p_leaf(e_max, psi_soil, kmax, b_plant, c_plant) # MPa
            if p < p_crit:
                e_max *= 2.0
            else:
                break

        while True:
            e = 0.5 * (e_max + e_min)
            p = self.get_p_leaf(e, psi_soil, kmax, b_plant, c_plant)
            if abs(p - p_crit) < 1E-3 or (e_max - e_min) < 1E-3:
                e_crit = e
                break
            if p > p_crit:
                e_max = e
            else:
                e_min = e

        # kg H2O 30 min-1 m-2 (leaf area)
        return e_crit

    def calc_leaf_temp(self, p, PM=None, tleaf=None, tair=None, gsc=None,
                       par=None, vpd=None, pressure=None, wind=None, rnet=None,
                       lai=None):
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

        (grn, gh, gbH, gw) = PM.calc_conductances(p, tair_k, tleaf, tair,
                                                  wind, gsc, cmolar, lai)

        if np.isclose(gsc, 0.0):
            et = 0.0
            le_et = 0.0
        else:
            (et, le_et) = PM.calc_et(tleaf, tair, vpd, pressure, wind, par,
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

        # Update leaf temperature:
        new_tleaf_k = tleaf_k + (new_Tleaf + c.DEG_2_KELVIN)

        # Update net radiation for canopy
        rnet -= c.CP * c.AIR_MASS * (new_tleaf_k - tair_k) * grn

        return (new_Tleaf, et, le_et, gbH, gw)
