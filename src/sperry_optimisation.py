#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, sin, exp, sqrt, acos, asin
import random
import math

import constants as c


__author__  = "Martin De Kauwe"
__version__ = "1.0 (19.08.2020)"
__email__   = "mdekauwe@gmail.com"


class ProfitMax(object):
    """
    Sperry model assumes that plant maximises the normalised (0-1) difference
    between the relative gain (A/Amax) and relative hydraulic risk (1-K/Kmax)

    References:
    -----------
    * Sperry JS, Venturas MD, Anderegg WRL, Mencuccini M, Mackay DS,
      Wang Y, Love DM. 2017. Predicting stomatal responses to the environment
      from the optimization of photosynthetic gain and hydraulic cost.
      Plant, Cell & Environment 40: 816–830.
    """

    def __init__(self, params=None, met_timestep=30.):

        self.p = params
        self.met_timestep = met_timestep
        self.timestep_sec = 60. * self.met_timestep

        if self.met_timestep == 60:
            self.e_min = 0.0
            self.e_max = 0.4  # ~10 mm/day to kg m-2 hour-1
            self.e_crit = 0.2 # ~5 mm/day to kg m-2 hour-1
        else:
            self.e_min = 0.0
            self.e_max = 0.2  # ~10 mm/day to kg m-2 30 min-1
            self.e_crit = 0.1 # ~5 mm/day to kg m-2 30 min-1

    def optimisation(self, params, F, psi_soil, vpd, ca, tleafK, par, press,
                     lai, scalex):
        """
        Optimisation wrapper

        Parameters:
        -----------
        p : struct
            contains all the model params
        F : class
            class to control photosynthesis model
        psi_soil : float
            soil water potential, MPa
        vpd : float
            vapour pressure deficit, kPa
        tleafK : float
            leaf temperature, K
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
            optimised E, mol H2O m-2 s-1
        opt_p : float
            optimised p_leaf (really total plant), MPa
        """
        N = 101
        store_k = np.zeros(N)
        store_p = np.zeros(N)
        store_a_can = np.zeros(N)
        store_e_can = np.zeros(N)
        store_gsw_can = np.zeros(N)


        conv = c.KG_TO_G * c.G_WATER_2_MOL_WATER / self.timestep_sec


        # Transpiration (per unit leaf) max before hydraulic failure (e_crit)
        # kg H2O 30 min-1 m-2 (leaf)
        e_crit = self.get_e_crit(psi_soil)

        de = 1.0
        step = 0.01

        for i in range(N):

            # Increment transpiration from zero (no cuticular conductance) to
            # its maximum (e_crit)
            e_leaf = i * step * e_crit

            # "whole plant" (leaf-xylem) water potential, MPa
            psi_leaf = self.get_p_leaf(e_leaf, psi_soil)

            if e_leaf > 0.0:

                # Convert e (kg m-2 30min-1) leaf to mol H2O m-2 s-1 and scale
                # to the canopy
                e_canopy = e_leaf * conv * lai

                # assuming perfect coupling ... will fix
                gsw = e_canopy / vpd * press # mol H20 m-2 s-1
                gsc = gsw * c.GSW_2_GSC # mol CO2 m-2 s-1

            else:
                e_canopy = 0.0
                gsw = 0.0
                gsc = 0.0
                g_umol_pa  = 0.0

            (ci, a_canopy) = self.get_a_and_ci(gsc, ca, tleafK, par, press,
                                               scalex, params, F)

            e_de = e_leaf + de
            p_de = self.get_p_leaf(e_de, psi_soil)

            # Soil–plant hydraulic conductance at canopy xylem pressure, dE/dP
            # mol H2O m-2 s-1 Pa-1
            k = de / (p_de - psi_leaf)

            store_k[i] = k
            store_p[i] = psi_leaf
            store_a_can[i] = a_canopy
            store_e_can[i] = e_canopy
            store_gsw_can[i] = gsw

        # Locate maximum profit
        gain = store_a_can / np.max(store_a_can)
        risk = 1.0 - store_k / np.max(store_k)
        profit = gain - risk
        idx = np.argmax(profit)
        opt_a_canopy = store_a_can[idx]
        opt_gsw_canopy = store_gsw_can[idx]
        opt_gsc_canopy = opt_gsw_canopy * c.GSW_2_GSC   # mol CO2 m-2 s-1
        opt_e_canopy = store_e_can[idx]
        opt_p = store_p[idx]

        return opt_a_canopy, opt_gsw_canopy, opt_gsc_canopy, opt_e_canopy, opt_p

    def get_p_leaf(self, transpiration, psi_soil):
        """
        Integrate vulnerability curve across soil–plant system. This is a
        simplification, as opposed to splitting between root-zone, stem & leaf.

        Parameters:
        -----------
        transpiration per unit leaf area : float
            kg H2O timestep (e.g. 30 min-1)-1 m-2 (leaf)
        psi_soil : float
            soil water potential, MPa

        Returns:
        -----------
        p_leaf : float
            integrated vulnerability curve, MPa
        """
        dp = 0.0
        N = 20 # P range
        p = psi_soil # MPa
        Kmin = 1E-12

        for i in range(N): # iterate through the P range

            # Vulnerability to cavitation
            weibull = np.exp(-1.0 * \
                            (p / self.p.b_plant)**self.p.c_plant)

            # Whole plant hydraulic conductance, including vulnerability to
            # cavitation, kg timestep (e.g. 30 min-1)-1 MPa-1 m-2
            Kplant = max(Kmin, self.p.Kmax * weibull * float(N))

            # Calculate the pressure drop
            dp += transpiration / Kplant # MPa
            p = psi_soil + dp # MPa

        p_leaf = p # MPa

        return p_leaf

    def get_e_crit(self, psi_soil):
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
            kg H2O timestep (e.g. 30 min-1)-1 m-2 (leaf area)
        """
        tol = 1E-3

        # P at Ecrit, beyond which tree desiccates
        # 1000 <- assumption that p_crit is when kh is 0.1% of the maximum, fix
        p_crit = self.p.b_plant * \
                        np.log(1000.0)**(1.0 / self.p.c_plant) # MPa


        while True:
            p = self.get_p_leaf(self.e_max, psi_soil) # MPa
            if p < p_crit:
                self.e_max *= 2.0
            else:
                break

        while True:
            e = 0.5 * (self.e_max + self.e_min)
            p = self.get_p_leaf(e, psi_soil)

            if abs(p - p_crit) < tol or (self.e_max - self.e_min) < tol:
                e_crit = e
                break

            if p > p_crit:
                self.e_max = e
            else:
                self.e_min = e

        # kg H2O 30 min-1 m-2 (leaf area)
        return e_crit

    def get_a_and_ci(self, gsc, ca, tleafK, par, press, scalex, params, F):

        # convert mol m-2 s-1 to umol m-2 s-1 Pa-1
        gsc *= c.MOL_TO_UMOL / (press * c.KPA_2_PA)

        tol = 1E-12

        ci_new  = 0.0
        an_new  = 0.0

        min_ci  = 2.5 # ~gamma_star Pa
        max_ci  = ca # Pa

        while True:
            ci_new = 0.5 * (max_ci + min_ci) # Pa

            ci_umol_mol = ci_new / (press * c.KPA_2_PA) * c.MOL_TO_UMOL
            an = F.photosynthesis_given_ci(params, Ci=ci_umol_mol, Tleaf=tleafK,
                                           Par=par, scalex=scalex)

            gsc_new = an / (ca - ci_new) # umol m-2 s-1 Pa-1

            if (abs(gsc_new - gsc) / gsc < tol):
                an_new = an
                break

            elif (gsc_new < gsc):
                min_ci = ci_new

            else:
                max_ci = ci_new

            if (abs(max_ci - min_ci) < tol):
                an_new = an
                break

        return ci_new, an_new
