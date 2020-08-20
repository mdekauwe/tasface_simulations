#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, sin, exp, sqrt, acos, asin
import random
import math

import constants as c
from photosynthesis import get_a_ci

__author__  = "Martin De Kauwe"
__version__ = "1.0 (19.08.2020)"
__email__   = "mdekauwe@gmail.com"


class ProfitMax(object):
    """Iteratively solve leaf temp, Ci, gs and An."""

    def __init__(self, Vcmax25, Jmax25, Kmax, b_plant, c_plant, hours_in_day):

        self.Vcmax25 = Vcmax25
        self.Jmax25 = Jmax25
        self.Kmax = Kmax # kg m-2 s-1 MPa-1
        self.b_plant = b_plant # sensitivity of VC, MPa
        self.c_plant = c_plant # shape of VC, [-]
        self.hours_in_day = hours_in_day
        self.laba = 1000.


    def optimisation(self, psi_soil, vpd, ca, tair, par, press, lai,
                     scalex):
        """
        write something



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
            optimised E, mol H2O m-2 s-1
        opt_p : float
            optimised p_leaf (really total plant), MPa
        """
        niter = 101

        store_e = np.zeros(0)
        store_k = np.zeros(0)
        store_a = np.zeros(0)
        store_p = np.zeros(0)
        store_g = np.zeros(0)

        # Transpiration (per unit leaf) max before hydraulic failure (e_crit)
        # kg H2O 30 min-1 m-2 (leaf)
        e_crit = self.get_e_crit(psi_soil)

        if self.hours_in_day == 24:
            conv = c.KG_TO_G * c.G_WATER_2_MOL_WATER * c.HR_2_SEC
        else:
            conv = c.KG_TO_G * c.G_WATER_2_MOL_WATER * c.HLFHR_2_SEC *

        de = 1.0
        step = 0.01

        for i in range(niter):

            # Increment transpiration from zero (no cuticular conductance) to
            # its maximum (e_crit)
            e_leaf = i * step * e_crit
            p = self.get_p_leaf(eleaf, psi_soil)

            #sys.exit()
            # Convert e (kg m-2 30min-1) leaf to mol H2O m-2 s-1
            if e_leaf > 0.0:

                # Scale Eleaf to Ecanopy, mol H20 m-2 s-1
                e_canopy = e_leaf * conv * lai

                # assume perfect coupling
                gsw = e_canopy / vpd * press  # mol H20 m-2 s-1
                gsc = gsw * c.GSW_2_GSC   # mol CO2 m-2 s-1
                g = gsc * (1.0 / press * c.KPA_2_PA) # convert to umol Pa-1
            else:
                e_canopy = 0.0
                gsw = 0.0
                gsc = 0.0
                g  = 0.0

            # One issue here is that this function will scale up A via scalex,
            # but not gs
            ci,a = get_a_ci(self.Vcmax25, self.Jmax25, 2.5, g, ca, tair,
                            par, scalex)

            e_de = e_leaf + de
            p_de = self.get_p_leaf(e_de, psi_soil)
            k = de / (p_de - p)

            store_k = np.append(store_k, k)
            store_a = np.append(store_a, a)
            store_p = np.append(store_p, p)
            store_e = np.append(store_e, e_canopy)
            store_g = np.append(store_g, gsw)

        # Locate maximum profit
        gain = store_a / np.max(store_a)
        risk = 1.0 - store_k / np.max(store_k)
        profit = gain - risk

        idx = np.argmax(profit)
        opt_a = store_a[idx]
        opt_gsw = store_g[idx]
        opt_gsc = opt_gsw * c.GSW_2_GSC   # mol CO2 m-2 s-1
        opt_e = store_e[idx]
        opt_p = store_p[idx]

        return opt_a, opt_gsw, opt_gsc, opt_e, opt_p

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
            weibull = np.exp(-1.0 * (p / self.b_plant)**self.c_plant)

            # Whole plant hydraulic conductance, including vulnerability to
            # cavitation, kg timestep (e.g. 30 min-1)-1 MPa-1 m-2
            Kplant = max(Kmin, self.Kmax * weibull * float(N))

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
        p_crit = self.b_plant * np.log(1000.0)**(1.0 / self.c_plant) # MPa

        if self.hours_in_day == 24:
            e_min = 0.0
            e_max = 0.4  # ~10 mm/day to kg m-2 hour-1
            e_crit = 0.2 # ~5 mm/day to kg m-2 hour-1
        else:
            e_min = 0.0
            e_max = 0.2  # ~10 mm/day to kg m-2 30 min-1
            e_crit = 0.1 # ~5 mm/day to kg m-2 30 min-1

        while True:
            p = self.get_p_leaf(e_max, psi_soil) # MPa
            if p < p_crit:
                e_max *= 2.0
            else:
                break

        while True:
            e = 0.5 * (e_max + e_min)
            p = self.get_p_leaf(e, psi_soil)

            if abs(p - p_crit) < tol or (e_max - e_min) < tol:
                e_crit = e
                break

            if p > p_crit:
                e_max = e
            else:
                e_min = e

        # kg H2O 30 min-1 m-2 (leaf area)
        return e_crit
