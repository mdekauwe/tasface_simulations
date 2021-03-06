#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, sin, exp, sqrt, acos, asin
import random
import math
from scipy.integrate import quad
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

        self.b_plant = params.b_plant
        self.c_plant = params.c_plant
        self.Kmax = params.Kmax
        self.Kcrit = 0.01 * self.Kmax

    def optimisation_manon(self, params, F, psi_soil, vpd, ca, tleafK, par,
                           press, lai, scalex):

        
        ratiocrit = 0.05
        Pcrit = - self.b_plant * np.log(1. / ratiocrit) ** (1. / self.c_plant)  # MPa
        P = np.linspace(psi_soil, Pcrit, 200)
        trans = self.transpiration(P, self.Kmax, self.b_plant, self.c_plant)
        e_canopy = trans * lai

        gsw = e_canopy / vpd * press # mol H20 m-2 s-1
        gsc = gsw * c.GSW_2_GSC # mol CO2 m-2 s-1

        ci = np.empty_like(P)
        a_canopy = np.empty_like(P)
        for i in range(len(P)):
            (ci[i], a_canopy[i]) = self.get_a_and_ci(gsc[i], ca, tleafK, par,
                                                     press, scalex, params, F)


        k = self.Kmax * self.f(P, self.b_plant, self.c_plant)  # mmol s-1 m-2 MPa-1

        # cost, from kmax @ Ps to kcrit @ Pcrit
        cost = (self.Kmax - k) / (self.Kmax - self.Kcrit)  # normalized, unitless

        # Locate maximum profit
        gain = a_canopy / np.max(a_canopy)

        profit = gain - cost
        idx = np.argmax(profit)

        opt_a_canopy = a_canopy[idx]
        opt_gsw_canopy = gsw[idx]
        opt_gsc_canopy = opt_gsw_canopy * c.GSW_2_GSC   # mol CO2 m-2 s-1
        opt_e_canopy = trans[idx]
        opt_p = P[idx]

        return opt_a_canopy, opt_gsw_canopy, opt_gsc_canopy, opt_e_canopy, opt_p

    def transpiration(self, P, kmax, b, c):

        zero = 1.e-17
        FROM_MILI = 1.e-3

        trans = np.empty_like(P)  # empty numpy array of right length

        for i in range(len(P)):  # at Ps, trans=0; at Pcrit, trans=transcrit
            trans[i], err = quad(self.f, P[i], P[0], args=(b, c))
        trans[trans > zero] *= kmax * FROM_MILI  # mol.s-1.m-2

        return np.maximum(zero, trans)

    def f(self, P, b, c):

        zero = 1.e-17
        return np.maximum(zero, np.exp(-(-P / b) ** c))

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

        # Logic is based on "postive" psi_soil
        psi_soil *= -1.

        de = 1.0
        step = 0.01

        for i in range(N):

            # Increment transpiration from zero (no cuticular conductance) to
            # its maximum (e_crit)
            e_leaf = i * step * e_crit

            # Integrated "whole plant" (leaf-xylem) water potential, MPa
            psi_leaf = self.get_p_leaf(e_leaf, psi_soil)

            if e_leaf > 0.0:

                # Convert e (kg m-2 30min-1) leaf to mol H2O m-2 s-1 and scale
                # to the canopy
                e_canopy = e_leaf * conv * lai

                # assuming perfect coupling ... will fix
                gsw = e_canopy / vpd * press # mol H20 m-2 s-1
                gsc = gsw * c.GSW_2_GSC # mol CO2 m-2 s-1

                (ci, a_canopy) = self.get_a_and_ci(gsc, ca, tleafK, par, press,
                                                   scalex, params, F)

            else:
                e_canopy = 0.0
                a_canopy = 0.0
                gsw = 0.0
                gsc = 0.0

            e_de = e_leaf + de
            p_de = self.get_p_leaf(e_de, psi_soil)

            # Soil–plant hydraulic conductance at canopy xylem pressure, dE/dP
            # mol H2O m-2 s-1 Pa-1
            k = de / (p_de - psi_leaf)

            store_k[i] = k
            store_p[i] = psi_leaf # integrated
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
            "whole plant" (leaf-xylem) water potential, MPa, MPa
        """
        dp = 0.0
        N = 20 # P range
        p = abs(psi_soil) # MPa
        Kmin = 1E-12

        for i in range(N): # iterate through the P range

            # Vulnerability to cavitation (-)
            weibull = self.get_xylem_vulnerability(p)

            # Whole plant hydraulic conductance, including vulnerability to
            # cavitation, kg timestep (e.g. 30 min-1)-1 MPa-1 m-2
            Kplant = max(Kmin, self.Kmax * weibull * float(N))

            # Calculate the pressure drop
            dp += transpiration / Kplant # MPa
            p = psi_soil + dp # MPa

        p_leaf = p # MPa

        return p_leaf

    def get_xylem_vulnerability(self, p):
        """
        Calculate the vulnerability to cavitation using a Weibull function

        Parameters:
        -----------
        p : float
            leaf water potential, MPa

        Returns:
        -----------
        weibull : float
            vulnerability [-]

        """
        return np.exp(-1.0 * (p / self.b_plant)**self.c_plant)

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

        if self.met_timestep == 60:
            e_min = 0.0
            e_max = 0.4  # ~10 mm/day to kg m-2 hour-1
            e_crit = 0.2 # ~5 mm/day to kg m-2 hour-1
        else:
            e_min = 0.0
            e_max = 0.2  # ~10 mm/day to kg m-2 30 min-1
            e_crit = 0.1 # ~5 mm/day to kg m-2 30 min-1

        # Canopy xylem pressure (P_crit) MPa, beyond which tree
        # desiccates (Ecrit)
        p_crit = self.b_plant * \
                    np.log(self.Kmax / self.Kcrit)**(1.0 / self.c_plant)

        while True:
            psi_leaf = self.get_p_leaf(e_max, psi_soil) # MPa

            if psi_leaf < p_crit:
                e_max *= 2.0
            else:
                break


        while True:
            e_leaf = 0.5 * (e_max + e_min)
            psi_leaf = self.get_p_leaf(e_leaf, psi_soil)

            if abs(psi_leaf - p_crit) < tol or (e_max - e_min) < tol:
                e_crit = e_leaf
                break

            if psi_leaf > p_crit:
                e_max = e_leaf
            else:
                e_min = e_leaf

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
