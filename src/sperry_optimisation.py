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
    between the relative gain and relative hydraulic risk

    Implementation broadly follows Manon's code.

    References:
    -----------
    * Sperry JS, Venturas MD, Anderegg WRL, Mencuccini M, Mackay DS,
      Wang Y, Love DM. 2017. Predicting stomatal responses to the environment
      from the optimization of photosynthetic gain and hydraulic cost.
      Plant, Cell & Environment 40: 816–830.
    * Sabot, M.E.B., De Kauwe, M.G., Pitman, A.J., Medlyn, B.E., Verhoef, A.,
      Ukkola, A.M. and Abramowitz, G. (2020), Plant profit maximization
      improves predictions of European forest responses to drought. New Phytol,
      226: 1638-1655. doi:10.1111/nph.16376
    """

    def __init__(self, params=None, met_timestep=30., resolution=100):

        self.p = params
        self.met_timestep = met_timestep
        self.timestep_sec = 60. * self.met_timestep

        self.b_plant = params.b_plant
        self.c_plant = params.c_plant
        self.Kmax = params.Kmax
        self.crit_ratio = 0.05

        # Critical soil–plant hydraulic cond below which cavitation occurs
        self.Kcrit = 0.01 * self.Kmax
        self.resolution = resolution # number of water potential samples
        self.zero = 1.0E-17

    def optimisation(self, params, F, psi_soil, vpd, ca, tleafK, par, press,
                     lai, scalex):
        """
        Optimisation wrapper

        Parameters:
        -----------
        params : struct
            contains all the model params
        F : class
            class to control photosynthesis model
        psi_soil : float
            soil water potential, MPa
        vpd : float
            vapour pressure deficit, kPa
        ca : float
            co2 concentration, umol mol-1
        tleafK : float
            leaf temperature, K
        par : float
            photosynthetically active radiation, umol m-2 s-1
        press : float
            air pressure, kPa
        lai : float
            sunlit or shaded LAI, m2 m-2
        scalex : float
            scaler to transform leaf to big leaf

        Returns:
        -----------
        opt_a : float
            optimised A, umol m-2 s-1
        opt_gsw : float
            optimised gsw, mol H2O m-2 s-1
        opt_gsc : float
            optimised gsc, mol CO2 m-2 s-1
        opt_e : float
            optimised E, mol H2O m-2 s-1
        opt_p : float
            optimised p_leaf (really total plant), MPa
        """
        # Canopy xylem pressure (P_crit) MPa, beyond which tree
        # desiccates (Ecrit), MPa
        P_crit = - self.b_plant * \
                    np.log(1. / self.crit_ratio)**(1. / self.c_plant)

        p = np.linspace(psi_soil, P_crit, self.resolution)
        ci = np.empty_like(p)
        a_canopy = np.empty_like(p)

        # Calculate transpiration for every water potential, integrating
        # vulnerability to cavitation, mol H20 m-2 s-1 (leaf)
        e_leaf = self.calc_transpiration(p)

        # Scale to the sunlit or shaded fraction of the canopy, mol H20 m-2 s-1
        e_canopy = e_leaf * lai

        # assuming perfect coupling ... will fix
        gsw = e_canopy / vpd * press # mol H20 m-2 s-1
        gsc = gsw * c.GSW_2_GSC # mol CO2 m-2 s-1

        # For every gsc/psi_leaf get a match An and Ci
        for i in range(len(p)):
            (ci[i], a_canopy[i]) = self.get_a_and_ci(gsc[i], ca, tleafK, par,
                                                     press, scalex, params, F)

        # Soil–plant hydraulic conductance at canopy xylem pressure,
        # mmol m-2 s-1 MPa-1
        K = self.Kmax * self.get_xylem_vulnerability(p)

        # normalised cost (-)
        cost = (self.Kmax - K) / (self.Kmax - self.Kcrit)
        #cost = 1.0 - K / np.max(K)

        # normalised gain (-)
        gain = a_canopy / np.max(a_canopy)

        # Locate maximum profit
        profit = gain - cost
        idx = np.argmax(profit)

        opt_a = a_canopy[idx] # umol m-2 s-1
        opt_gsw = gsw[idx] # mol H2O m-2 s-1
        opt_gsc = opt_gsw * c.GSW_2_GSC   # mol CO2 m-2 s-1
        opt_e = e_canopy[idx] # mol H2O m-2 s-1
        opt_p = p[idx] # MPa

        return opt_a, opt_gsw, opt_gsc, opt_e, opt_p

    def calc_transpiration(self, p):
        """
        At steady-state, transpiration is the integral of the plant's
        vulnerability curve from zero (no cuticular conductance) to its \
        maximum (e_crit) (Sperry & Love 2015). By integrating across the
        soil–plant vulnerability curve, the relation between transpiration and
        a given total pressure drop can be found.

        References:
        ----------
        * Sperry J.S. & Love D.M. (2015) Tansley review: What plant hydraulics
          can tell us about plant responses to climate-change droughts.
          New Phytologist 207, 14–27.

        Parameters:
        -----------
        p : float
            leaf water potential, MPa

        Returns:
        -----------
        e_leaf : float
            transpiration, mol m-2 s-1

        """
        e_leaf = np.empty_like(p)

        # integrate over the full range of water potentials from psi_soil to
        # e_crit
        for i in range(len(p)):
            e_leaf[i], err = quad(self.get_xylem_vulnerability, p[i], p[0])

        # mol m-2 s-1
        e_leaf[e_leaf > self.zero] *= self.Kmax * c.MMOL_2_MOL

        return np.maximum(self.zero, e_leaf)


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
            vulnerability, (-)

        """
        weibull = np.maximum(self.zero,
                             np.exp(-(-p / self.b_plant)**self.c_plant))

        return weibull

    def get_a_and_ci(self, gsc, ca, tleafK, par, press, scalex, params, F,
                     tol=1E-12):

        ci_new  = 0.0
        an_new  = 0.0

        # gamma star, umol m-2 s-1
        min_ci = F.arrh(params.gamstar25, params.Eag, tleafK)
        max_ci = ca # umol m-2 s-1

        while True:
            ci_new = 0.5 * (max_ci + min_ci) # umol mol-1

            # umol m-2 s-1
            an = F.photosynthesis_given_ci(params, Ci=ci_new, Tleaf=tleafK,
                                           Par=par, scalex=scalex)

            gsc_new = an / (ca - ci_new) # mol m-2 s-1

            if (abs(gsc_new - gsc) / gsc < tol):
                an_new = an # umol m-2 s-1
                break

            elif (gsc_new < gsc):
                min_ci = ci_new # umol mol-1

            else:
                max_ci = ci_new # umol mol-1

            if (abs(max_ci - min_ci) < tol):
                an_new = an # umol m-2 s-1
                break

        return ci_new, an_new
