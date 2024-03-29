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
__version__ = "1.0 (28.09.2021)"
__email__   = "mdekauwe@gmail.com"


class ProfitMax(object):
    """
    Sperry model assumes that plant maximises the normalised (0-1) difference
    between the relative gain and relative hydraulic risk

    Implementation broadly follows CABLE implementation, albeit without energy
    balance.

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

    def __init__(self, params=None, met_timestep=30., resolution=1000,
                 derive_weibull_params=False, previous_p=None):

        self.p = params
        self.met_timestep = met_timestep
        self.timestep_sec = 60. * self.met_timestep

        if derive_weibull_params:
            (self.b, self.c) = self.get_weibull_params()
        else:
            self.b = params.b
            self.c = params.c

        self.Kmax = params.Kmax

        # Critical soil–plant hydraulic cond below which cavitation occurs
        self.Kcrit = 0.05 * self.Kmax
        self.resolution = resolution # number of water potential samples
        self.zero = 1.0E-17

    def optimisation(self, params, F, psi_soil, vpd, Cs, Tleaf, Par, press,
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
        p_crit = -self.b * np.log(self.Kmax / self.Kcrit)**(1.0 / self.c)

        p = np.zeros(self.resolution)
        Ci = np.empty_like(p)
        an_leaf = np.empty_like(p)

        Vcmax = F.peaked_arrh(params.Vcmax25, params.Eav, Tleaf,
                              params.deltaSv, params.Hdv)
        Jmax = F.peaked_arrh(params.Jmax25, params.Eaj, Tleaf, params.deltaSj,
                             params.Hdj)
        Rd = 0.015 * Vcmax
        gamma_star = F.arrh(params.gamstar25, params.Eag, Tleaf)
        Km = F.calc_michaelis_menten_constants(params, Tleaf)
        J = F.calc_electron_transport_rate(params, Par, Jmax)
        Vj = J / 4.0

        # Generate a sequence of Ci's that we will solve the optimisation
        # model for, range btw gamma_star and Cs. umol mol-1
        lower = gamma_star
        upper = Cs
        for k in range(self.resolution):
            Ci[k] = lower + float(k) * (upper - lower) / float(self.resolution-1)

        # Calculate the sunlit/shaded A_leaf (i.e. scaled up), umol m-2 s-1
        Ac = F.assim(Ci, gamma_star, a1=Vcmax, a2=Km)
        Aj = F.assim(Ci, gamma_star, a1=Vj, a2=2.0*gamma_star)
        A = -self.QUADP(1.0-1E-04, Ac+Aj, Ac*Aj)
        an_leaf = A - Rd # Net photosynthesis, umol m-2 s-1

        # Use an_leaf to infer gsc_sun/sha. NB. An is the scaled up values
        # via scalex applied to Vcmax/Jmax
        gsc = an_leaf / np.maximum(1.e-6, Cs - Ci) # mol CO2 m-2 s-1

        # Assuming perfect coupling, infer E_sun/sha from gsc. NB. as we're
        # iterating, Tleaf will change and so VPD, maintaining energy
        # balance
        e_leaf = gsc * c.GSC_2_GSW / press * vpd # mol H2O m-2 s-1

        # MPa
        p = self.calc_psi_leaf(psi_soil, e_leaf, scalex)

        # Plant hydraulic conductance (mmol m-2 leaf s-1 MPa-1)
        kcmax = self.Kmax * self.get_xylem_vulnerability(psi_soil)

        # Soil–plant hydraulic conductance at canopy xylem pressure,
        # mmol m-2 s-1 MPa-1
        Kc = self.Kmax * self.get_xylem_vulnerability(p)

        # normalised gain (-)
        gain = an_leaf / np.max(an_leaf)

        # Ensure we don't check for profit in bad psi_leaf search space
        gain = np.ma.masked_where(np.logical_or(p>=psi_soil, p<=p_crit), gain)

        # normalised cost (-)
        cost = (kcmax - Kc) / (kcmax - self.Kcrit)

        # Locate maximum profit
        profit = gain - cost

        # Ensure we don't check for profit in bad psi_leaf search space
        profit = np.ma.masked_where(np.logical_or(p>=psi_soil, p<=p_crit), profit)

        idx = np.argmax(profit)

        opt_a = an_leaf[idx] # umol m-2 s-1
        opt_gsc = gsc[idx] # mol CO2 m-2 s-1
        opt_gsw = opt_gsc * c.GSC_2_GSW # mol H2O m-2 s-1
        opt_e = e_leaf[idx] # mol H2O m-2 s-1
        opt_p = p[idx] # MPa

        return opt_a, opt_gsw, opt_gsc, opt_e, opt_p

    def calc_psi_leaf(self, psi_soil, e_leaf, scalex):
        # Calculation the approximate matching psi_leaf from the transpiration
        # array

        MOL_TO_MMOL = 1E3

        # Rescale from canopy to leaf..as e_leaf is E_sun/sha i.e. big-leaf to
        # unit leaf, mmol m-2 s-1
        eleaf_mmol = e_leaf * MOL_TO_MMOL / scalex

        # Infer the matching leaf water potential (MPa).
        psi_leaf = psi_soil - eleaf_mmol / self.Kmax

        return psi_leaf

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
        #e_leaf_test = np.empty_like(p)
        # integrate over the full range of water potentials from psi_soil to
        # e_crit
        for i in range(len(p)):
            e_leaf[i], err = quad(self.get_xylem_vulnerability, p[i], p[0])

        #for i in range(len(p)):
        #    e_leaf_test[i] = self.integrate(self.get_xylem_vulnerability,
        #                                    len(p), p[i], p[0])


        #plt.plot(e_leaf, "ro")
        #plt.plot(e_leaf_test, "ob")
        #plt.show()
        #sys.exit()

        # mol m-2 s-1
        e_leaf[e_leaf > self.zero] *= self.Kmax * c.MMOL_2_MOL

        return np.maximum(self.zero, e_leaf)

    def integrate(self, f, N, a, b):
        """
        Approximate the integration with the mid-point rule
        """

        value = 0.
        value2 = 0.
        for n in range(1, N+1):
            value += f( a + ((n - 0.5) * ((b - a) / float(N))) )
        value2 = ((b - a) / float(N)) * value

        return value2

    def get_xylem_vulnerability(self, p):
        """
        Calculate the vulnerability to cavitation using a Weibull function

        References:
        -----------
        * Neufeld H.S., Grantz D.A., Meinzer F.C., Goldstein G., Crisosto G.M.
          and Crisosto C. (1992) Genotypic variability in vulnerability of leaf
          xylem to cavitation in water-stressed and well-irrigated sugarcane.
          Plant Physiology 100, 1020–1028.


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
                             np.exp(-(-p / self.b)**self.c))

        return weibull

    def get_a_and_ci(self, params, F, gsc, ca, tleafK, par, press, scalex,
                     tol=1E-12):
        """
        Find the matching An and Ci for a given gsc.

        Parameters:
        -----------
        params : struct
            contains all the model params
        F : class
            class to control photosynthesis model
        gsc : float
            stomatal conductance to CO2, mol CO2 m-2 s-1
        ca : float
            co2 concentration, umol mol-1
        tleafK : float
            leaf temperature, K
        par : float
            photosynthetically active radiation, umol m-2 s-1
        press : float
            air pressure, kPa
        scalex : float
            scaler to transform leaf to big leaf

        Returns:
        --------
        ci_new : float
            interceullular CO2 concentration, umol mol-1
        an_new : float
            net leaf assimilation rate, umol m-2 s-1
        """
        gamma_star = F.arrh(params.gamstar25, params.Eag, tleafK) # umol m-2 s-1

        min_ci = gamma_star
        max_ci = ca # umol m-2 s-1
        an_new  = 0.0

        while True:
            ci_new = 0.5 * (max_ci + min_ci) # umol mol-1

            # umol m-2 s-1
            an = F.photosynthesis_given_ci(params, Ci=ci_new, Tleaf=tleafK,
                                           Par=par, scalex=scalex)


            gsc_new = an / (ca - ci_new) # mol m-2 s-1

            if (abs(gsc_new - gsc) / gsc < tol):
                an_new = an # umol m-2 s-1
                break

            # narrow search space, shift min up
            elif (gsc_new < gsc):
                min_ci = ci_new # umol mol-1

            # narrow search space, shift max down
            else:
                max_ci = ci_new # umol mol-1

            if (abs(max_ci - min_ci) < tol):
                an_new = an # umol m-2 s-1
                break

        return ci_new, an_new

    def get_weibull_params(self):
        """
        Calculate the Weibull sensitivity (b) and shape (c) parameters
        """
        p12 = self.p.p12
        p50 = self.p.p50
        p88 = self.p.p88

        if p12 is not None and p50 is not None:
            px1 = p12
            x1 = 12. / 100.
            px2 = p50
            x2 = 50. / 100.
        elif p12 is not None and p88 is not None:
            px1 = p12
            x1 = 12. / 100.
            px2 = p88
            x2 = 88. / 100.
        elif p50 is not None and p88 is not None:
            px1 = p50
            x1 = 50. / 100.
            px2 = p88
            x2 = 88. / 100.

        b = px1 / ((-np.log(1 - x1))**(1. / c))

        num = np.log(np.log(1. - x1) / np.log(1. - x2))
        den = np.log(px1) - np.log(px2)
        c = num / den

        return b, c


    def QUADP(self, A,B,C):
        # Solves the quadratic equation - finds larger root.
        d = B*B - 4.0 * A * C # discriminant
        d = np.where(d<0.0, 0.0, d)

        root = (- B + np.sqrt(B*B - 4*A*C)) / (2.*A)
        root = np.where(np.logical_and(A == 0.0, B == 0.0), 0.0, root)
        root = np.where(np.logical_and(A == 0.0, B > 0.0), -C/B, root)

        return root
