# -*- coding: utf-8 -*-
# --------------------------------
# Name:       lbl_absorption_models.py
# Purpose:    LBL absorption models for transfer calculations.
# Author:     Moritz Loeffler
# Created:    2022-11-28
# Python Version:   3.6
# Version:    1.0
# Last Edit:  2025-08-20
# --------------------------------
"""
Line by line absorption coefficients translated from Rosenkranz, P.W.:
Line-by-line microwave radiative transfer (non-scattering) [software]
(version 2022/08/25), http://cetemps.aquila.infn.it/mwrnet/lblmrt_ns.html
(last access: 19 August 2025)

Rosenkranz 98(R98), Rosenkranz 2020(wvr20/o2r20) and cloud absorption models translated from
IDL code (Ulrich Loehnert et al).

"""

import numpy as np
import xarray as xr
import sys
import os
import json
from typing import List, Optional, Union
import warnings

sys.path.append(os.path.dirname(__file__))
import lbl_functions

# variable definitions
c = 299.792458e6
coeff_dir = os.path.dirname(__file__) + "/coeff"


# RT functions

class Rosenkranz(object):
    """Basic class for all Rosenkranz RT models."""

    # class variables
    H2OM = 2.9915075E-23  # water mass(g)
    O3M = 7.970089E-23  # ozone  mass(g)
    DRYM = 4.80992E-23  # average  dry - air  mass(g)
    grav = .980665  # hPa  cm ^ 2 / g
    RH2O = 4.615228E-3  # H2O  gas  constant(hPa / K  m ^ 3 / g)
    RDRY = 2.870419E-3  # dry - air  gas  constant(hPa / K  m ^ 3 / g)
    GKMK = .029270127  # geopotential  km / K( in dry  air)

    def __init__(self):
        """Only a superclass"""
        pass

    def loadCoeffs(self, fname, var_names: Optional[List[str]] = None, continuum_vars: Optional[List[str]] = None):
        """Open coeff file and store in self."""
        with open(fname, 'r') as file:
            vars = {}
            end = False
            for lineNo, line in enumerate(file):
                if end:
                    if not (continuum_vars is None):
                        cvs = line.split(",")
                        for i in np.arange(len(continuum_vars)):
                            vars[continuum_vars[i]] = float(cvs[i])
                    break
                if "end list" in line:
                    end = True
                    continue
                if lineNo > 0:
                    values = line.split(",")
                    for i in np.arange(len(var_names)):
                        try:
                            vars[var_names[i]].append(float(values[i]))
                        except ValueError:
                            vars[var_names[i]].append("")
                else:
                    if var_names is None:
                        var_names = line.split(",")
                    for i in np.arange(len(var_names)):
                        var_names[i] = var_names[i].strip()
                        vars[var_names[i]] = []
        for var in np.append(continuum_vars, var_names):
            vars[var] = np.array(vars[var])
        self.__dict__ = dict(self.__dict__, **vars)

    def loadCoeffsJSON(self, path):
        """Load coefficients required for O2 absorption."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                try:
                    vars = dict(**json.load(f))
                except json.decoder.JSONDecodeError:
                    print(path)
                    raise
        self.__dict__ = dict(self.__dict__, **vars)


class R98(Rosenkranz):
    """Object, which loads all R98 coefficients and contains lbl rt code."""

    def __init__(self, cont_corr: str = "org", linew_22: str = "org"):
        """Load coefficients."""
        super(R98, self).__init__()
        self.cont_corr = cont_corr
        if linew_22 == "lil05":
            self.loadCoeffsJSON(path=coeff_dir + "/r98/h2o_list_lil_05.json")
        else:
            self.loadCoeffsJSON(path=coeff_dir + "/r98/h2o_list.json")
        self.loadCoeffsJSON(path=coeff_dir + "/r98/o2_list.json")

    def o2(self, ds: xr.Dataset) -> np.array:
        """RETURN POWER ABSORPTION COEFFICIENT DUE TO OXYGEN IN AIR."""
        # WIDTHS IN MHZ/MB
        WB300 = .56
        X = .8

        TH = 300. / ds["T_mean"].values
        TH1 = TH - 1.
        B = TH ** X
        PRESWV = ds["absolute_humidity_mean"] * 1000 * ds["T_mean"] / 217
        PRESDA = (ds["p_mean"] / 100) - PRESWV
        DEN = .001 * (PRESDA * B + 1.1 * PRESWV * TH)
        DENS = .001 * (PRESDA + 1.1 * PRESWV) * TH
        DFNR = WB300 * DEN

        # 1.571e-17 (o16-o16) + 1.3e-19 (o16-o18) = 1.584e-17
        SUM = 1.6E-17 * ds["f"] ** 2 * DFNR / (TH * (ds["f"] ** 2 + DFNR * DFNR))
        for k in np.arange(len(self.F)):
            if k > 0:
                DF = self.W300[k] * DEN
            else:
                DF = self.W300[k] * DENS
            Y = ds["p_mean"] / 100 / 1000 * B * (self.Y300[k] + self.V[k] * TH1)
            STR = self.S300[k] * np.exp(-self.BE[k] * TH1)

            DEL1 = ds["f"] - self.F[k]
            DEL2 = ds["f"] + self.F[k]
            D1 = DEL1 * DEL1 + DF * DF
            D2 = DEL2 * DEL2 + DF * DF
            SF1 = (DF + DEL1 * Y) / D1
            SF2 = (DF - DEL2 * Y) / D2

            SUM = SUM + STR * (SF1 + SF2) * (ds["f"] / self.F[k]) ** 2
        O2ABS = .5034E12 * SUM * PRESDA * TH ** 3 / 3.14159
        O2ABS[O2ABS < 0] = 0
        return O2ABS.values

    def wvr(self, ds: xr.Dataset) -> np.array:
        """COMPUTE ABSORPTION COEF IN ATMOSPHERE DUE TO WATER VAPOR."""
        # ****number of frequencies
        n_f = len(ds["f"])

        if ds["absolute_humidity_mean"] <= 0:
            return np.zeros(n_f)

        # ****LOCAL VARIABLES:
        NLINES = 15
        DF = np.zeros((2, n_f))

        # Initial calculations
        PVAP = (ds["absolute_humidity_mean"].values * 1000) * ds["T_mean"].values / 217
        PDA = (ds["p_mean"].values / 100) - PVAP
        DEN = 3.335E16 * ds["absolute_humidity_mean"].values * 1000
        TI = 300. / ds["T_mean"].values
        TI2 = TI ** (2.5)

        #  continuum Terms
        bf_org = 5.43E-10
        bs_org = 1.8E-8

        bf_mult = 1.0
        bs_mult = 1.0
        if self.cont_corr == "tur09":
            bf_mult = 1.105
            bs_mult = 0.79
        bf = bf_org * bf_mult
        bs = bs_org * bs_mult

        CON = (bf * PDA * TI ** 3 + bs * PVAP * TI ** 7.5) * PVAP * ds["f"] ** 2

        SUM = 0
        for i in np.arange(NLINES):
            WIDTH = self.W3[i] * PDA * TI ** self.X[i] + self.WS[i] * PVAP * TI ** self.XS[i]
            WSQ = WIDTH ** 2
            S = self.S1[i] * TI2 * np.exp(self.B2[i] * (1. - TI))
            DF[0, :] = ds["f"] - self.FL[i]
            DF[1, :] = ds["f"] + self.FL[i]

            # USE CLOUGH'S DEFINITION OF LOCAL LINE CONTRIBUTION
            BASE = WIDTH / (562500. + WSQ)

            # DO FOR POSITIVE AND NEGATIVE RESONANCES
            RES = np.zeros(n_f)
            for j in [0, 1]:
                index = np.abs(DF[j, :]) < 750
                RES[index] = (RES + WIDTH / (DF[j] ** 2 + WSQ) - BASE)[index]

            SUM = SUM + S * RES * (ds["f"] / self.FL[i]) ** 2
        ALPHA = .3183E-4 * DEN * SUM + CON
        return ALPHA.values


class R22(Rosenkranz):
    """Object, which loads all R22 coefficients and contains lbl rt code."""

    def __init__(self):
        """Load coefficients."""
        super(R22, self).__init__()
        self.loadH2OCoeffs()
        self.loadCoeffsJSON(path=coeff_dir + "/r22/o2_list.json")

    def loadH2OCoeffs(self):
        """Load coefficients required for H2O absorption."""
        fname = coeff_dir + "/r22/h2o_sdlist.asc"
        var_names = ["MOLEC", "FL", "S1", "B2", "Wair", "X", "Wself", "XS", "Sair", "Xh", "Sself", "Xhs", "Aair",
                     "Aself",
                     "W2air", "XW2", "W2self", "XW2S", "D2air", "D2self"]
        continuum_vars = ["REFTCON", "CF", "Xcf", "CS", "Xcs"]
        self.loadCoeffs(fname, var_names, continuum_vars)
        self.W0 = self.Wair / 1000.
        self.W0S = self.Wself / 1000.
        self.W2 = self.W2air / 1000.
        self.W2S = self.W2self / 1000.
        self.SH = self.Sair / 1000.
        self.SHS = self.Sself / 1000.
        self.D2 = self.D2air / 1000.
        self.D2S = self.D2self / 1000.
        self.Xh[self.Xh <= 0] = self.X[self.Xh <= 0]
        self.Xhs[self.Xhs <= 0] = self.XS[self.Xhs <= 0]

    def wvr(self, ds: xr.Dataset) -> np.array:
        """COMPUTE ABSORPTION COEF IN ATMOSPHERE DUE TO WATER VAPOR."""
        # ****number of frequencies
        n_f = len(ds["f"])

        if ds["absolute_humidity_mean"] <= 0:
            return np.zeros(n_f)

        # ****LOCAL VARIABLES:
        DF = np.zeros((2, n_f))

        # Initial calculations
        PVAP = (ds["absolute_humidity_mean"].values * 1000) * ds["T_mean"].values * R22.RH2O
        PDA = (ds["p_mean"].values / 100) - PVAP

        # ****CONTINUUM TERMS
        TI = self.REFTCON / ds["T_mean"].values

        #   Xcf and Xcs include 3 for density & stimulated emission
        CON = (self.CF * PDA * TI ** self.Xcf + self.CS * PVAP * TI ** self.Xcs) * PVAP * ds["f"] ** 2

        # ****ADD RESONANCES
        REFTLINE = 296.
        TI = REFTLINE / ds["T_mean"].values
        TILN = np.log(TI)
        TI2 = np.exp(2.5 * TILN)

        SUM = 0
        for i in np.arange(len(self.W2)):
            WIDTH0 = self.W0[i] * PDA * TI ** self.X[i] + self.W0S[i] * PVAP * TI ** self.XS[i]
            if self.W2[i] > 0:
                WIDTH2 = self.W2[i] * PDA * TI ** self.XW2[i] + self.W2S[i] * PVAP * TI ** self.XW2S[i]
            else:
                WIDTH2 = 0
            DELTA2 = self.D2[i] * PDA + self.D2S[i] * PVAP  # DELTA2 assumed independent of T
            SHIFTF = self.SH[i] * PDA * (1. - self.Aair[i] * TILN) * TI ** self.Xh[i]
            SHIFTS = self.SHS[i] * PVAP * (1. - self.Aself[i] * TILN) * TI ** self.Xhs[i]
            SHIFT = SHIFTF + SHIFTS
            WSQ = WIDTH0 ** 2

            S = self.S1[i] * TI2 * np.exp(self.B2[i] * (1. - TI))
            DF[0, :] = ds["f"] - self.FL[i] - SHIFT
            DF[1, :] = ds["f"] + self.FL[i] + SHIFT
            # USE CLOUGH'S DEFINITION OF LOCAL LINE CONTRIBUTION
            BASE = WIDTH0 / (562500. + WSQ)

            # DO FOR POSITIVE AND NEGATIVE RESONANCES
            RES = np.zeros(n_f)
            for j in [0, 1]:
                # speed dependant resonant shape factor, minus base
                if (j == 0) & (WIDTH2 > 0.):
                    index1 = np.abs(DF[j]) < 10. * WIDTH0
                    index2 = (np.abs(DF[j, :]) < 750) & ~index1
                    Xc = (WIDTH0 - 1.5 * WIDTH2 + (DF[j] + 1.5 * DELTA2) * 1j) / (WIDTH2 - DELTA2 * 1j)
                    Xrt = np.sqrt(Xc)
                    pxw = 1.77245385090551603 * Xrt * lbl_functions.dcerror(-np.imag(Xrt), np.real(Xrt))
                    SD = 2. * (1. - pxw) / (WIDTH2 - DELTA2 * 1j)
                    RES[index1] = (RES + np.real(SD) - BASE)[index1]
                    RES[index2] = (RES + WIDTH0 / (DF[j] ** 2 + WSQ) - BASE)[index2]
                else:
                    index = np.abs(DF[j, :]) < 750
                    RES[index] = (RES + WIDTH0 / (DF[j] ** 2 + WSQ) - BASE)[index]
            SUM = SUM + S * RES * (ds["f"] / self.FL[i]) ** 2
        ALPHA = 1.E-10 * ds["absolute_humidity_mean"].values * 1000 * SUM / (np.pi * R22.H2OM) + CON
        return ALPHA.values

    def wvr_fast(self, ds: xr.Dataset) -> np.array:
        """COMPUTE ABSORPTION COEF IN ATMOSPHERE DUE TO WATER VAPOR, vectorized."""
        # ****number of frequencies
        n_f = len(ds["f"])

        if ds["absolute_humidity_mean"] <= 0:
            return np.zeros(n_f)

        # Initial calculations
        PVAP = (ds["absolute_humidity_mean"].values * 1000) * ds["T_mean"].values * R22.RH2O
        PDA = (ds["p_mean"].values / 100) - PVAP

        # ****CONTINUUM TERMS
        TI = self.REFTCON / ds["T_mean"].values

        #   Xcf and Xcs include 3 for density & stimulated emission
        CON = (self.CF * PDA * TI ** self.Xcf + self.CS * PVAP * TI ** self.Xcs) * PVAP * ds["f"] ** 2

        # ****ADD RESONANCES
        REFTLINE = 296.
        TI = REFTLINE / ds["T_mean"].values
        TILN = np.log(TI)
        TI2 = np.exp(2.5 * TILN)
        SUM2 = 0

        WIDTH0 = self.W0 * PDA * TI ** self.X + self.W0S * PVAP * TI ** self.XS
        WIDTH2 = np.where(self.W2 > 0, self.W2 * PDA * TI ** self.XW2 + self.W2S * PVAP * TI ** self.XW2S, 0)
        DELTA2 = self.D2 * PDA + self.D2S * PVAP  # DELTA2 assumed independent of T
        SHIFTF = self.SH * PDA * (1. - self.Aair * TILN) * TI ** self.Xh
        SHIFTS = self.SHS * PVAP * (1. - self.Aself * TILN) * TI ** self.Xhs
        SHIFT = SHIFTF + SHIFTS
        WSQ = WIDTH0 ** 2

        S = self.S1 * TI2 * np.exp(self.B2 * (1. - TI))

        BASE = WIDTH0 / (562500. + WSQ)

        RES = np.zeros((n_f, len(WIDTH0))).T  # Initialize RES matrix
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            for j in [0, 1]:
                if j == 0:
                    DF = np.where(WIDTH2[:, None] > 0 * ds["f"].values,
                                  ds["f"].values - self.FL[:, None] - SHIFT[:, None],
                                  ds["f"].values + self.FL[:, None] + SHIFT[:, None])
                    index1 = np.abs(DF) < 10. * WIDTH0[:, None]
                    index2 = (np.abs(DF) < 750) & ~index1
                    Xc = (WIDTH0[:, None] - 1.5 * WIDTH2[:, None] + (DF + 1.5 * DELTA2[:, None]) * 1j) / (WIDTH2[:, None] - DELTA2[:, None] * 1j)
                    Xrt = np.sqrt(Xc)
                    pxw = 1.77245385090551603 * Xrt * lbl_functions.dcerror(-np.imag(Xrt), np.real(Xrt))
                    SD = 2. * (1. - pxw) / (WIDTH2[:, None] - DELTA2[:, None] * 1j)
                    RES[index1] = (RES + np.real(SD - BASE[:, None]))[index1]
                    RES[index2] = (RES + WIDTH0[:, None] / (DF ** 2 + WSQ[:, None]) - BASE[:, None])[index2]
                else:
                    DF = ds["f"].values + self.FL[:, None] + SHIFT[:, None]
                    index = np.abs(DF) < 750
                    RES[index] = (RES + WIDTH0[:, None] / (DF ** 2 + WSQ[:, None]) - BASE[:, None])[index]

        SUM2 += np.sum(S[:, None] * RES * (ds["f"].values / self.FL[:, None]) ** 2, axis = 0)
        ALPHA = 1.E-10 * ds["absolute_humidity_mean"].values * 1000 * SUM2 / (np.pi * R22.H2OM) + CON
        return ALPHA.values

    def o2_1sd(self, ds: xr.Dataset) -> np.array:
        """RETURN POWER ABSORPTION COEFFICIENT DUE TO OXYGEN IN AIR.
        (8/25/2022) a preliminary alternative with speed-dependence for 118 GHz"""

        # correction factors (rel. to best-fit Voigt) for speed-dependence of 1- line
        R = [1.014, .0768]

        TH = 300. / ds["T_mean"].values
        TH1 = TH - 1.
        B = TH ** self.X
        PRESWV = ds["absolute_humidity_mean"] * 1000 * ds["T_mean"] * R22.RH2O
        PRESDA = (ds["p_mean"] / 100) - PRESWV
        DEN = .001 * (PRESDA * B + 1.2 * PRESWV * TH)
        # DENS = .001 * (PRESDA + 1.2 * PRESWV) * TH
        DFNR = self.WB300 * DEN
        PE2 = DEN * DEN

        # 1.571e-17 (o16-o16) + 1.3e-19 (o16-o18) = 1.584e-17
        SUM = 1.584E-17 * ds["f"] ** 2 * DFNR / (TH * (ds["f"] ** 2 + DFNR * DFNR))
        for k in np.arange(len(self.F)):
            WIDTH = self.W300[k] * DEN
            if k == 0:
                WIDTH2 = WIDTH * R[1]
                WIDTH = WIDTH * R[0]
            Y = DEN * (self.Y300[k] + self.V[k] * TH1)
            STR = self.S300[k] * np.exp(-self.BE[k] * TH1)
            if k == 0:
                # speed-dependent resonant shape factor with no shift
                index = np.abs(ds["f"] - self.F[k]) < (10 * WIDTH)
                Xrt = np.sqrt(((WIDTH - 1.5 * WIDTH2) + (ds["f"] - self.F[k]) * 1j) / WIDTH2)
                pxw = 1.77245385090551603 * Xrt * lbl_functions.dcerror(-np.imag(Xrt), np.real(Xrt))
                A = (1 + Y * 1j) * 2 * (1 - pxw) / WIDTH2
                SF1[index] = np.real(A)[index]
                SF1[~index] = ((WIDTH + (ds["f"] - self.F[k]) * Y) / ((ds["f"] - self.F[k]) ** 2 + WIDTH * WIDTH))[
                    ~index]
            else:
                SF1 = ((WIDTH + (ds["f"] - self.F[k]) * Y) / ((ds["f"] - self.F[k]) ** 2 + WIDTH * WIDTH))
            SF2 = ((WIDTH - (ds["f"] + self.F[k]) * Y) / ((ds["f"] + self.F[k]) ** 2 + WIDTH * WIDTH))
            SUM = SUM + STR * (SF1 + SF2) * (ds["f"] / self.F[k]) ** 2
        O2ABS = 1.6097e11 * SUM * PRESDA * TH ** 3
        O2ABS[O2ABS < 0] = 0
        return O2ABS.values

    def o2(self, ds: xr.Dataset) -> np.array:
        """RETURN POWER ABSORPTION COEFFICIENT DUE TO OXYGEN IN AIR.
        depricated."""
        # WIDTHS IN MHZ/MB
        WB300 = .56
        X = .754

        TH = 300. / ds["T_mean"].values
        TH1 = TH - 1.
        B = TH ** X
        PRESWV = ds["absolute_humidity_mean"] * 1000 * ds["T_mean"] / 216.68
        PRESDA = (ds["p_mean"] / 100) - PRESWV
        DEN = .001 * (PRESDA * B + 1.2 * PRESWV * TH)
        DFNR = WB300 * DEN
        PE2 = DEN * DEN

        # 1.571e-17 (o16-o16) + 1.3e-19 (o16-o18) = 1.584e-17
        SUM = 1.584E-17 * ds["f"] ** 2 * DFNR / (TH * (ds["f"] ** 2 + DFNR * DFNR))
        for k in np.arange(len(self.F)):
            Y = DEN * (self.Y0[k] + self.Y1[k] * TH1)
            DNU = PE2 * (self.DNU0[k] + self.DNU1[k] * TH1)
            GFAC = 1. + PE2 * (self.G0[k] + self.G1[k] * TH1)
            DF = self.W300[k] * DEN
            STR = self.S300[k] * np.exp(-self.BE[k] * TH1)
            DEL1 = ds["f"] - self.F[k] - DNU
            DEL2 = ds["f"] + self.F[k] + DNU
            D1 = DEL1 * DEL1 + DF * DF
            D2 = DEL2 * DEL2 + DF * DF
            SF1 = (DF * GFAC + DEL1 * Y) / D1
            SF2 = (DF * GFAC - DEL2 * Y) / D2
            SUM = SUM + STR * (SF1 + SF2) * (ds["f"] / self.F[k]) ** 2
        O2ABS = 1.6097e11 * SUM * PRESDA * TH ** 3
        O2ABS[O2ABS < 0] = 0
        O2ABS = O2ABS * 1.004  # increase absorption to match Koshelev2017
        return O2ABS.values

    def o2_fast(self, ds: xr.Dataset) -> np.array:
        """RETURN POWER ABSORPTION COEFFICIENT DUE TO OXYGEN IN AIR, fast without loop."""
        # WIDTHS IN MHZ/MB
        WB300 = .56
        X = .754
        TH = 300. / ds["T_mean"].values
        TH1 = TH - 1.
        B = TH ** X
        PRESWV = float(ds["absolute_humidity_mean"] * 1000 * ds["T_mean"] / 216.68)
        PRESDA = float((ds["p_mean"] / 100) - PRESWV)
        DEN = float(.001 * (PRESDA * B + 1.2 * PRESWV * TH))
        DFNR = WB300 * DEN
        PE2 = float(DEN * DEN)

        self.Y0 = np.array(self.Y0)
        self.Y1 = np.array(self.Y1)
        self.DNU0 = np.array(self.DNU0)
        self.DNU1 = np.array(self.DNU1)
        self.G0 = np.array(self.G0)
        self.G1 = np.array(self.G1)
        self.W300 = np.array(self.W300)
        self.S300 = np.array(self.S300)
        self.BE = np.array(self.BE)
        self.F = np.array(self.F)

        f = ds["f"].values

        SUM0 = 1.584E-17 * ds["f"] ** 2 * DFNR / (TH * (ds["f"] ** 2 + DFNR * DFNR))
        Y = DEN * (self.Y0[:, None] + self.Y1[:, None] * TH1)
        DNU = PE2 * (self.DNU0[:, None] + self.DNU1[:, None] * TH1)
        GFAC = 1. + PE2 * (self.G0 + self.G1 * TH1)
        DF = self.W300 * DEN
        STR = self.S300 * np.exp(-self.BE * TH1)
        DEL1 = f - self.F[:, None] - DNU
        DEL2 = f + self.F[:, None] + DNU
        D1 = DEL1 ** 2 + DF[:, None] ** 2
        D2 = DEL2 ** 2 + DF[:, None] ** 2
        SF1 = (DF[:, None] * GFAC[:, None] + DEL1 * Y) / D1
        SF2 = (DF[:, None] * GFAC[:, None] - DEL2 * Y) / D2
        SUM = SUM0 + np.sum(STR[:, None] * (SF1 + SF2) * (f / self.F[:, None]) ** 2, axis=0)

        O2ABS = 1.6097e11 * SUM * PRESDA * TH ** 3
        O2ABS[O2ABS < 0] = 0
        O2ABS = O2ABS * 1.004  # increase absorption to match Koshelev2017
        return O2ABS.values


def wvr20(ds: xr.Dataset) -> np.array:
    """"""
    # ****number of frequencies
    n_f = len(ds["f"])

    if ds["absolute_humidity_mean"] <= 0:
        return np.zeros(n_f)

    # ****LOCAL VARIABLES:
    NLINES = 16
    DF = np.zeros((2, n_f))

    # ****LINE FREQUENCIES:
    FL = np.array([22.235080, 183.310087, 321.225630, 325.152888, 380.197353,
                   439.150807, 443.018343, 448.001085, 470.888999, 474.689092,
                   488.490108, 556.935985, 620.700807, 658.006072, 752.033113, 916.171582])
    # ****LINE INTENSITIES AT 296K:
    S1 = np.array([0.1335e-13, 0.2319e-11, 0.7657e-13, 0.2721e-11, 0.2477e-10, 0.2137e-11,
                   0.4440e-12, 0.2588e-10, 0.8196e-12, 0.3268e-11, 0.6628e-12, 0.1570e-8,
                   0.1700e-10, 0.9033e-12, 0.1035e-8, 0.4275e-10])
    # ****T COEFF. OF INTENSITIES:
    B2 = np.array([2.172, 0.677, 6.262, 1.561, 1.062, 3.643, 5.116, 1.424,
                   3.645, 2.411, 2.890, 0.161, 2.423, 7.921, 0.402, 1.461])
    # ****AIR-BROADENED WIDTH PARAMETERS AT 296K:
    WA = np.array([2.699, 2.945, 2.426, 2.847, 2.868, 2.055, 1.819, 2.612,
                   2.169, 2.366, 2.616, 3.115, 2.468, 3.154, 3.114, 2.695]) * 1.e-3

    # ****T-EXPONENT OF AIR-BROADENING:
    X = np.array([0.76, 0.77, 0.73, 0.64, 0.54, 0.69, 0.70, 0.70, 0.73,
                  0.71, 0.75, 0.75, 0.79, 0.73, 0.77, 0.79])
    # ****SELF-BROADENED WIDTH PARAMETERS AT 296K
    WS = np.array([13.29, 14.78, 10.65, 13.95, 14.40, 9.06, 7.96,
                   13.01, 9.70, 11.24, 13.58, 14.24, 11.94, 13.84, 13.58, 13.55]) * 1.e-3

    # ****T-EXPONENT OF SELF-BROADENING:
    XS = np.array([1.20, 0.78, 0.54, 0.74, 0.89, 0.52, 0.50, 0.67, 0.65, 0.64,
                   0.72,
                   1.0, 0.75, 1.00, 0.84, 0.48])
    SH = np.array([-.033, -.072, -.143, -.013, -.074, 0.051, 0.140, -.116, 0.061,
                   -.027,
                   -.065, 0.187, 0.0000, 0.176, 0.162, 0.0000]) * 1e-3

    Xh = np.array([2.6, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])

    SHS = np.array([0.814, 0.173, 0.278, 1.325, 0.240, 0.165, -.229, -.615,
                    -.465, -.720, -.360, -1.693, 0.687, -1.496, -.878, 0.521]) * 1.e-3

    Xhs = np.array([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.92, 0.0000, 0.0000, 0.47])
    AAIR = np.array([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                     0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
    ASELF = np.array([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                      0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
    Xh[Xh <= 0] = X[Xh <= 0]
    Xhs[Xhs <= 0] = XS[Xhs <= 0]

    # Initial calculations
    PVAP = (ds["absolute_humidity_mean"].values * 1000) * ds["T_mean"].values / 216.68
    PDA = (ds["p_mean"].values / 100) - PVAP
    DEN = 3.344E16 * (ds["absolute_humidity_mean"].values * 1000)

    # ****CONTINUUM TERMS
    REFTCON = 300.
    TI = REFTCON / ds["T_mean"].values
    CF = 5.946E-10
    XCF = 3.
    CS = 1.42E-8
    XCS = 7.5

    #   Xcf and Xcs include 3 for density & stimulated emission
    CON = (CF * PDA * TI ** XCF + CS * PVAP * TI ** XCS) * PVAP * ds["f"] ** 2

    # ****ADD RESONANCES
    REFTLINE = 296.
    TI = REFTLINE / ds["T_mean"].values
    TILN = np.log(TI)
    TI2 = np.exp(2.5 * TILN)

    SUM = 0
    for i in np.arange(NLINES):
        WIDTHF = WA[i] * PDA * TI ** X[i]
        WIDTHS = WS[i] * PVAP * TI ** XS[i]
        WIDTH = WIDTHF + WIDTHS
        WSQ = WIDTH ** 2

        SHIFTF = SH[i] * PDA * (1. - AAIR[i] * TILN) * TI ** Xh[
            i]  # see Smith et al, Spectrochimica Acta v .48 A(9), 1257 - 72(1992)
        SHIFTS = SHS[i] * PVAP * (1. - ASELF[i] * TILN) * TI ** Xhs[i]
        SHIFT = SHIFTF + SHIFTS
        # line intensities include isotopic abundance
        S = S1[i] * TI2 * np.exp(B2[i] * (1. - TI))

        DF[0, :] = ds["f"] - FL[i] - SHIFT
        DF[1, :] = ds["f"] + FL[i] + SHIFT
        # USE CLOUGH'S DEFINITION OF LOCAL LINE CONTRIBUTION
        BASE = WIDTH / (562500. + WSQ)

        # DO FOR POSITIVE AND NEGATIVE RESONANCES
        RES = np.zeros(n_f)
        for j in [0, 1]:
            index = np.abs(DF[j, :]) < 750
            RES[index] = RES[index] + WIDTH / (DF[j, index] ** 2 + WSQ) - BASE
        SUM = SUM + S * RES * (ds["f"] / FL[i]) ** 2
    ALPHA = .3183E-4 * DEN * SUM + CON
    return ALPHA.values


def n2(ds) -> np.array:
    """"""
    TH = 300. / ds["T_mean"].values
    FDEPEN = .5 + .5 / (1. + (ds["f"].values / 450.) ** 2.)
    ALPHA = 9.95e-14 * FDEPEN * (ds["p_mean"].values / 100) ** 2. * ds["f"].values ** 2. * TH ** 3.22
    return ALPHA


def o2r20(ds: xr.Dataset) -> np.array:
    """"""
    F = np.array([118.7503, 56.2648, 62.4863, 58.4466, 60.3061, 59.5910,
                  59.1642, 60.4348, 58.3239, 61.1506, 57.6125, 61.8002,
                  56.9682, 62.4112, 56.3634, 62.9980, 55.7838, 63.5685,
                  55.2214, 64.1278, 54.6712, 64.6789, 54.1300, 65.2241,
                  53.5958, 65.7648, 53.0669, 66.3021, 52.5424, 66.8368,
                  52.0214, 67.3696, 51.5034, 67.9009, 50.9877, 68.4310,
                  50.4742, 68.9603, 233.9461, 368.4982, 401.7398, 424.7630,
                  487.2493, 566.8956, 715.3929, 731.1866,
                  773.8395, 834.1455, 895.0710])

    S300 = np.array([0.2906E-14, 0.7957E-15, 0.2444E-14, 0.2194E-14,
                     0.3301E-14, 0.3243E-14, 0.3664E-14, 0.3834E-14,
                     0.3588E-14, 0.3947E-14, 0.3179E-14, 0.3661E-14,
                     0.2590E-14, 0.3111E-14, 0.1954E-14, 0.2443E-14,
                     0.1373E-14, 0.1784E-14, 0.9013E-15, 0.1217E-14,
                     0.5545E-15, 0.7766E-15, 0.3201E-15, 0.4651E-15,
                     0.1738E-15, 0.2619E-15, 0.8880E-16, 0.1387E-15,
                     0.4272E-16, 0.6923E-16, 0.1939E-16, 0.3255E-16,
                     0.8301E-17, 0.1445E-16, 0.3356E-17, 0.6049E-17,
                     0.1280E-17, 0.2394E-17,
                     0.3287E-16, 0.6463E-15, 0.1334E-16, 0.7049E-14,
                     0.3011E-14, 0.1797E-16, 0.1826E-14, 0.2193E-16,
                     0.1153E-13, 0.3974E-14, 0.2512E-16])
    BE = np.array([.010, .014, .083, .083, .207, .207, .387, .387, .621, .621,
                   .910, .910, 1.255, 1.255, 1.654, 1.654, 2.109, 2.109, 2.618, 2.618,
                   3.182, 3.182, 3.800, 3.800, 4.474, 4.474, 5.201, 5.201, 5.983, 5.983, 6.819, 6.819,
                   7.709, 7.709, 8.653, 8.653, 9.651, 9.651,
                   .019, .048, .045, .044, .049, .084, .145, .136, .141, .145, .201])
    # WIDTHS IN MHZ/MB
    WB300 = .56
    X = .754
    W300 = np.array([1.685, 1.703, 1.513, 1.495, 1.433, 1.408,
                     1.353, 1.353, 1.303, 1.319, 1.262, 1.265,
                     1.238, 1.217, 1.207, 1.207, 1.137, 1.137,
                     1.101, 1.101, 1.037, 1.038, 0.996, 0.996,
                     0.955, 0.955, 0.906, 0.906, 0.858, 0.858,
                     0.811, 0.811, 0.764, 0.764, 0.717, 0.717,
                     0.669, 0.669, 1.65, 1.64, 1.64, 1.64, 1.60, 1.60, 1.60, 1.60, 1.62, 1.47, 1.47])
    Y0 = np.array([-0.041, 0.277, -0.373, 0.560, -0.573, 0.618,
                   -0.366, 0.278, -0.089, -0.021, 0.0599, -0.152,
                   0.216, -0.293, 0.374, -0.436, 0.491, -0.542,
                   0.571, -0.613, 0.636, -0.670, 0.690, -0.718,
                   0.740, -0.763, 0.788, -0.807, 0.834, -0.849,
                   0.876, -0.887, 0.915, -0.922, 0.950, -0.955,
                   0.987, -0.988, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    Y1 = np.array([0., 0.11, -0.009, 0.007, 0.049, -0.1,
                   0.260, -0.346, 0.364, -0.422, 0.315, -0.341,
                   0.483, -0.503, 0.598, -0.610, 0.630, -0.633,
                   0.613, -0.611, 0.570, -0.564, 0.58, -0.57,
                   0.61, -0.60, 0.64, -0.62, 0.65, -0.64,
                   0.66, -0.64, 0.66, -0.64, 0.66, -0.64,
                   0.65, -0.63, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    # 2nd-order mixing coeff. in 1/bar^2
    G0 = np.array([-0.000695, -0.090, -0.103, -0.239, -0.172, -0.171,
                   0.028, 0.150, 0.132, 0.170, 0.087, 0.069,
                   0.083, 0.068, 0.007, 0.016, -0.021, -0.066,
                   -0.095, -0.116, -0.118, -0.140, -0.173, -0.186,
                   -0.217, -0.227, -0.234, -0.242, -0.266, -0.272,
                   -0.301, -0.304, -0.334, -0.333, -0.362, -0.358,
                   -0.348, -0.344, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    G1 = np.array([0., -0.042, 0.004, 0.025, 0.083, 0.167,
                   0.178, 0.223, 0.054, 0.003, 0.002, -0.044,
                   -0.019, -0.054, -0.177, -0.208, -0.294, -0.334,
                   -0.368, -0.386, -0.374, -0.384, -0.387, -0.389,
                   -0.423, -0.422, -0.46, -0.46, -0.51, -0.50,
                   -0.55, -0.53, -0.58, -0.56, -0.62, -0.59,
                   -0.68, -0.65, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    # dnu in GHz/bar^2
    DNU0 = np.array([-0.00028, 0.00596, -0.01950, 0.032, -0.0475, 0.0541,
                     -0.0232, 0.0155, 0.0007, -0.0086, -0.0026, -0.0013,
                     -0.0004, -0.002, 0.005, -0.007, 0.007, -0.008,
                     0.006, -0.007, 0.006, -0.006, 0.005, -0.0049,
                     0.0040, -0.0041, 0.0036, -0.0037, 0.0033, -0.0034,
                     0.0032, -0.0032, 0.0030, -0.0030, 0.0028, -0.0029,
                     0.0029, -0.0029, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    DNU1 = np.array([-0.00037, 0.0086, -0.013, 0.019, -0.026, 0.027,
                     0.005, -0.014, 0.012, -0.018, -0.015, 0.015,
                     0.003, -0.004, 0.012, -0.013, 0.012, -0.012,
                     0.009, -0.009, 0.002, -0.002, 0.0005, -0.0005,
                     0.002, -0.002, 0.002, -0.002, 0.002, -0.002,
                     0.002, -0.002, 0.002, -0.002, 0.001, -0.001,
                     0.0004, -0.0004, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    TH = 300. / ds["T_mean"].values
    TH1 = TH - 1.
    B = TH ** X
    # PRESWV = VAPDEN*TEMP/217.
    PRESWV = ds["absolute_humidity_mean"] * 1000 * ds["T_mean"] / 216.68
    PRESDA = (ds["p_mean"] / 100) - PRESWV
    # DEN = .001*(PRESDA*B + 1.1*PRESWV*TH)
    DEN = .001 * (PRESDA * B + 1.2 * PRESWV * TH)
    DFNR = WB300 * DEN
    PE2 = DEN * DEN

    # 1.571e-17 (o16-o16) + 1.3e-19 (o16-o18) = 1.584e-17
    SUM = 1.584E-17 * ds["f"] ** 2 * DFNR / (TH * (ds["f"] ** 2 + DFNR * DFNR))
    for k in np.arange(len(F)):
        Y = DEN * (Y0[k] + Y1[k] * TH1)
        DNU = PE2 * (DNU0[k] + DNU1[k] * TH1)
        GFAC = 1. + PE2 * (G0[k] + G1[k] * TH1)
        DF = W300[k] * DEN
        STR = S300[k] * np.exp(-BE[k] * TH1)
        DEL1 = ds["f"] - F[k] - DNU
        DEL2 = ds["f"] + F[k] + DNU
        D1 = DEL1 * DEL1 + DF * DF
        D2 = DEL2 * DEL2 + DF * DF
        SF1 = (DF * GFAC + DEL1 * Y) / D1
        SF2 = (DF * GFAC - DEL2 * Y) / D2
        SUM = SUM + STR * (SF1 + SF2) * (ds["f"] / F[k]) ** 2
    O2ABS = 1.6097e11 * SUM * PRESDA * TH ** 3
    O2ABS[O2ABS < 0] = 0
    O2ABS = O2ABS * 1.004  # increase absorption to match Koshelev2017
    return O2ABS.values


####### Cloud Absorption Models

def abliq(ds, ds_mean) -> np.array:
    """Return liquid water absoption coefficient according to Liebe.

    LIEBE, HUFFORD AND MANABE, INT. J. IR & MM WAVES V.12, pp.659-675
    (1991);  Liebe et al, AGARD Conf. Proc. 542, May 1993."""
    if ds["lwc"] <= 0:
        return np.zeros(len(ds["f"]))

    THETA1 = 1. - 300. / ds_mean["T_mean"].values
    EPS0 = 77.66 - 103.3 * THETA1
    EPS1 = .0671 * EPS0
    EPS2 = 3.52  # from MPM93
    FP = 20.1 * np.exp(7.88 * THETA1)  # from eq.2b
    FS = 39.8 * FP
    EPS = (EPS0 - EPS1) / (1. + (ds["f"] / FP) * 1j) + (EPS1 - EPS2) / (1. + (ds["f"].values / FS) * 1j) + EPS2
    RE = (EPS - 1.) / (EPS + 2.)
    ALPHA = -.06286 * np.imag(RE) * ds["f"].values * ds["lwc"]
    return ALPHA.values


def rewat_ellison(ds, ds_mean, verbose=False, salinity=0):
    """Return liquid water absorption coefficient according to ELLISON 2006.

     REFERENCES
     BOOK ARTICLE FROM WILLIAM ELLISON IN MAETZLER 2006 (p.431-455):
     THERMAL MICROWAVE RADIATION:
     APPLICATIONS FOR REMOTE SENSING IET ELECTROMAGNETIC WAVES SERIES 52
     ISBN: 978-086341-573-9"""
    # *** Convert Salinity from parts per thousand to SI
    salinity = salinity * 1e-3
    Temp = ds_mean["T_mean"] - 273.15
    freq = ds["f"] * 1e9
    # *** Check the input ranges:

    # --------------------------------------------------------------------------------------------------------
    # COEFFS AND CALCULATION OF eps(FREQ, Temp, SAL) according to (5.21, p.445)
    # --------------------------------------------------------------------------------------------------------

    # *** Coefficients a_i (Table 5.5 or p. 454):

    a_1 = 0.46606917e-2
    a_2 = -0.26087876e-4
    a_3 = -0.63926782e-5
    a_4 = 0.63000075e1
    a_5 = 0.26242021e-2
    a_6 = -0.42984155e-2
    a_7 = 0.34414691e-4
    a_8 = 0.17667420e-3
    a_9 = -0.20491560e-6
    a_10 = 0.58366888e3
    a_11 = 0.12634992e3
    a_12 = 0.69227972e-4
    a_13 = 0.38957681e-6
    a_14 = 0.30742330e3
    a_15 = 0.12634992e3
    a_16 = 0.37245044e1
    a_17 = 0.92609781e-2
    a_18 = -0.26093754e-1

    # *** Calculate parameter functions (5.24)-(5.28), p.447

    EPS_S = 87.85306 * np.exp(-0.00456992 * Temp - a_1 * salinity - a_2 * salinity ** 2. - a_3 * salinity * Temp)
    EPS_1 = a_4 * np.exp(-a_5 * Temp - a_6 * salinity - a_7 * salinity * Temp)
    tau_1 = (a_8 + a_9 * salinity) * np.exp(a_10 / (Temp + a_11)) * 1e-9
    tau_2 = (a_12 + a_13 * salinity) * np.exp(a_14 / (Temp + a_15)) * 1e-9
    EPS_INF = a_16 + a_17 * Temp + a_18 * salinity

    # *** Calculate seawater conductivity (5.20), p.437

    if salinity > 0:
        c_alpha_0 = (6.9431 + 3.2841 * salinity - 0.099486 * salinity ** 2.)
        d_alpha_0 = (84.85 + 69.024 * salinity + salinity ** 2.)
        alpha_0 = c_alpha_0 / d_alpha_0
        alpha_1 = 49.843 - 0.2276 * salinity + 0.00198 * salinity ** 2.
        Q = 1. + alpha_0 * (Temp - 15) / (Temp + alpha_1)
        c_P = (37.5109 + 5.45216 * salinity + 0.014409 * salinity ** 2.)
        d_P = (1004.75 + 182.283 * salinity + salinity ** 2.)
        P = salinity * c_P / d_P
        sigma_35 = 2.903602 + 8.607e-2 * Temp + 4.738817e-4 * Temp ** 2. - \
                   2.991e-6 * Temp ** 3. + 4.3041e-9 * Temp ** 4.

        SIGMA = sigma_35 * P * Q
    else:
        SIGMA = 0

    # *** Finally apply the interpolation formula (5.21)

    first_term = (EPS_S - EPS_1) / (1. + (-2. * np.pi * freq * tau_1) * 1j)
    second_term = (EPS_1 - EPS_INF) / (1. + (-2. * np.pi * freq * tau_2) * 1j)
    third_term = (EPS_INF + (17.9751 * SIGMA / freq) * 1j)
    # third_term = EPS_INF
    EPS = first_term + second_term + third_term

    # *** compute absorption coefficients
    RE = (EPS - 1) / (EPS + 2)
    MASS_ABSCOF = 6. * np.pi * np.imag(RE) * freq * 1e-3 / c
    VOL_ABSCOF = MASS_ABSCOF * ds["lwc"]
    if verbose:
        print(VOL_ABSCOF)

    # *** Convert to refractive index

    N = np.sqrt(EPS)
    N_R = np.real(N)
    N_I = np.imag(N)

    if verbose:
        print(N_R, N_I)
    return VOL_ABSCOF


def refwat_ellison07(ds, ds_mean, verbose=False, ndebyeterms=3):
    """Return liquid water absorption coefficient according to ELLISON 2007.

     REFERENCES
     Ellison, W. J., J.Phys.Chem.Ref.Data, Vol. 36, No. 1, 2007
     """
    # T in degr. Celsius and f Hz)
    Temp = ds_mean["T_mean"] - 273.15
    freq = ds["f"] * 1e9
    # *** Check the input ranges:
    if verbose:
        if np.max(freq) > 25e12:
            print("WARNING (lbl_rt_models.refwat_ellison07(): Frequency range: 0-25 THz, extrapolating.")
        if np.min(Temp) < 0:
            print("WARNING (lbl_rt_models.refwat_ellison07(): Temperature (%d) outside of range: 0-100 degC" % np.min(
                Temp))
        elif np.max(Temp) > 100:
            print("WARNING (lbl_rt_models.refwat_ellison07(): Temperature (%d) outside of range: 0-100 degC" % np.max(
                Temp))

    # *** Coefficients for the single relaxation approximation (section 7.1, p.9)

    if ndebyeterms == 1:
        a_1 = 80.69715
        b_1 = 0.004415996
        c_1 = 1.367283e-13
        d_1 = 651.4728
        t_c = 133.0699

    # *** Coefficients for the double relaxation approximation (section 7.2, p.9)

    elif ndebyeterms == 2:
        a_1 = 79.42385
        a_2 = 3.611638
        b_1 = 0.004319728
        b_2 = 0.01231281
        c_1 = 1.352835e-13
        c_2 = 1.005472e-14
        d_1 = 653.3092
        d_2 = 743.0733
        t_c = 132.6248

    # *** Coefficients for the three relaxation and two resonance frequencies (Table 2)

    elif ndebyeterms == 3:
        t_c = 133.1383

        a_1 = 79.23882
        a_2 = 3.815866
        a_3 = 1.634967

        b_1 = 0.004300598
        b_2 = 0.01117295
        b_3 = 0.006841548

        c_1 = 1.382264e-13
        c_2 = 3.510354e-16
        c_3 = 6.30035e-15

        d_1 = 652.7648
        d_2 = 1249.533
        d_3 = 405.5169

        p_0 = 0.8379692
        p_1 = -0.006118594
        p_2 = -0.000012936798
        p_3 = 4235901.0e6
        p_4 = -14260880.0e3
        p_5 = 273815700.0
        p_6 = -1246943.0
        p_7 = 9.618642e-14
        p_8 = 1.795786e-16
        p_9 = -9.310017e-18
        p_10 = 1.655473e-19
        p_11 = 0.6165332
        p_12 = 0.007238532
        p_13 = -0.00009523366
        p_14 = 15983170.0e6
        p_15 = -74413570.0e3
        p_16 = 497448.0e3
        p_17 = 2.882476e-14
        p_18 = -3.142118e-16
        p_19 = 3.528051e-18

    # --------------------------------------------------------------------------------------------------------
    # APPLY CORRESPONDING(1 / 2 / 3 Debye formulas) INTERPOLATION FORMULAS(p .17)
    # --------------------------------------------------------------------------------------------------------

    # *** Static dielectric permettivity (for all the same)

    eps_s = 87.9144 - 0.404399 * Temp + 9.58726e-4 * Temp ** 2. - 1.32802e-6 * Temp ** 3.

    delta_1 = a_1 * np.exp(-b_1 * Temp)
    tau_1 = c_1 * np.exp(d_1 / (Temp + t_c))

    # *** add succesively additional delta/tau/freqs for the various interp. formulas

    if ndebyeterms >= 2:
        delta_2 = a_2 * np.exp(-b_2 * Temp)
        tau_2 = c_2 * np.exp(d_2 / (Temp + t_c))
    if ndebyeterms == 3:
        delta_3 = a_3 * np.exp(-b_3 * Temp)
        delta_4 = p_0 + p_1 * Temp + p_2 * Temp ** 2.
        delta_5 = p_11 + p_12 * Temp + p_13 * Temp ** 2.

        tau_3 = c_3 * np.exp(d_3 / (Temp + t_c))
        tau_4 = p_7 + p_8 * Temp + p_9 * Temp ** 2. + p_10 * Temp ** 3.
        tau_5 = p_17 + p_18 * Temp + p_19 * Temp ** 2.

        f_0 = p_3 + p_4 * Temp + p_5 * Temp ** 2. + p_6 * Temp ** 3.
        f_1 = p_14 + p_15 * Temp + p_16 * Temp ** 2.
    else:
        delta_4 = 0
        delta_5 = 0
        tau_4 = 0
        tau_5 = 0

    # *** Finally apply the interpolation formula (17a/b)
    # EPS_prime (real part, (17a)):
    # --- Relaxation terms
    term1_p1 = (tau_1 ** 2. * delta_1) / (1 + (2 * np.pi * freq * tau_1) ** 2.)
    if ndebyeterms >= 2:
        term2_p1 = (tau_2 ** 2. * delta_2) / (1 + (2 * np.pi * freq * tau_2) ** 2.)
    else:
        term2_p1 = np.zeros_like(freq)
    if ndebyeterms == 3:
        term3_p1 = (tau_3 ** 2. * delta_3) / (1 + (2 * np.pi * freq * tau_3) ** 2.)
    else:
        term3_p1 = np.zeros_like(freq)
    # --- First resonance term
    if ndebyeterms == 3:
        term1_p2 = (freq * (f_0 + freq)) / (1 + (2 * np.pi * tau_4 * (f_0 + freq)) ** 2.)
        term2_p2 = (freq * (f_0 - freq)) / (1 + (2 * np.pi * tau_4 * (f_0 - freq)) ** 2.)
    else:
        term1_p2 = np.zeros_like(freq)
        term2_p2 = np.zeros_like(freq)
    # --- Second resonance term
    if ndebyeterms == 3:
        term1_p3 = (freq * (f_1 + freq)) / (1 + (2 * np.pi * tau_5 * (f_1 + freq)) ** 2.)
        term2_p3 = (freq * (f_1 - freq)) / (1 + (2 * np.pi * tau_5 * (f_1 - freq)) ** 2.)
    else:
        term1_p3 = np.zeros_like(freq)
        term2_p3 = np.zeros_like(freq)
    EPS_prime = eps_s - ((2 * np.pi * freq) ** 2.) * (term1_p1 + term2_p1 + term3_p1) - \
                ((2 * np.pi * tau_4) ** 2.) * (delta_4 / 2) * (term1_p2 - term2_p2) - \
                ((2 * np.pi * tau_5) ** 2.) * (delta_5 / 2) * (term1_p3 - term2_p3)

    # EPS_primeprime (imag. part, (17b)):
    # --- Relaxation terms
    term1_p1 = (tau_1 * delta_1) / (1 + (2 * np.pi * freq * tau_1) ** 2.)
    if ndebyeterms >= 2:
        term2_p1 = (tau_2 * delta_2) / (1 + (2 * np.pi * freq * tau_2) ** 2.)
    else:
        term2_p1 = np.zeros_like(freq)
    if ndebyeterms == 3:
        term3_p1 = (tau_3 * delta_3) / (1 + (2 * np.pi * freq * tau_3) ** 2.)
    else:
        term3_p1 = np.zeros_like(freq)
    # --- First resonance term
    if ndebyeterms == 3:
        term1_p2 = 1 / (1 + (2 * np.pi * tau_4 * (f_0 + freq)) ** 2.)
        term2_p2 = 1 / (1 + (2 * np.pi * tau_4 * (f_0 - freq)) ** 2.)
    else:
        term1_p2 = np.zeros_like(freq)
        term2_p2 = np.zeros_like(freq)
    # --- Second resonance term
    if ndebyeterms == 3:
        term1_p3 = 1 / (1 + (2 * np.pi * tau_5 * (f_1 + freq)) ** 2.)
        term2_p3 = 1 / (1 + (2 * np.pi * tau_5 * (f_1 - freq)) ** 2.)
    else:
        term1_p3 = np.zeros_like(freq)
        term2_p3 = np.zeros_like(freq)
    EPS_primeprime = 2 * np.pi * freq * (term1_p1 + term2_p1 + term3_p1) + \
                     np.pi * freq * tau_4 * delta_4 * (term1_p2 + term2_p2) + \
                     np.pi * freq * tau_5 * delta_5 * (term1_p3 + term2_p3)

    EPS = EPS_prime + EPS_primeprime * 1j

    # calculate refractivity and mass/volume absorption coefficient
    # frequency in Hz, lwc in g/m³
    RE = (EPS - 1) / (EPS + 2)
    MASS_ABSCOF = 6. * np.pi * np.imag(RE) * freq * 1e-3 / c
    VOL_ABSCOF = MASS_ABSCOF * ds["lwc"]
    return VOL_ABSCOF
