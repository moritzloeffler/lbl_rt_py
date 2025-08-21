# -*- coding: utf-8 -*-
# --------------------------------
# Name:       lbl_functions.py
# Purpose:    LBL RT model helper functions.
# Author:     Moritz Loeffler
# Created:    2022-11-24
# Python Version:   3.6
# Version:    1.0
# Last Edit:  2025-08-20
# --------------------------------
"""
Helper functions required for line by line RT modelling translated from IDL code (Ulrich Loehnert et al).

"""

#########################
# Modules

import numpy as np
import xarray as xr
import os
import sys
import warnings
import datetime as dt
from typing import List, Optional, Tuple, Union

try:
    from bottleneck import nanmean, nansum, nanmax
except ModuleNotFoundError:
    from numpy import nanmean, nansum, nanmax

sys.path.append(os.path.dirname(__file__))
import basic_functions

# paths
file_standard_atmospheres = os.path.dirname(__file__) + "/data/standard_atmospheres.nc"
coeff_dir = os.path.dirname(__file__) + "/coeff"

# global variables
Rl = 287.  # indiv. gas constant dry air
Rv = 462.  # indiv. gas constant water
eps = Rl/Rv

def dewpoint(T, rh):
    """Input: temperature(K) and rel.humidity( %)
    Output: dewpoint"""
    Rv = 462.
    L = lvap(T)
    T0 = 273.15
    fak = L / (Rv * T0)
    fak = 1. / fak
    e0 = 610.78

    f = rh / 100.
    es = esat(T)
    e = f * es

    Td = T0 / (1. - fak * np.log(e / e0))
    return Td


def abhum(T, rh):
    """Return absolute humidity [kg/m**3] input temperature [ K ] and rel. humidity [ 1 ]."""
    # constants
    MW = 18.016  # molecular weight [kg/kmol]
    R = 8314.3  # gas constant [J/kmol K]
    # Calculate saturation vapor pressure
    es = estet(T)  # ESTET: water vapor pressure at saturation
    # Calculate absolute humidity
    x = (MW * es * rh) / (R * T)
    return x

def estet(T):
    """Compute the saturation vapor pressure above a plain water surface in Pa"""
    x = 611.0 * 10.0 ** (7.5 * (T - 273.15) / (T - 35.85))
    return x


def abs_hum_to_mixr(ds: xr.Dataset) -> xr.Dataset:
    """Convert abs hum to mixing ratio and return ds."""
    e = ds["absolute_humidity"] * Rv * ds["T"]
    #es = esat(T)
    m = eps * e / (ds["p"] - e)
    ds["wvmr"] = ds["absolute_humidity"].copy()
    ds["wvmr"] = m
    return ds


def specific_hum_to_abs_hum(q: np.array, T: np.array, p: np.array) -> np.array:
    """Convert specific humidity to absolute humidity and return as array.
    p in [Pa], T in [K], q in [kg/kg]"""
    absolute_humidity = q*p/(Rv*T)/(eps+(1-eps)*q)
    return absolute_humidity


def w2e(ds: xr.Dataset) -> xr.Dataset:
    """Convert water vapor mixing ratio to the partial pressure and return in ds."""
    # wvmr[kg/kg], p[Pa]
    ww = ds["wvmr"]
    ds["e"] = ds["wvmr"].copy()
    ds["e"] = ds["p"]/100 * ww / (eps + ww)  # e in [hPa;mbar]
    return ds

def extend_atm_profile(ds: xr.Dataset, levelVar: str) -> xr.Dataset:
    """Extend a given thermodynamic profile into the stratosphere."""
    # load standard atmosphere
    fname = file_standard_atmospheres
    ds_statm = xr.open_dataset(fname)
    # choose summer/winter:
    month = ds["time"].values.astype("datetime64[M]").astype(dt.datetime).month
    if (month >= 5) & (month <= 10):  # summer
        atm_id = 0
    else:  # winter
        atm_id = 1
    ds_statm = ds_statm.loc[{"standard_atmospheres": atm_id}]
    if levelVar == "p":
        return extend_atm_profile_p(ds, ds_statm)
    elif levelVar == "height":
        return extend_atm_profile_height(ds, ds_statm)
    elif levelVar == "level":
        return extend_atm_profile_level(ds, ds_statm)
    else:
        raise KeyError("Currently one of 'level', 'p' or 'height' are required coordinates.")

def extend_atm_profile_p(ds: xr.Dataset, ds_statm: xr.Dataset) -> xr.Dataset:
    """Extend a given thermodynamic profile into the stratosphere."""
    # check if data reaches too high
    altitude_limit = 30100  # m
    if np.nanmax(ds["height"].values) > altitude_limit:
        return ds.loc[{"p": ds["p"].values[ds["height"].values < altitude_limit]}]
    # make matching dataset
    vars_in = ["height", "t_atmo", "a_atmo"]
    vars_out = ["height", "T", "absolute_humidity"]
    index = (ds_statm["p_atmo"] * 100 < np.min(ds["p"]) * 0.7) & (ds_statm["height"] * 1000 < 30100)
    ds_upper = xr.Dataset()
    for var_in, var_out in zip(vars_in, vars_out):
        ds_upper[var_out] = xr.DataArray(ds_statm[var_in].values[index],
                                         dims=["p"],
                                         coords={"p": ds_statm["p_atmo"].values[index] * 100})
    vars_extra = ["lwc", "crwc"]
    for var in vars_extra:
        ds_upper[var] = xr.DataArray(np.zeros_like(ds_upper["p"]),
                                     dims=["p"],
                                     coords={"p": ds_statm["p_atmo"].values[index] * 100})
    ds_upper["height"] = ds_upper["height"] * 1000
    ds_out = xr.concat([ds, ds_upper], dim = "p", data_vars = "minimal")
    # check that p is monotonous
    dp = ds_out["p"].values[1:] - ds_out["p"].values[:-1]
    if np.any(dp >= 0):
        ds_out = ds_out.reindex({"p": ds_out["p"].values[np.append([True], [dp < 0])]})
    return ds_out


def extend_atm_profile_height(ds: xr.Dataset, ds_statm: xr.Dataset) -> xr.Dataset:
    """Extend a given thermodynamic profile into the stratosphere."""
    # check if data reaches too high
    altitude_limit = 30100  # m
    if np.nanmax(ds["height"].values) > altitude_limit:
        return ds.loc[{"height": ds["height"].values[ds["height"].values < altitude_limit]}]
    # make matching dataset
    vars_in = ["p_atmo", "t_atmo", "a_atmo"]
    vars_out = ["p", "T", "absolute_humidity"]
    index = (ds_statm["p_atmo"] * 100 < np.min(ds["p"]) * 0.7) & (ds_statm["height"] * 1000 < 30100)
    ds_upper = xr.Dataset()
    for var_in, var_out in zip(vars_in, vars_out):
        ds_upper[var_out] = xr.DataArray(ds_statm[var_in].values[index],
                                         dims=["height"],
                                         coords={"height": ds_statm["height"].values[index] * 1000})
    vars_extra = ["lwc", "crwc"]
    for var in vars_extra:
        ds_upper[var] = xr.DataArray(np.zeros_like(ds_upper["height"]),
                                     dims=["height"],
                                     coords={"height": ds_statm["height"].values[index] * 1000})
    ds_upper["p"] = ds_upper["p"] * 100
    ds_out = xr.concat([ds, ds_upper], dim = "height", data_vars = "minimal")
    # check that p is monotonous
    dp = ds_out["p"].values[1:] - ds_out["p"].values[:-1]
    if np.any(dp >= 0):
        ds_out = ds_out.reindex({"height": ds_out["height"].values[np.append([True], [dp < 0])]})
    return ds_out


def extend_atm_profile_level(ds: xr.Dataset, ds_statm: xr.Dataset) -> xr.Dataset:
    """Extend a given thermodynamic profile into the stratosphere."""
    # check if data reaches too high
    altitude_limit = 30100  # m
    if np.nanmax(ds["height"].values)>altitude_limit:
        return ds.loc[{"level": ds["level"].values[ds["height"].values < altitude_limit]}]
    # make matching dataset
    vars_in = ["p_atmo", "height", "t_atmo", "a_atmo"]
    vars_out = ["p", "height", "T", "absolute_humidity"]
    index = (ds_statm["p_atmo"] * 100 < np.min(ds["p"]) * 0.7) & (ds_statm["height"] * 1000 < altitude_limit)
    ds_upper = xr.Dataset()
    for var_in, var_out in zip(vars_in, vars_out):
        ds_upper[var_out] = xr.DataArray(ds_statm[var_in].values[index],
                                         dims=["level"],
                                         coords={"level": np.arange(np.sum(index) + np.max(ds["level"].values) + 1)})
    vars_extra = ["lwc", "crwc"]
    for var in vars_extra:
        ds_upper[var] = xr.DataArray(np.zeros_like(ds_upper["p"]),
                                     dims=["level"],
                                     coords={"level": np.arange(np.sum(index) + np.max(ds["level"].values) + 1)})
    ds_upper["height"] = ds_upper["height"] * 1000
    ds_upper["p"] = ds_upper["p"] * 100
    ds_out = xr.concat([ds, ds_upper], dim = "level", data_vars = "minimal")
    # check that p is monotonous
    dp = ds_out["p"].values[1:] - ds_out["p"].values[:-1]
    if np.any(dp >= 0):
        ds_out = ds_out.reindex({"level": ds_out["level"].values[np.append([True], [dp < 0])]})
    return ds_out


def addGroundLevel(ds: xr.Dataset, station_height: Optional[float]) -> xr.Dataset:
    """Extend or cut-off profile so that lowest level matches ground height."""
    degree = 4
    if station_height is None:
        return ds
    # determine index of height relevant for indexing
    heights = ds["height"].values
    h_buffer = np.max([100, heights[0]-station_height])  # m
    fit_index = degree + 1
    fit_index = np.max([fit_index, np.searchsorted(heights, station_height + h_buffer, side="left")])
    # reindex dataset
    new_height = False
    if not any(station_height == heights):
        new_height = True
        heights = np.sort(np.append(heights, station_height))
    if "height" in list(ds.coords):
        ds_out = ds.reindex({"height": heights}).copy()
        vars = ["p", "absolute_humidity", "T", "lwc"]
        coord = "height"
        coord0 = station_height
    elif "p" in list(ds.coords):
        # determine p0
        if new_height:
            fit_params = np.polyfit(ds["height"].values[:fit_index], ds["p"].values[:fit_index], deg=degree)
            p0 = 0
            for n in np.arange(degree):
                p0 += fit_params[-n-1] * station_height ** n
            # add p0 into coords
            ps = ds["p"].values
            ps = np.sort(np.append(ps, p0))[::-1]
            ds_out = ds.reindex({"p": ps}).copy()
            vars = ["absolute_humidity", "T", "lwc"]
            ds_out["height"].loc[{"p": p0}] = station_height
        coord = "p"
        coord0 = p0
    elif "level" in list(ds.coords):
        if new_height:
            if station_height < ds["height"].values[0]:
                level0 = 0

            else:
                levels = ds["level"].values
                level0 = np.max(levels[ds["height"].values < station_height])
                levelsWithGap = np.append(levels[levels <= level0] - 1,  levels[levels > level0])
                ds["level"] = levelsWithGap
            ds_out = ds.reindex({"level": np.arange(len(ds["level"].values) + 1)}).copy()
            # add p0 into coords
            vars = ["absolute_humidity", "T", "lwc", "p"]
            ds_out["height"].loc[{"level": level0}] = station_height
        coord = "level"
        coord0 = level0
    else:
        raise KeyError("Currently one of 'level', 'p' or 'height' are required coordinates.")
    # interpolate or extrapolate profile
    if new_height:
        for var in vars:
            fit_params = np.polyfit(ds["height"].values[:fit_index], ds[var].values[:fit_index], deg = degree)
            v0 = 0
            for n in np.arange(degree):
                v0 += fit_params[-n-1] * station_height**n
            ds_out[var].loc[{coord: coord0}] = v0

    if station_height > ds["height"].values[0]:
        coords_out = ds_out[coord].values[ds_out["height"].values >= station_height]
        ds_out = ds_out.reindex({coord: coords_out})

    return ds_out


def dcerror(x, y):
    """SIXTH-ORDER APPROX TO THE COMPLEX ERROR FUNCTION OF z=X+iY."""
    a = [122.607931777104326, 214.382388694706425, 181.928533092181549, 93.155580458138441,
         30.180142196210589, 5.912626209773153, 0.564189583562615]
    b = [122.607931773875350, 352.730625110963558, 457.334478783897737, 348.703917719495792,
         170.354001821091472, 53.992906912940207, 10.479857114260399]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        ZH = (np.abs(y) -x * 1j)
        ASUM = (((((a[6] * ZH + a[5]) * ZH + a[4]) * ZH + a[3]) * ZH + a[2]) * ZH + a[1]) * ZH + a[0]
        BSUM = ((((((ZH + b[6]) * ZH + b[5]) * ZH + b[4]) * ZH + b[3]) * ZH + b[2]) * ZH + b[1]) * ZH + b[0]
        w = ASUM / BSUM
        w2 = 2. * np.exp(-(x + y * 1j) ** 2) - np.conj(w)
        DCERROR = w
        DCERROR[y < 0] = w2[y < 0]
        return DCERROR


