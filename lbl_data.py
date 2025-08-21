# -*- coding: utf-8 -*-
# --------------------------------
# Name:       lbl_data.py
# Purpose:    load data for LBL radiative transfer modelling.
# Author:     Moritz Loeffler
# Created:    2025-07-16
# Python Version:   3.6
# Version:    1.0
# Last Edit:  2025-07-16
# --------------------------------
"""

"""

#########################
# Imports

import numpy as np
import xarray as xr
import sys
import os
import datetime as dt
import multiprocessing
import glob
from typing import List, Optional, Tuple, Union

try:
    from bottleneck import nanmean, nansum, nanmax
except ModuleNotFoundError:
    from numpy import nanmean, nansum, nanmax

sys.path.append(os.path.dirname(__file__))
import lbl_calc
import lbl_functions
import basic_functions as functions

# static parameters
coeff_dir = os.path.dirname(__file__) + "/coeff"
g = 9.81  # m/s2


def load_era5(timePeriod: List[np.datetime64], dir_in: Optional[str] = None, files_in: Optional[str] = None,
              file_list: Optional[str] = None, clear_sky_only: bool = False
              ) -> xr.Dataset:
    """Load ERA5 (pressure levels) dataset and prepare for RT modelling."""
    # load files in time period
    files = functions.makeFileList(dir_in, files_in, file_list)
    files.sort()
    # configure progressBar
    j = 0
    j_max = len(files)
    startTime = dt.datetime.utcnow()
    for file in files:
        if (j % 100) == 0:
            functions.progressBar2(j, j_max, startTime, message="Loading files: ")
        ds_i = xr.open_dataset(file)
        try:
            ds_in = xr.concat([ds_in, ds_i], dim="time")
        except NameError:
            ds_in = ds_i
    print("")
    ds_in = ds_in.loc[{"time": slice(timePeriod[0], timePeriod[1])}]
    # convert format
    # desired units of absolute_humidity: abs. hum. [kg m^-3] (q,clwc: kg/kg) and p: Pa (level: mbar), T in K, height in m
    ds_in = ds_in.rename({"z": "height", "t": "T", "q": "absolute_humidity", "clwc": "lwc", "level": "p"})
    ds_in = ds_in.reindex(p=ds_in.p[::-1])
    ds_in = ds_in.isel({"latitude": 0, "longitude": 0})  # reduce data dim. Should only contain one lat/lon pair.
    ds_in["height"] = ds_in["height"] / g
    ds_in["p"] = ds_in["p"] * 100
    for p in ds_in["p"].values:
        ds_in["absolute_humidity"].loc[{"p": p}] = lbl_functions.specific_hum_to_abs_hum(
            q=ds_in["absolute_humidity"].loc[{"p": p}].values,
            T=ds_in["T"].loc[{"p": p}].values,
            p=ds_in["p"].loc[{"p": p}].values)
        ds_in["lwc"].loc[{"p": p}] = lbl_functions.specific_hum_to_abs_hum(q=ds_in["lwc"].loc[{"p": p}].values,
                                                                           T=ds_in["T"].loc[{"p": p}].values,
                                                                           p=ds_in["p"].loc[{"p": p}].values)
    if clear_sky_only:
        ds_in["lwc"][:] = 0
    return ds_in


def load_era5ml(timePeriod: List[np.datetime64],
                dir_in: Optional[str] = None, files_in: Optional[str] = None, file_list: Optional[str] = None,
                clear_sky_only: bool = False
                ) -> xr.Dataset:
    """Load ERA5 (model levels) dataset and prepare for RT modelling."""
    # load files in time period
    files = functions.makeFileList(dir_in, files_in, file_list)
    files.sort()
    # configure progressBar
    j = 0
    j_max = len(files)
    startTime = dt.datetime.utcnow()
    for file in files:
        if (j % 100) == 0:
            functions.progressBar2(j, j_max, startTime, message="Loading files: ")
        ds_i = xr.open_dataset(file)
        try:
            ds_in = xr.concat([ds_in, ds_i], dim="time")
        except NameError:
            ds_in = ds_i
    print("")
    ds_in = ds_in.loc[{"time": slice(timePeriod[0], timePeriod[1])}]
    # prepare data, compute z and p
    ds_in = ds_in.isel({"latitude": 0, "longitude": 0})  # reduce data dim. Should only contain one lat/lon pair.
    _, index = np.unique(ds_in['time'], return_index=True)  # remove double times
    ds_in = era5_ml_z_p_mp(ds_in.isel({"time": index}))
    ds_in = ds_in.reindex({"time": np.sort(ds_in["time"])})
    # convert format
    # check desired units of absolute_humidity: abs. hum. [kg m^-3] (q,clwc: kg/kg) and p: Pa (p: Pa), T in K, height in m
    ds_in = ds_in.rename({"z": "height", "t": "T", "q": "absolute_humidity", "clwc": "lwc"})
    ds_in["height"] = ds_in["height"] / g
    ds_in["p"] = ds_in["p"]
    for level in ds_in["level"].values:
        ds_in["absolute_humidity"].loc[{"level": level}] = lbl_functions.specific_hum_to_abs_hum(
            q=ds_in["absolute_humidity"].loc[{"level": level}].values,
            T=ds_in["T"].loc[{"level": level}].values,
            p=ds_in["p"].loc[{"level": level}].values)
        ds_in["lwc"].loc[{"level": level}] = lbl_functions.specific_hum_to_abs_hum(
            q=ds_in["lwc"].loc[{"level": level}].values,
            T=ds_in["T"].loc[{"level": level}].values,
            p=ds_in["p"].loc[{"level": level}].values)
    if clear_sky_only:
        ds_in["lwc"][:] = 0
    return ds_in


def load_icond2(timePeriod: List[np.datetime64], hhl_file: str, dir_in: Optional[str] = None,
                files_in: Optional[str] = None, file_list: Optional[str] = None, clear_sky_only: bool = False
                ) -> xr.Dataset:
    """Load Icon-D2 dataset and prepare for RT modelling."""
    # load files in time period
    files = functions.makeFileList(dir_in, files_in, file_list)
    fname_hhl = glob.glob(hhl_file.format(**{"path": os.path.dirname(__file__)}))
    ds_hhl = xr.open_dataset(fname_hhl[0])
    heights = (ds_hhl["HHL"].values[0, 1:, 0] + ds_hhl["HHL"].values[0, :-1, 0]) / 2
    files.sort()
    # configure progressBar
    j = 0
    j_max = len(files)
    startTime = dt.datetime.utcnow()
    for file in files:
        if (j % 100) == 0:
            functions.progressBar2(j, j_max, startTime, message="Loading files: ")
        j += 1
        try:
            ds_i = xr.open_dataset(file)
        except:
            continue
        try:
            ds_i = ds_i.drop_vars(["W"])
        except ValueError:
            pass
        try:
            ds_i = ds_i.rename({"QV": "absolute_humidity", "QC": "lwc", "P": "p"})
        except ValueError:
            continue
        try:
            # switch of heights on dec 14 2023 leads to mismatch in concat
            if np.any(~(ds_in["height"].values - ds_i["height"].values == 0)):
                ds_i["height"] = ds_in["height"]
            ds_in = xr.concat([ds_in, ds_i], dim="time")
        except NameError:
            ds_in = ds_i
    print("")
    ds_in = ds_in.loc[{"time": slice(timePeriod[0], timePeriod[1])}]
    # convert format
    # check desired units of absolute_humidity: abs. hum. [kg m^-3] (q,clwc: kg/kg) and p: Pa (level: mbar), T in K, height in m
    if clear_sky_only:
        ds_in["lwc"][:] = 0
    ds_in["height"] = heights
    ds_in = ds_in.reindex(height=ds_in.height[::-1])
    ds_in = ds_in.isel({"ncells": 0})  # reduce data dim. Should only contain one lat/lon pair.
    for h in ds_in["height"].values:
        ds_in["absolute_humidity"].loc[{"height": h}] = lbl_functions.specific_hum_to_abs_hum(
            q=ds_in["absolute_humidity"].loc[{"height": h}].values,
            T=ds_in["T"].loc[{"height": h}].values,
            p=ds_in["p"].loc[{"height": h}].values)
        ds_in["lwc"].loc[{"height": h}] = lbl_functions.specific_hum_to_abs_hum(
            q=ds_in["lwc"].loc[{"height": h}].values,
            T=ds_in["T"].loc[{"height": h}].values,
            p=ds_in["p"].loc[{"height": h}].values)
    keys = list(ds_in.keys())
    keep_keys = ["height", "T", "absolute_humidity", "lwc", "p"]
    drop_keys = [i for i in keys if not (i in keep_keys)]
    ds_in = ds_in.drop_vars(drop_keys)
    return ds_in


def load_standard_atm():
    """Load standard atmospheres dataset and prepare for RT modelling."""
    # initialize xarray
    file_standard_atmospheres = os.path.dirname(__file__) + "/data/standard_atmospheres.nc"
    ds_statm = xr.open_dataset(file_standard_atmospheres)
    ds_statm = ds_statm.rename({"standard_atmospheres": "time"})

    # make matching dataset
    vars_in = ["p_atmo", "t_atmo", "a_atmo"]
    vars_out = ["p", "T", "absolute_humidity"]
    ds_upper = xr.Dataset()
    times = np.arange(np.datetime64("2022-01-01"),
                      np.datetime64("2022-01-06T01:00"),
                      np.timedelta64(1, "D")).astype('datetime64[ns]')
    for var_in, var_out in zip(vars_in, vars_out):
        ds_upper[var_out] = xr.DataArray(ds_statm[var_in].values,
                                         dims=["height", "time"],
                                         coords={"height": ds_statm["height"].values * 1000,
                                                 "time": times})
    vars_extra = ["lwc", "crwc"]
    for var in vars_extra:
        ds_upper[var] = xr.DataArray(np.zeros_like(ds_upper["p"]),
                                     dims=["height", "time"],
                                     coords={"height": ds_statm["height"].values * 1000,
                                             "time": times})
    ds_upper["p"] = ds_upper["p"] * 100
    return ds_upper


def era5_ml_z_p_mp(ds: xr.Dataset) -> xr.Dataset:
    """Multiprocessing call of era5_ml_z_p"""
    # configure variables for multiprocessing
    times = ds["time"].values
    nProcessor = multiprocessing.cpu_count()
    if len(times) % nProcessor != 0:
        times = np.append(times, [np.datetime64("NaT")] * (nProcessor - len(times) % nProcessor))
    timesArray = np.reshape(times, [int(len(times) / nProcessor), nProcessor]).T
    # initialize multiprocessing
    id = 0
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []

    for times_i in timesArray:
        times_i = times_i[~np.isnan(times_i)]
        if len(times_i) == 0:
            continue
        ds_i = ds.loc[{"time": times_i}]
        p = multiprocessing.Process(target=era5_ml_z_p, args=(ds_i, id, return_dict))
        jobs.append(p)
        p.start()
        id += 1
    for proc in jobs:
        proc.join()
    ds_out = xr.concat(dict(sorted(return_dict.items())).values(), dim="time")
    del (return_dict)
    return ds_out


def era5_ml_z_p(ds: xr.Dataset, id: int = 0, return_dict: Optional[dict] = None) -> xr.Dataset:
    """compute z and p on model levels and return ds."""
    import pandas as pd
    R_D = 287.06
    fname = coeff_dir + "/era_5_model_level_definitions.csv"
    coef = pd.read_csv(fname)
    a = coef["a [Pa]"].values
    b = coef['b'].values
    skippedTimes = 0
    # configure progress bar
    times = ds["time"].values
    j = 0
    j_max = len(times)
    startTime = dt.datetime.utcnow()
    for time in ds["time"]:
        if id == 0:
            functions.progressBar2(j, j_max, startTime, message="Converting input data: ")
        j += 1
        try:
            ds_t = ds.loc[{"time": time}]
        except ValueError:
            skippedTimes += 1
            continue
        p_surf = np.exp(ds_t["lnsp"].values[0])
        z_h = ds_t["z"].values[0]
        ds_t["p"] = ds_t["z"].copy(deep=True)
        for level in ds["level"].values[::-1]:
            t_level = ds_t["t"].loc[{"level": level}]
            q_level = ds_t["q"].loc[{"level": level}]
            # compute moist temperature
            t_level = t_level * (1. + 0.609133 * q_level)
            # compute the pressures (on half-levels)
            ph_l1 = a[level - 1] + (b[level - 1] * p_surf)
            ph_l2 = a[level] + (b[level] * p_surf)
            ds_t["p"].loc[{"level": level}] = (ph_l1 + ph_l2) / 2
            if level == 1:
                dlog_p = np.log(ph_l2 / 0.1)
                alpha = np.log(2)
            else:
                dlog_p = np.log(ph_l2 / ph_l1)
                alpha = 1. - ((ph_l1 / (ph_l2 - ph_l1)) * dlog_p)

            t_level = t_level * R_D

            # z_f is the geopotential of this full level
            # integrate from previous (lower) half-level z_h to the
            # full level
            z_f = z_h + (t_level * alpha)
            ds_t["z"].loc[{"level": level}] = z_f
            # z_h is the geopotential of 'half-levels'
            # integrate z_h to next half level
            z_h = z_h + (t_level * dlog_p)
        try:
            ds_out = xr.concat([ds_out, ds_t], dim="time")
        except NameError:
            ds_out = ds_t
    # reverse order of levels
    ds_out["level"] = ds_out["level"].values[::-1]
    ds_out = ds_out.reindex({"level": ds_out["level"].values[::-1]})
    if return_dict is None:
        return ds_out
    else:
        return_dict[id] = ds_out
