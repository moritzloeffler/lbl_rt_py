# -*- coding: utf-8 -*-
# --------------------------------
# Name:       lbl.py
# Purpose:    LBL radiative transfer modelling.
# Author:     Moritz Loeffler
# Created:    2022-11-24
# Python Version:   3.6
# Version:    1.0
# Last Edit:  2025-08-20
# --------------------------------
"""
Line by line RT modelling main program.

"""

#########################
# Modules

import numpy as np
import xarray as xr
import sys
import os
import datetime as dt
import glob
import multiprocessing
from typing import List, Optional, Tuple, Union

try:
    from bottleneck import nanmean, nansum, nanmax
except ModuleNotFoundError:
    from numpy import nanmean, nansum, nanmax

sys.path.append(os.path.dirname(__file__))
import lbl_calc
import lbl_functions
import basic_functions as functions
import lbl_data

# static parameters
g = 9.81  # m/s2

# variable definitions

freqs = {
    "freq_MWR_2CH_LG": np.array([23.8, 31.4]),
    "freq_MWP1": np.array(
        [22.235, 23.035, 23.835, 26.235, 30.000, 51.250, 52.280, 53.850, 54.940, 56.660, 57.290, 58.800]),
    "freq_MWP2": np.array([22.234, 22.5, 23.034, 23.834, 25.0, 26.234, 28.0, 30.0, 51.248, 51.76, 52.28, 52.804, 53.336,
                           53.848, 54.4, 54.94, 55.5, 56.02, 56.66, 57.288, 57.964, 58.8]),
    "freq_hatpro_14": np.array(
        [22.240, 23.040, 23.840, 25.440, 26.240, 27.840, 31.400, 51.260, 52.280, 53.860, 54.940, 56.660, 57.300, 58.00])
}
for key in list(freqs.keys()):
    try:
        np.append(freq_all, freqs[key])
    except NameError:
        freq_all = freqs[key]

_, index_ = np.unique(freq_all, return_index=True)
freqs["freq_all"] = freq_all[index_]


def one_time(ds_in: xr.Dataset, theta, ds_cloud: Optional[xr.Dataset] = None, do_tau_calc: bool = True,
             do_tmr_calc: bool = False, abs_mod='r17', cloud_abs='lie', linew_22='lil05', cont_corr='tur09',
             air_corr='no',
             z_site: int = 0, ds_out: xr.Dataset = xr.Dataset()):
    """Run all processes needed for radiative transfer modelling
    """
    if z_site is None:
        z_site = ds_in["height"].values[0]
    # make sure LWP is not below zero
    ds_in["lwc"][ds_in["lwc"] < 0] = 0
    # mu  =  cos(theta*pi/180.)
    mu = np.cos(np.pi / 180 * theta) + 0.025 * np.exp(-11. * np.cos(np.pi / 180 * theta))
    # mu = sqrt(((re/h_atm) + (h_obs/h_atm))^2.*(cos(!dtor*(90.-el))^2.) + 2.*(re/h_atm)*(1.-(h_obs/h_atm)) - (h_obs/h_atm)^2. + 1.) - ((re/h_atm) + (h_obs/h_atm))*cos(!dtor*(90.-el)) #spherical-geometric (GM120514)
    tau = None
    # make sure all variables are lists if required

    if not (ds_cloud is None):
        z_new = np.append(ds_in["height"].values, ds_cloud["height"].values)
        z_new = np.sort(z_new)
        # ****create new vertical grid combining z_final & z_cloud
        ds_in = ds_in.reindex({'height': z_new}, method=None)
        ds_in["T"] = ds_in["T"].interpolate_na()
        ds_in["q"] = ds_in["q"].interpolate_na()
        ds_in["p"] = ds_in["p"].interpolate_na()
        ds_in["lwc"] = ds_in["p"].copy()
        ds_in["lwc"] = np.zeros(len(ds_in["height"]))

        # distribute lwc according to cloud layer
        for height in ds_cloud["height"]:
            ds_in["lwc"].loc[{"height": slice(ds_cloud["z_base"].loc[{"height": height}],
                                              ds_cloud["z_top"].loc[{"height": height}])}] = ds_cloud["lwc"].loc[
                {"height": height}]
    elif not ("lwc" in list(ds_in.keys())):
        ds_in["lwc"] = np.zeros(len(ds_in["height"]))
    # radiative transfer
    if do_tau_calc:
        ds_out = lbl_calc.tau(ds_in, abs_mod=abs_mod, cloud_abs=cloud_abs,
                              linew_22=linew_22, cont_corr=cont_corr)

    ds_out = lbl_calc.mu(ds_in, ds_out, theta, air_corr=air_corr, z_site=z_site)  # calculate inverse air mass
    ds_out = lbl_calc.tb_pl(ds_in, ds_out)
    # calculate TB

    if do_tmr_calc:
        ds_out = lbl_calc.tmr(ds_in, ds_out)

    return ds_out


def multiple_times(ds_in: xr.Dataset, thetas, f_out: str, ds_cloud: Optional[xr.Dataset] = None,
                   do_tmr_calc: bool = False, abs_mod='r17', cloud_abs='ell', linew_22='lil05', cont_corr='tur09',
                   air_corr='no',
                   z_site: Optional[float] = 0, comment: str = "", verbose: bool = True):
    """Return the brightness temperatures for multiple cases and save time series in outfile."""
    do_tau_calc_init = True
    j = 0
    j_max = len(ds_in["time"])
    year = 1970
    startTime = dt.datetime.utcnow()
    levelVars = ["p", "height", "level"]
    for time in ds_in["time"].values:
        if verbose:
            functions.progressBar2(j, j_max, startTime, message="Progress: ")
        j += 1
        # save yearly files
        if year == 1970:
            year = time.astype("datetime64[s]").astype(dt.datetime).year
        elif year < time.astype("datetime64[s]").astype(dt.datetime).year:
            try:
                ds_out_y = ds_out.copy()
                ds_out_y = ds_out_y.reindex({"time": np.sort(ds_out_y["time"])})
                ds_out_y.loc[{"time": slice(np.datetime64(str(year)), np.datetime64(str(year + 1)))}
                ].to_netcdf(f_out[:-3] + "_" + str(year) + ".nc")
                del (ds_out_y)
            except Exception:  # this is not essential (NameError, ValueError)
                pass
            year = time.astype("datetime64[s]").astype(dt.datetime).year
        try:
            del (ds_t)
        except UnboundLocalError:
            pass
        for levelVar in levelVars:
            if not (levelVar in list(ds_in.coords)):
                continue
            ds_in_t = lbl_functions.extend_atm_profile(
                ds_in.loc[{"time": time,
                           levelVar: ds_in[levelVar][~np.isnan(ds_in["T"].loc[{"time": time}]).values]}], levelVar)
        ds_in_t = lbl_functions.addGroundLevel(ds_in_t, z_site)
        do_tau_calc = do_tau_calc_init
        ds_t_t = xr.Dataset()
        try:
            for i in np.arange(len(thetas)):
                theta = thetas[i]
                ds_t_t = one_time(ds_in_t, theta, ds_cloud, do_tau_calc, do_tmr_calc, abs_mod, cloud_abs, linew_22,
                                  cont_corr, air_corr, z_site, ds_t_t)
                do_tau_calc = False
                try:
                    ds_t = xr.merge([ds_t, ds_t_t.expand_dims(dim={"ele": [90 - theta]}, axis=-1)])  # , dim="ele")
                except NameError:
                    ds_t = ds_t_t.expand_dims(dim={"ele": [90 - theta]}, axis=-1)
                    keys = list(ds_t.keys())
                    keep_keys = ["tb", "I"]
                    drop_keys = [i for i in keys if not (i in keep_keys)]
            ds_t = functions.dropDims(ds_t, drop_keys, "ele")
        except TypeError as e:
            if ("has no len()" in str(e)) or ("object is not iterable" in str(e)):
                ds_t = one_time(ds_in_t, thetas, ds_cloud, do_tau_calc, do_tmr_calc, abs_mod, cloud_abs, linew_22,
                                cont_corr, air_corr, z_site)
                keys = list(ds_t.keys())
                keep_keys = ["tb", "I"]
                drop_keys = [i for i in keys if not (i in keep_keys)]
            else:
                raise
        ds_t.coords["time"] = ds_in_t["time"]
        try:
            ds_t = ds_t.drop_vars(drop_keys)
        except ValueError:
            pass
        try:
            ds_out = xr.concat([ds_out, ds_t], dim="time")
        except NameError:
            ds_out = ds_t.copy()
    if verbose:
        print("")
    ds_out = ds_out.reindex({"time": np.sort(ds_in["time"].values)})  # putting the times in order
    ds_out.attrs = {"absorption model": abs_mod, "cloud absorption": cloud_abs, "line width 22 (R98)": linew_22,
                    "continuum contribution (R98)": cont_corr, "air mass correction": air_corr,
                    "z site above sea level": str(z_site), "comment": comment}
    ds_out.to_netcdf(f_out.format(**{"path": os.path.dirname(__file__)}))


def multiple_times_mp(ds_in: xr.Dataset, thetas, f_out: str, ds_cloud: Optional[xr.Dataset] = None,
                      do_tmr_calc: bool = False, abs_mod='r17', cloud_abs='ell', linew_22='lil05', cont_corr='tur09',
                      air_corr='no',
                      z_site: Optional[float] = 0, comment: str = "", verbose=True):
    """Return the brightness temperatures for multiple cases and save time series in outfile. Multiprocessing version."""
    levelVars = ["p", "height", "level"]
    # configure variables for multiprocessing
    times = ds_in["time"].values
    nProcessor = multiprocessing.cpu_count()
    if len(times) % nProcessor != 0:
        times = np.append(times, [np.datetime64("NaT")] * (nProcessor - len(times) % nProcessor))
    timesArray = np.reshape(times, [int(len(times) / nProcessor), nProcessor])
    # configure progressBar
    j = 0
    j_max = np.shape(timesArray)[0]
    year = 1970
    startTime = dt.datetime.utcnow()
    for times_i in timesArray:
        ds_in_list = []
        if verbose:
            functions.progressBar2(j, j_max, startTime, message="Progress: ")
        j += 1
        # save yearly files
        try:
            if year == 1970:
                year = times_i[0].astype("datetime64[s]").astype(dt.datetime).year
            elif year < times_i[0].astype("datetime64[s]").astype(dt.datetime).year:
                ds_out_y = ds_out.copy()
                ds_out_y = ds_out_y.reindex({"time": np.sort(ds_out_y["time"])})
                ds_out_y.loc[{"time": slice(np.datetime64(str(year)), np.datetime64(str(year + 1)))}
                ].to_netcdf(f_out[:-3] + "_" + str(year) + ".nc")
                year = times_i[0].astype("datetime64[s]").astype(dt.datetime).year
        except:  # this is not essential (NameError, ValueError, KeyError)
            pass
        for time in times_i:
            try:
                for levelVar in levelVars:
                    if not (levelVar in list(ds_in.coords)):
                        continue
                    ds_in_t = lbl_functions.extend_atm_profile(
                        ds_in.loc[{"time": time,
                                   levelVar: ds_in[levelVar][~np.isnan(ds_in["T"].loc[{"time": time}]).values]}],
                        levelVar)
            except KeyError as e:
                if "are required coordinates" in str(e):
                    raise
                else:
                    # tried to select np.nat value, which was included to allow shape of timeArray
                    continue
            ds_in_t = lbl_functions.addGroundLevel(ds_in_t, z_site)
            ds_in_list.append(ds_in_t)
        id = 0
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        for ds_in_i in ds_in_list:
            p = multiprocessing.Process(target=mp_call, args=(id, ds_in_i, thetas, ds_cloud, do_tmr_calc, abs_mod,
                                                              cloud_abs, linew_22, cont_corr, air_corr, z_site,
                                                              return_dict))
            jobs.append(p)
            p.start()
            id += 1
        for proc in jobs:
            proc.join()

        try:
            ds_out = xr.concat([ds_out, xr.concat(dict(sorted(return_dict.items())).values(), dim="time")], dim="time")
        except NameError:
            ds_out = xr.concat(dict(sorted(return_dict.items())).values(), dim="time")
        del (return_dict)
    if verbose:
        print("")
    ds_out = ds_out.reindex({"time": np.sort(ds_in["time"].values)})  # putting the times in order
    ds_out.attrs = {"absorption model": abs_mod, "cloud absorption": cloud_abs, "line width 22 (R98)": linew_22,
                    "continuum contribution (R98)": cont_corr, "air mass correction": air_corr,
                    "z site above sea level": str(z_site), "comment": comment}
    ds_out.to_netcdf(f_out.format(**{"path": os.path.dirname(__file__)}))


def mp_call(id: int, ds_in_t, thetas, ds_cloud, do_tmr_calc, abs_mod, cloud_abs, linew_22, cont_corr, air_corr, z_site,
            return_dict):
    """Run one process thread for multiple times."""
    do_tau_calc = True
    ds_t_t = xr.Dataset()
    try:
        for i in np.arange(len(thetas)):
            theta = thetas[i]
            ds_t_t = one_time(ds_in_t, theta, ds_cloud, do_tau_calc, do_tmr_calc, abs_mod, cloud_abs, linew_22,
                              cont_corr, air_corr, z_site, ds_t_t)
            do_tau_calc = False
            try:
                ds_t = xr.merge([ds_t, ds_t_t.expand_dims(dim={"ele": [90 - theta]}, axis=-1)])  # , dim="ele")
            except NameError:
                ds_t = ds_t_t.expand_dims(dim={"ele": [90 - theta]}, axis=-1)
                # drop ele dim from all variables except tb
                keys = list(ds_t.keys())
                keep_keys = ["tb", "I"]
                drop_keys = [i for i in keys if not (i in keep_keys)]
        ds_t = functions.dropDims(ds_t, drop_keys, "ele")
    except TypeError as e:
        if ("has no len()" in str(e)) or ("object is not iterable" in str(e)):
            ds_t = one_time(ds_in_t, thetas, ds_cloud, do_tau_calc, do_tmr_calc, abs_mod, cloud_abs, linew_22,
                            cont_corr, air_corr, z_site)
            keys = list(ds_t.keys())
            keep_keys = ["tb", "I"]
            drop_keys = [i for i in keys if not (i in keep_keys)]
        else:
            raise
    try:
        ds_t = ds_t.drop_vars(drop_keys)
    except ValueError:
        pass
    try:
        ds_t = ds_t.drop_dims("height2")
    except ValueError:
        pass
    ds_t.coords["time"] = ds_in_t["time"]
    return_dict[str(id)] = ds_t


def rt_era5(timePeriod: List[np.datetime64], f_out: str, theta: Union[float, np.array], freq: np.array,
            dir_in: Optional[str] = None, files_in: Optional[str] = None, file_list: Optional[str] = None,
            clear_sky_only: bool = False, use_multiprocessing: bool = True,
            station_height: Optional[float] = None, comment: str = "", do_tmr_calc: bool = False, abs_mod: str = 'r22',
            cloud_abs: str = 'ell07', linew_22: str = 'lil05', cont_corr: str = 'tur09',
            air_corr: str = 'rueeger_aver_02', **__):
    """Perform RT calculations for ERA5 dataset."""
    # load files in time period
    ds_in = lbl_data.load_era5(timePeriod, dir_in, files_in, file_list, clear_sky_only)
    ds_in["f"] = xr.DataArray(data=freq)
    # rt calculations
    if use_multiprocessing:
        multiple_times_mp(ds_in, theta, f_out, do_tmr_calc=do_tmr_calc, abs_mod=abs_mod, cloud_abs=cloud_abs,
                          linew_22=linew_22, cont_corr=cont_corr, air_corr=air_corr, z_site=station_height,
                          comment=comment)
    else:
        multiple_times(ds_in, theta, f_out, do_tmr_calc=do_tmr_calc, abs_mod=abs_mod, cloud_abs=cloud_abs,
                          linew_22=linew_22, cont_corr=cont_corr, air_corr=air_corr, z_site=station_height,
                          comment=comment)


def rt_era5ml(timePeriod: List[np.datetime64], f_out: str, theta: Union[float, np.array], freq: np.array,
              dir_in: Optional[str] = None, files_in: Optional[str] = None, file_list: Optional[str] = None,
              clear_sky_only: bool = False, use_multiprocessing: bool = True,
              station_height: Optional[float] = None, comment: str = "", do_tmr_calc: bool = False,
              abs_mod: str = 'r22',
              cloud_abs: str = 'ell07', linew_22: str = 'lil05', cont_corr: str = 'tur09',
              air_corr: str = 'rueeger_aver_02', **__):
    """Perform RT calculations for ERA5 model level dataset."""
    # load files in time period
    ds_in = lbl_data.load_era5ml(timePeriod, dir_in, files_in, file_list, clear_sky_only)
    ds_in["f"] = xr.DataArray(data=freq)
    # rt calculations
    if use_multiprocessing:
        multiple_times_mp(ds_in, theta, f_out, do_tmr_calc=do_tmr_calc, abs_mod=abs_mod, cloud_abs=cloud_abs,
                          linew_22=linew_22, cont_corr=cont_corr, air_corr=air_corr, z_site=station_height,
                          comment=comment)
    else:
        multiple_times(ds_in, theta, f_out, do_tmr_calc=do_tmr_calc, abs_mod=abs_mod, cloud_abs=cloud_abs,
                          linew_22=linew_22, cont_corr=cont_corr, air_corr=air_corr, z_site=station_height,
                          comment=comment)


def rt_icond2(timePeriod: List[np.datetime64], f_out: str, theta: Union[float, np.array], freq: np.array, hhl_file: str,
              dir_in: Optional[str] = None, files_in: Optional[str] = None, file_list: Optional[str] = None,
              clear_sky_only: bool = False, use_multiprocessing: bool = True,
              station_height: Optional[float] = None, comment: str = "", do_tmr_calc: bool = False,
              abs_mod: str = 'r22',
              cloud_abs: str = 'ell07', linew_22: str = 'lil05', cont_corr: str = 'tur09',
              air_corr: str = 'rueeger_aver_02', **__):
    """Perform RT calculations for ICON-D2 dataset. Should work for other icon resolutions."""
    # load files in time period
    ds_in = lbl_data.load_icond2(timePeriod, hhl_file= hhl_file, dir_in= dir_in, files_in= files_in,
                                 file_list = file_list, clear_sky_only = clear_sky_only)

    ds_in["f"] = xr.DataArray(data=freq)
    # rt calculations
    if use_multiprocessing:
        multiple_times_mp(ds_in, theta, f_out, do_tmr_calc=do_tmr_calc, abs_mod=abs_mod, cloud_abs=cloud_abs,
                       linew_22=linew_22, cont_corr=cont_corr, air_corr=air_corr, z_site=station_height,
                       comment=comment)
    else:
        multiple_times(ds_in, theta, f_out, do_tmr_calc=do_tmr_calc, abs_mod=abs_mod, cloud_abs=cloud_abs,
                       linew_22=linew_22, cont_corr=cont_corr, air_corr=air_corr, z_site=station_height,
                       comment=comment)


def rt_standard_atm(f_out: str, theta: Union[float, np.array], freq: np.array,
                    station_height: Optional[float] = None, comment: str = "", do_tmr_calc: bool = False,
                    abs_mod: str = 'r22',
                    cloud_abs: str = 'ell07', linew_22: str = 'lil05', cont_corr: str = 'tur09',
                    air_corr: str = 'rueeger_aver_02', **__):
    """Perform RT calculations for ERA5 dataset."""
    ds_in = lbl_data.load_standard_atm()
    ds_in["f"] = xr.DataArray(data=freq)
    # rt calculations
    multiple_times(ds_in, theta, f_out, do_tmr_calc=do_tmr_calc, abs_mod=abs_mod, cloud_abs=cloud_abs,
                   linew_22=linew_22, cont_corr=cont_corr, air_corr=air_corr, z_site=station_height,
                   comment=comment)

def run(config_path: str):
    """Run complete radiative transfer as specified in config."""
    config = functions.loadConfig(config_path)

    # delay calculation by hours
    delay = 0  # hours
    hours = np.arange(delay)
    for hour in hours[::-1]:
        print("starting in %d hours" % (hour + 1))
        time.sleep(60 ** 2)

    freq = freqs[config["freqs"]]
    ## calculate theta from ele
    try:
        (config["rt_config"])["theta"] = 90 - (config["rt_config"])["ele"]  # zenith angle
    except TypeError:
        (config["rt_config"])["theta"] = 90 - np.array((config["rt_config"])["ele"])
    #
    getTimePeriod = lambda conf: [np.datetime64(conf["timePeriod"][0]),
                                  np.datetime64(conf["timePeriod"][1])]
    model = config["model"]
    if model == "era5":  # era
        timePeriod = getTimePeriod(config)
        rt_era5(timePeriod, freq = freq, **config["rt_config"])
    elif model == "era5ml":  # era
        timePeriod = getTimePeriod(config)
        rt_era5ml(timePeriod, freq = freq, **config["rt_config"])
    elif model == "icon":  # icon
        timePeriod = getTimePeriod(config)
        rt_icond2(timePeriod, freq = freq, **config["rt_config"])
    elif model == "st-atm":
        rt_standard_atm(freq = freq, **config["rt_config"])


if __name__ == "__main__":
    import time

    # load config
    config_path = sys.argv[1]
    if config_path == "examples":
        example_configs = ["era5_r22_config.json", "era5ml_r22_mp_config.json",
                           "icond2_r98_config.json", "statm_r22_config.json"]
        example_dir = os.path.dirname(__file__) + "/config/examples/"
        for config_file in example_configs:
            run(example_dir + config_file)
    else:
        run(config_path)