# -*- coding: utf-8 -*-
# --------------------------------
# Name:       lbl_calc.py
# Purpose:    LBL RT model required higher level "calc" functions.
# Author:     Moritz Loeffler
# Created:    2022-11-24
# Python Version:   3.6
# Version:    1.0
# Last Edit:  2025-08-20
# --------------------------------
"""
Line by line RT modelling translated from IDL code (Ulrich Loehnert et al).

"""
import sys
import xarray as xr
import numpy as np
import os

sys.path.append(os.path.dirname(__file__))
import lbl_functions
import lbl_absorption_models

# global variables
h = 6.6262e-34  # Planck constant
kB = 1.3806e-23  # Boltzmann constant
c = 2.997925 * 1e8  # c.

def tau(ds: xr.Dataset, abs_mod: str, cloud_abs:str = 'ell', linew_22:str = 'lil05', cont_corr:str = 'tur09'):
    """Calculate the optical depth (tau) and return."""
    # check if specified model is implemented
    if not (abs_mod in ['r98', 'r20', "r22"]):
        sys.exit("Unknown gas absorption model %s" % abs_mod)
    if not (cloud_abs in ['ula', 'lie', 'ell', 'sto', 'ray', "ell07"]):
        sys.exit("Unknown liquid absorption model %s" % cloud_abs)
    if cloud_abs in ['ula', 'sto', 'ray']:
        sys.exit("Liquid absorption model %s is not implemented." % cloud_abs)
    if not (cont_corr in ['org', 'tur09']):
        sys.exit("Unknown WV continuum correction: %s" % cont_corr)
    if not (linew_22 in ['org', 'lil05']):
        sys.exit("Unknown line width modification: %s" % linew_22)
    # variable definitions
    #wavel = 0.299792458 / f  # wavelength in m
    ds_tau = xr.Dataset()
    vars = ["abs_all", "abs_wv", "abs_o2", "abs_liq", "tau", "tau_wv", "tau_o2", "tau_liq"]
    n_heights = len(ds["height"].values)
    n_freq = len(ds["f"].values)
    for var in vars:
        ds_tau[var] = xr.DataArray(np.zeros((n_heights-1, n_freq)),
                                   dims=["height2", "f"],
                                   coords={"height2": (["height2"], (ds["height"][1:].values+ds["height"][:-1].values)/2),
                                           "f": (["f"], ds["f"])})

    ds_tau["delta_z"] = xr.DataArray(ds["height"].values[1:]-ds["height"].values[:-1],
                                   dims=["height2"],
                                   coords={"height2": (["height2"], (ds["height"][1:].values+ds["height"][:-1].values)/2)})
    ds_tau["T_mean"] = ds_tau["delta_z"].copy()
    ds_tau["delta_p"] = ds_tau["delta_z"].copy()
    ds_tau["T_mean"][:] = (ds["T"].values[1:] + ds["T"].values[:-1]) / 2
    ds_tau["delta_p"][:] = ds["p"].values[1:] - ds["p"].values[:-1]
    if np.any(ds_tau["delta_p"] >= 0):
        n_ps = np.arange(len(ds_tau["delta_p"]))[ds_tau["delta_p"] >= 0]
        for n_p in n_ps:
            ds["p"][n_p] = ds["p"][n_p - 1] - 0.1
            if ds_tau["delta_p"][n_p] > 1:
                print('Warning: p profile adjusted by %d.1 to assure monotonic decrease!' % ds_tau["delta_p"].values[n_p])
    ds_tau["p_mean"] = ds_tau["delta_z"].copy()
    xp = -np.log(ds["p"].values[1:] / ds["p"].values[:-1])
    ds_tau["p_mean"][:] = -ds["p"].values[:-1] / xp * (np.exp(-xp) - 1.0)
    ds_tau["absolute_humidity_mean"] = ds_tau["delta_z"].copy()
    ds_tau["absolute_humidity_mean"][:] = (ds["absolute_humidity"].values[1:] + ds["absolute_humidity"].values[:-1]) / 2
    ds_tau["pda_mean"] = ((ds_tau["p_mean"]/100.) - ((ds_tau["absolute_humidity_mean"]*1000.)*ds_tau["T_mean"]/216.68))*100

    # gas absorption
    if abs_mod == "r98":
        r98 = lbl_absorption_models.R98(linew_22=linew_22, cont_corr=cont_corr)
        awv_m = lambda ds_: r98.wvr(ds_) / 1000
        ao2_m = lambda ds_: r98.o2(ds_) / 1000
        # nitrogen (only with Rosenkranz o2)
        an2_m = lambda ds_: lbl_absorption_models.n2(ds_) / 1000
        abs_total_m = lambda ds_: awv_m(ds_) + ao2_m(ds_) + an2_m(ds_)
    elif abs_mod == "r20":
        awv_m = lambda ds_: lbl_absorption_models.wvr20(ds_) / 1000
        ao2_m = lambda ds_: lbl_absorption_models.o2r20(ds_) / 1000
        # nitrogen (only with Rosenkranz o2)
        an2_m = lambda ds_: lbl_absorption_models.n2(ds_) / 1000
        abs_total_m = lambda ds_: awv_m(ds_) + ao2_m(ds_) + an2_m(ds_)
    elif abs_mod == "r22":
        r22 = lbl_absorption_models.R22()
        awv_m = lambda ds_: r22.wvr_fast(ds_) / 1000
        ao2_m = lambda ds_: r22.o2_fast(ds_) / 1000
        # nitrogen (only with Rosenkranz o2)
        an2_m = lambda ds_: lbl_absorption_models.n2(ds_) / 1000
        abs_total_m = lambda ds_: awv_m(ds_) + ao2_m(ds_) + an2_m(ds_)

    # cloud absorption
    if cloud_abs == "ula":
        # not implemented
        aql_m = lambda ds_, ds_mean: lbl_absorption_models.dielco(ds_, ds_mean)
    elif cloud_abs == "lie":
        aql_m = lambda ds_, ds_mean: lbl_absorption_models.abliq(ds_, ds_mean)
    elif cloud_abs == "ell":
        aql_m = lambda ds_, ds_mean: lbl_absorption_models.rewat_ellison(ds_, ds_mean)
    elif cloud_abs == "ell07":
        aql_m = lambda ds_, ds_mean: lbl_absorption_models.refwat_ellison07(ds_, ds_mean)
    elif cloud_abs == "sto":
        # not implemented
        aql_m = lambda ds_, ds_mean: lbl_absorption_models.rewat_stogryn(ds_, ds_mean)
    elif cloud_abs == "ray":
        # not implemented
        aql_m = lambda ds_, ds_mean: lbl_absorption_models.rewat_ray(ds_, ds_mean)
    levelVars = ["p", "height", "level"]


    delta_z = np.tile(ds_tau["delta_z"].values, ( n_freq, 1)).T
    for i in np.arange(len(ds_tau["height2"]))[::-1]:
        h2 = ds_tau["height2"][i]
        ds_tau_h = ds_tau.loc[{"height2": ds_tau["height2"][i]}]
        for levelVar in levelVars:
            try:
                ds_h = ds.loc[{levelVar: ds[levelVar][i]}]
                levelVars = [levelVar]
                break
            except ValueError:
                continue
        ds_tau["abs_all"].loc[{"height2": h2}] = abs_total_m(ds_tau_h) + aql_m(ds_h, ds_tau_h)
        ds_tau["abs_wv"].loc[{"height2": h2}] = awv_m(ds_tau_h)
        ds_tau["abs_o2"].loc[{"height2": h2}] = ao2_m(ds_tau_h)
        ds_tau["abs_liq"].loc[{"height2": h2}] = aql_m(ds_h, ds_tau_h)
        ds_tau["abs_all"].loc[{"height2": h2}] = ds_tau["abs_wv"].loc[{"height2": h2}] + \
                                                 ds_tau["abs_o2"].loc[{"height2": h2}] + \
                                                 ds_tau["abs_liq"].loc[{"height2": h2}] + an2_m(ds_tau_h)

        ds_tau["tau"].loc[{"height2": h2}] = np.sum(ds_tau["abs_all"].values * delta_z, axis = 0)
        ds_tau["tau_wv"].loc[{"height2": h2}] = np.sum(ds_tau["abs_wv"].values * delta_z, axis = 0)
        ds_tau["tau_o2"].loc[{"height2": h2}] = np.sum(ds_tau["abs_o2"].values * delta_z, axis = 0)
        ds_tau["tau_liq"].loc[{"height2": h2}] = np.sum(ds_tau["abs_liq"].values * delta_z, axis = 0)

    keep_keys = ["tau", "tau_wv", "tau_o2", "tau_liq"]
    drop_keys = [i for i in list(ds_tau.keys()) if not (i in keep_keys)]
    ds_tau = ds_tau.drop_vars(drop_keys)
    return ds_tau


def mu(ds, ds_out, theta0, air_corr: str, z_site = 0):
    """Calculate the air mass correction (mu)."""
    # variable definitions
    re = 6370950. + z_site
    ele = 90 - theta0
    theta0 = np.pi / 180 * theta0

    vars = ["mu"]
    n_heights = len(ds["height"].values)
    n_freq = len(ds["f"].values)
    for var in vars:
        ds_out[var] = xr.DataArray(np.zeros((n_heights - 1, n_freq)),
                                   dims=["height2", "f"],
                                   coords={"height2": (["height2"], (ds["height"][1:].values + ds["height"][:-1].values) / 2),
                                           "f": (["f"], ds["f"])})
    ds_out["delta"] = xr.DataArray(np.zeros(n_heights - 1),
                                  dims=["height2"],
                                  coords={"height2": (["height2"], (ds["height"][1:].values + ds["height"][:-1].values) / 2)})

    # select coefficients
    if air_corr == "thayer_74":
        coeff = [77.604, 64.79, 3.776]
    elif air_corr == "liebe_77":
        coeff = [77.676, 71.631, 3.74656]
    elif air_corr == "hill_80":
        coeff = [0., 98., 3.58300]
    elif air_corr == "bevis_94":
        coeff = [77.6, 70.4, 3.739]
    elif air_corr == "rueeger_avai_02":
        coeff = [77.695, 71.97, 3.75406]
    elif air_corr == "rueeger_aver_02":
        coeff = [77.689, 71.2952, 3.75463]
    elif air_corr == "sphere":
        coeff = [0., 0., 0.]
    elif air_corr == "43":
        coeff = [0., 0., 0.]
        re = 4. / 3. * re
    else:
        if air_corr == "no":
            ds_out["mu"][:,:] = np.cos(theta0)
        elif air_corr == "rozenberg_66":
            ds_out["mu"][:,:] = np.cos(theta0) + 0.025 * np.exp(-11 * np.cos(theta0))
        elif air_corr == "young_94":
            ds_out["mu"][:,:] = (np.cos(theta0)**3. + 0.149864 * np.cos(theta0)**2. + 0.0102963 * np.cos(theta0) +
                                0.000303978)/(1.002432 * np.cos(theta0)**2. + 0.148386 * np.cos(theta0) + 0.0096467)
        elif air_corr == "pickering_02":
            ds_out["mu"][:,:] = np.sin(np.pi / 180 * (ele + 244./(165. + 47. * ele**1.1)))
        return ds_out
    ds = lbl_functions.abs_hum_to_mixr(ds)
    ds = lbl_functions.w2e(ds)
    if air_corr == "liebe_93":
        # unclear how ref is defined/calculated
        sys.exit("Air mass correction 'liebe_93' is not implemented.")
        ds_out["n_top"] = ds_out["delta"].copy()
        ds_out["n_bot"] = ds_out["delta"].copy()
        for i in np.arange(1, len(ds["height"])):
            ds_out["n_top"].loc[{"height2": ds_out["height2"][i-1]}] = 1 + \
                (ds["ref"].loc[{"height": ds["height"][i-1]}] * 1e-6)
            if i > 1:
                ds_out["n_bot"].loc[{"height2": ds_out["height2"][i-1]}] = 1 + \
                    (ds["ref"].loc[{"height": ds["height"][i-2]}] * 1e-6)
            else:
                ds_out["n_bot"].loc[{"height2": ds_out["height2"][i-1]}] = ds_out["n_top"].values
    else:
        ds_out["n_top"] = ds_out["delta"].copy()
        T_top = (ds["T"][1:].values + ds["T"][:-1].values) / 2
        p_top = (ds["p"][1:].values + ds["p"][:-1].values) / 2
        e_top = (ds["e"][1:].values + ds["e"][:-1].values) / 2
        ds_out["n_top"][:] = 1 + (coeff[0]*(((p_top/100.)-e_top)/T_top) +
                              coeff[1]*(e_top/T_top) +
                              coeff[2]*(e_top/(T_top**2.)))*1e-6

        ds_out["n_bot"] = ds_out["delta"].copy()
        T_bot = (ds["T"][1:-1].values + ds["T"][:-2].values) / 2
        p_bot = (ds["p"][1:-1].values + ds["p"][:-2].values) / 2
        e_bot = (ds["e"][1:-1].values + ds["e"][:-2].values) / 2
        n_bot = 1 + (coeff[0]*(((p_bot/100.)-e_bot)/T_bot) +
                     coeff[1]*(e_bot/T_bot) +
                     coeff[2]*(e_bot/(T_bot**2.)))*1e-6
        ds_out["n_bot"][:] = np.append(np.array([ds_out["n_top"].values[0]]), n_bot)
    ds_out["delta_z"] = ds_out["delta"].copy()
    ds_out["delta_z"][:] = ds["height"].values[1:]-ds["height"].values[:-1]
    r_bot = re
    theta_bot = theta0
    for i in np.arange(len(ds_out["height2"])):
        # from RADIOWAVE PROPAGATION , Levis, pp.121
        r_top = r_bot + ds_out["delta_z"].values[i]
        theta_top = np.arcsin(((ds_out["n_bot"].values[i] * r_bot) / (ds_out["n_top"].values[i] * r_top)) * np.sin(theta_bot))
        # from cosine law (r_top^2 = ds^2 + r_bot^2 -2*r_bot^2*ds*cos(alpha) -> ds^2 +-p/2*sqrt((p/2)^2.-q) ... ds =)
        alpha = np.pi - theta_bot
        ds_out["delta"][i] = r_bot * np.cos(alpha) + np.sqrt(
            r_top ** 2. + r_bot ** 2. * ((np.cos(alpha)) ** 2. - 1.))  # use solution >0.)
        # from DOPPLER RADAR AND WEATHER OBSERVATIONS , Dvorak/Zrnic, pp.18 (elevation phi = asin((ds*cos(phi)/re) + sin(phi)) and phi = !pi/2 - theta_bot # used by Veronique Meunier)
        #dels = np.sqrt((re * np.cos(theta_top)) ** 2. + ds_out["delta_z"].values[i] ** 2. + 2. * re * ds_out["delta_z"].values[i]) - re * np.cos(theta_top)
        #phi = np.arcsin((dels*np.cos(phi)/re) + np.sin(phi))
        ds_out["mu"].loc[{"height2": ds_out["height2"][i]}] = ds_out["delta_z"].values[i] / ds_out["delta"][i]
        theta_bot = theta_top
        r_bot = r_top
    keep_keys = ["ref", "delta", "mu", "tau", "tau_wv", "tau_o2", "tau_liq"]
    drop_keys = [i for i in list(ds_out.keys()) if not (i in keep_keys)]
    ds_out = ds_out.drop_vars(drop_keys)
    return ds_out


def tb_pl(ds: xr.Dataset, ds_out: xr.Dataset):
    """Calculate the radiation intensity IN and the brightness temperature TB."""
    # variable definitions

    f = ds["f"]*1e9
    n_height = len(ds["height"])
    wavelength = c/f
    IN = 2.73 * np.ones(len(f))  # cosmic background
    IN = (2. * h * f / (wavelength ** 2.)) * 1. / (np.exp(h * f / (kB * IN)) - 1.)

    tau_top = np.zeros(len(f))
    tau_bot = ds_out["tau"].loc[{"height2": ds_out["height2"].values[-1]}]
    for i in np.arange(n_height - 1):
        try:
            tau_top = ds_out["tau"].loc[{"height2": ds_out["height2"].values[n_height - 1 - i]}]
            tau_bot = ds_out["tau"].loc[{"height2": ds_out["height2"].values[n_height - 1 - i - 1]}]
        except IndexError:
            pass
        delta_tau = tau_bot - tau_top
        if np.any(delta_tau < 0):
            print('warning, negative absorption coefficient')
            ds_out["tb"] = xr.DataArray([np.nan]*len(f), dims=["f"], coords={"f": (["f"], ds["f"])})
            return ds_out
        elif np.any(delta_tau == 0):
            print('warning, zero absorption coefficient')
            ds_out["tb"] = xr.DataArray([np.nan]*len(f), dims=["f"], coords={"f": (["f"], ds["f"])})
            return ds_out
        h2 = ds_out["height2"].values[-i - 1]
        mu_i = ds_out["mu"].loc[{"height2": h2}]
        A = np.ones_like(ds["f"]) - np.exp(- delta_tau / mu_i)
        B = delta_tau - mu_i * (1 - np.exp(- delta_tau / mu_i))

        T_pl2 = (2. * h * f / (wavelength ** 2.)) * 1. / (np.exp(h * f / (kB * ds["T"].values[n_height - 2 - i])) - 1)
        T_pl1 = (2. * h * f / (wavelength ** 2.)) * 1. / (np.exp(h * f / (kB * ds["T"].values[n_height - 1 - i])) - 1)
        diff = (T_pl2 - T_pl1).values / delta_tau.values
        IN = IN * np.exp(-delta_tau.values / mu_i.values) + T_pl1.values * A.values + diff * B.values

    TB = h * f / kB / np.log((2 * h * f / (IN * wavelength ** 2.)) + 1.)
    ds_out["tb"] = xr.DataArray(TB, dims=["f"], coords={"f": (["f"], ds["f"])})
    ds_out["I"] = xr.DataArray(IN, dims=["f"], coords={"f": (["f"], ds["f"])})

    return ds_out

def tmr(ds: xr.Dataset, ds_out: xr.Dataset) -> xr.Dataset:
    """Calculate mean radiating temperature Tmr."""
    n_height = len(ds["height"])
    tau_top = np.zeros(len(ds["f"]))
    tau_bot = ds_out["tau"].loc[{"height2": ds_out["height2"].values[-1]}]
    tau_surf = ds_out["tau"].loc[{"height2": ds_out["height2"].values[0]}]
    T_mr = np.zeros(len(ds["f"]))
    for i in np.arange(n_height - 2):
        try:
            tau_top = ds_out["tau"].loc[{"height2": ds_out["height2"].values[n_height - 2 - i]}]
            tau_bot = ds_out["tau"].loc[{"height2": ds_out["height2"].values[n_height - 2 - i - 1]}]
        except IndexError:
            pass
        delta_tau = tau_bot - tau_top
        if np.max(delta_tau) < 1e-5:
            continue
        T_mr = (ds["T"][n_height - 2 - i] * np.exp(-(tau_surf - tau_bot) / ds_out["mu"]) * delta_tau /
                ds_out["mu"]) + T_mr
    ds_out["tmr"] = xr.DataArray(T_mr, dims=["f"], coords={"f": (["f"], ds["f"])})
    return ds_out

