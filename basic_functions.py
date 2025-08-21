# -*- coding: utf-8 -*-
# --------------------------------
# Name:       basic_functions.py
# Purpose:    Collection of useful functions.
# Author:     Moritz Loeffler 
# Created:    20.05.2020
# Python Version:   3.8
# Version:    1
# Last Edit:  2023-05-25
# --------------------------------
"""
Collection of functions that are repeated across all modules.
"""
import sys
import os
import json
import xarray as xr
from typing import Union, List, Any, Optional
import datetime as dt
import glob


def loadConfig(config_file: str) -> dict:
    """
    Load config files and return params as dictionary.
    """
    # fname = "/env_vars.json"
    config = dict()

    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            try:
                envvars = dict(**json.load(f))
            except json.decoder.JSONDecodeError:
                raise
    else:
        raise FileNotFoundError('File not found: ' + config_file)
    return envvars

def dropDims(ds: xr.Dataset, vars: List[str], dim: str) -> xr.Dataset:
    """Drop dim from variables (vars) and return ds."""
    try:
        dsTemp = ds.copy()
        ds = ds.drop_dims(dim)
        keys = list(dsTemp.keys())
        for key in keys:
            if not (key in vars):
                ds[key] = dsTemp[key]
            else:
                ds[key] = dsTemp[key].loc[{dim: dsTemp[dim][0]}]
    except AttributeError as e:
        if "NoneType" in str(e):
            pass
        else:
            raise
    except ValueError as e:
        if "Dataset does not contain the dimensions:" in str(e):
            return dsTemp
        else:
            raise
    return ds


def progressBar(current: Union[float, int], total: Union[float, int], time):
    """write a progress bar to stout."""
    i = int(current / total * 50)
    sys.stdout.write('\r')
    sys.stdout.write(
        "Writing feedback files [%-50s] %d%%" % ('=' * i, i*2) + ' %d of %d' % (current, total))  # +str(time))
    sys.stdout.flush()

def progressBar2(current: Union[float, int], total: Union[float, int], startTime: dt.datetime, message: str = ""):
    """write a progress bar to stout."""
    try:
        remainingTime = (dt.datetime.utcnow() - startTime) / current * (total - current)
    except ZeroDivisionError:
        remainingTime = "inf"
    i = int(current / total * 50)
    sys.stdout.write('\r')
    sys.stdout.write(
        message + "[%-50s] %d%%" % ('=' * i, i*2) + ' %d of %d' % (current, total) +
        " time remaining: " + str(remainingTime))
    sys.stdout.flush()

def makeFileList(dir_in:Optional[str] = None,
                 files_in: Optional[str] = None,  # string with wild-card(s), i.e. "*"
                 file_list: Optional[List[str]] = None) -> List[str]:
    """Return a list of file names. """
    if not (files_in is None):
        files_in =files_in.format(**{"path": os.path.dirname(__file__)})
        file_list = glob.glob(files_in)
    elif not (dir_in is None):
        file_list = glob.glob(dir_in + "\*")
    return file_list