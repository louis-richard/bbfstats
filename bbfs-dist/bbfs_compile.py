#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import glob
import os

# 3rd party imports
import numpy as np
import pandas as pd
import xarray as xr

from bbfsdfs.load import load_fpi
from pyrfu import mms, pyrf
from scipy import constants

# Setup MMS datapath
mms.db_init("/Volumes/PHD/bbfs/data")

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2022"
__license__ = "Apache 2.0"

db_fname = "mms_bbfsdb_2017-2021"
data_path = os.path.join(os.path.pardir, "data")
bbfs = pd.read_csv(os.path.join(data_path, f"{db_fname}.csv"), header=None)
time_line = np.asarray(bbfs.values[:, 0], dtype="<M8[ns]")
start = np.asarray(bbfs.values[:, 0], dtype="<M8[ns]")
end = np.asarray(bbfs.values[:, 1], dtype="<M8[ns]")

data_files = glob.glob(f"{data_path}/*.nc")

if os.path.join(data_path, f"{db_fname}.nc") not in data_files:
    out = {
        "beta": pyrf.ts_scalar(time_line, np.zeros(len(bbfs))),
        "x": pyrf.ts_scalar(time_line, np.zeros(len(bbfs))),
        "y": pyrf.ts_scalar(time_line, np.zeros(len(bbfs))),
        "z": pyrf.ts_scalar(time_line, np.zeros(len(bbfs))),
        "n": pyrf.ts_scalar(time_line, np.zeros(len(bbfs))),
        "t": pyrf.ts_scalar(time_line, np.zeros(len(bbfs))),
        "v_x": pyrf.ts_scalar(time_line, np.zeros(len(bbfs))),
        "v_y": pyrf.ts_scalar(time_line, np.zeros(len(bbfs))),
        "v_z": pyrf.ts_scalar(time_line, np.zeros(len(bbfs))),
        "v_t": pyrf.ts_scalar(time_line, np.zeros(len(bbfs))),
        "b_x": pyrf.ts_scalar(time_line, np.zeros(len(bbfs))),
        "b_y": pyrf.ts_scalar(time_line, np.zeros(len(bbfs))),
        "b_z": pyrf.ts_scalar(time_line, np.zeros(len(bbfs))),
        "b_t": pyrf.ts_scalar(time_line, np.zeros(len(bbfs))),
        "delta_bz": pyrf.ts_scalar(time_line, np.zeros(len(bbfs))),
        "theta_b": pyrf.ts_scalar(time_line, np.zeros(len(bbfs))),
        "d_theta": pyrf.ts_scalar(time_line, np.zeros(len(bbfs))),
    }
    out = xr.Dataset(out)
    out.to_netcdf(os.path.join(data_path, f"{db_fname}.nc"))


def main():
    duration = np.diff(pyrf.iso86012datetime64(bbfs.values), axis=1)
    duration = np.squeeze(duration).astype(np.int64) / 1e9

    for i, tint in enumerate(bbfs.values):
        data = xr.load_dataset(os.path.join(data_path, f"{db_fname}.nc"))
        print(i, tint)
        b_gsm = mms.get_data("b_gsm_fgm_srvy_l2", tint, 1)

        if duration[i] < 60:
            dt = (60 - duration[i]) / 2 + 1
            r_gsm = mms.get_data(
                "r_gsm_mec_srvy_l2", pyrf.extend_tint(tint, [-dt, dt]), 1
            )

        else:
            r_gsm = mms.get_data("r_gsm_mec_srvy_l2", tint, 1)

        if duration[i] < 3 * 60.0:
            dt = (3.0 * 60 - duration[i]) / 2 + 1
            b_dfs = mms.get_data(
                "b_gsm_fgm_srvy_l2", pyrf.extend_tint(tint, [-dt, dt]), 1
            )
        else:
            b_dfs = b_gsm

        n_i, v_gsm_i, t_i, _ = load_fpi(tint)
        p_i = n_i.data * t_i  # nPa
        p_i.data *= 1e15 * constants.elementary_charge

        p_b = 1e-18 * pyrf.norm(b_gsm) ** 2 / (2 * constants.mu_0)
        p_b = pyrf.resample(p_b, p_i, f_s=pyrf.calc_fs(p_i))

        el_ = np.arctan(
            b_dfs.data[:, 2] / np.sqrt(np.nansum(b_dfs.data[:, :2] ** 2, axis=1))
        )
        data.delta_bz.data[i] = np.nanmax(b_dfs.data[:, 2]) - np.nanmin(
            b_dfs.data[:, 2]
        )
        data.theta_b.data[i] = np.nanmax(np.abs(el_))
        data.d_theta.data[i] = (
            el_[np.nanargmax(b_dfs.data[:, 2])] - el_[np.nanargmin(b_dfs.data[:, 2])]
        )

        data.beta.data[i] = np.nanmean(1e-9 * p_i / p_b.data)
        data.n.data[i] = np.nanmean(n_i)
        data.t.data[i] = np.nanmean(1e-15 * p_i / (n_i * constants.electron_volt))

        data.x.data[i] = np.nanmean(r_gsm.data[:, 0])
        data.y.data[i] = np.nanmean(r_gsm.data[:, 1])
        data.z.data[i] = np.nanmean(r_gsm.data[:, 2])

        v_perp = np.sqrt(np.nansum(v_gsm_i.data[:, :2] ** 2, axis=1))
        v_max = v_gsm_i.data[np.nanargmax(v_perp), :]
        data.v_x.data[i] = v_max[0]
        data.v_y.data[i] = v_max[1]
        data.v_z.data[i] = v_max[2]
        data.v_t.data[i] = np.linalg.norm(v_max)

        data.b_x.data[i] = np.nanmean(b_gsm.data[:, 0])
        data.b_y.data[i] = np.nanmean(b_gsm.data[:, 1])
        data.b_z.data[i] = np.nanmean(b_gsm.data[:, 2])
        data.b_t.data[i] = np.nanmean(pyrf.norm(b_gsm))

        data.to_netcdf(os.path.join(data_path, f"{db_fname}.nc"))


if __name__ == "__main__":
    main()
