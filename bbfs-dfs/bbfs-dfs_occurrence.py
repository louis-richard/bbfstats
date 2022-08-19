#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import argparse
import os
import glob

# 3rd party imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from pyrfu import pyrf

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2022"
__license__ = "Apache 2.0"

# Setup matplotlib plotting style
plt.style.use("classic")
plt.rcParams["mathtext.sf"] = "sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"
color = ["tab:blue", "tab:green", "tab:red", "k"]
plt.rc("axes", prop_cycle=mpl.cycler(color=color))


def main(args):
    # Load saved quantities (.nc)
    data = xr.load_dataset(os.path.join(args.path, "mms_bbfsdb_2017-2021.nc"))
    bbfs = pd.read_csv(os.path.join(args.path, "mms_bbfsdb_2017-2021.csv"), header=None)

    # Selection criteria:
    # |Y_{GSM}| < 12 Re
    cond_0 = np.abs(data.y.data) < 12 * 6371.0
    # CPS (beta_i > 0.5)
    cond_1 = np.logical_and(data.beta.data > 0.5, cond_0)

    files_path = os.path.join(args.path, "bbfs-dfs_search")
    files = glob.glob(os.path.join(files_path, "*.npy"))
    schmidt, fu = [], []

    for i in range(len(bbfs)):
        f_name = f"{pyrf.date_str([t_[:-3] for t_ in bbfs.values[i]], 3)}.npy"

        if os.path.join(files_path, f_name) in files:
            tmp = np.load(os.path.join(files_path, f_name), allow_pickle=True)
            cond_schimdt = np.logical_and(
                np.logical_and(tmp[:, 0] > 4, tmp[:, 1] > 11), tmp[:, 2] > 45
            )
            cond_fu = np.logical_and(
                np.logical_and(np.abs(tmp[:, 5]) > 4, tmp[:, 5] * tmp[:, 6] > 0),
                np.logical_and(
                    np.abs(tmp[:, 6]) < 8, np.abs(tmp[:, 7]) < 0.5 * np.abs(tmp[:, 5])
                ),
            )
            t = np.where(tmp[:, 3].astype(np.int64) < tmp[0, 4].astype(np.int64) + int(3 * 60e9))
            t = t[0]

            if len(t) >= 1:
                idx_jf = t[-1] + 1
                idx_jf = np.min([idx_jf + 1, len(tmp)])
                schmidt.append(cond_schimdt[:idx_jf].any())
                fu.append(np.logical_and(cond_schimdt[:idx_jf], cond_fu[:idx_jf]).any())
            else:
                schmidt.append(False)
                fu.append(False)
        else:
            schmidt.append(False)
            fu.append(False)

    cond_f = np.logical_and(schmidt, fu)
    cond_t = np.logical_and(schmidt, np.logical_not(fu))
    cond_q = np.logical_and(cond_1, np.logical_not(schmidt))
    cond_ew_cps = np.logical_and(data.v_x.data >= 0, cond_1)
    cond_tw_cps = np.logical_and(data.v_x.data < 0, cond_1)

    n_qf_ew = np.sum(np.logical_and(cond_q, cond_ew_cps))  # Quiet Earthward BBFs JFs
    n_qf_tw = np.sum(np.logical_and(cond_q, cond_tw_cps))  # Quiet tailward BBFs JFs
    n_df_ew = np.sum(np.logical_and(cond_f, cond_ew_cps))  # Earthward BBFs with DFs
    n_df_tw = np.sum(np.logical_and(cond_f, cond_tw_cps))  # Tailward BBFs with DFs
    n_tf_ew = np.sum(np.logical_and(cond_t, cond_ew_cps))  # Turbulent Earthward BBFs
    n_tf_tw = np.sum(np.logical_and(cond_t, cond_tw_cps))  # Turbulent tailward BBFs

    qfs_cells = [
        f"{n_qf_ew:<4d} ({n_qf_ew / np.sum(cond_ew_cps):>.0%})",
        f"{n_qf_tw:<4d} ({n_qf_tw / np.sum(cond_tw_cps):>.0%})",
        f"{np.sum(cond_q):<4d} ({np.sum(cond_q) / np.sum(cond_1):>.0%})",
    ]
    dfs_cells = [
        f"{n_df_ew:<4d} ({n_df_ew / np.sum(cond_ew_cps):>.0%})",
        f"{n_df_tw:<4d} ({n_df_tw / np.sum(cond_tw_cps):>.0%})",
        f"{np.sum(cond_f):<4d} ({np.sum(cond_f) / np.sum(cond_1):>.0%})",
    ]
    tfs_cells = [
        f"{n_tf_ew:<4d} ({n_tf_ew / np.sum(cond_ew_cps):>.0%})",
        f"{n_tf_tw:<4d} ({n_tf_tw / np.sum(cond_tw_cps):>.0%})",
        f"{np.sum(cond_t):<4d} ({np.sum(cond_t) / np.sum(cond_1):>.0%})",
    ]
    tot_cells = [
        f"{np.sum(cond_ew_cps):<4d} (100%)",
        f"{np.sum(cond_tw_cps):<4d} (100%)",
        f"{np.sum(cond_1):<4d} (100%)",
    ]

    cell_text = [qfs_cells, dfs_cells, tfs_cells, tot_cells]
    rows = (
        "Quiet JFs $\\overline{\\mathrm{S11}}$",
        "DFs S11$\\cap$ F12",
        "Turbulent JFs S11 - F12",
        "Total",
    )
    columns = ("Earthward BBFs", "Tailward BBFs", "All BBFs")
    fig, ax = plt.subplots(1, figsize=(8, 5))
    fig.patch.set_visible(False)
    ax.axis("off")
    ax.axis("tight")

    ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc="center")

    fig.tight_layout()
    fig.savefig("../figures/figure_4.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Plot the occurrence rate of BBFs with "
        "DFs, turbulent BBFs and quiet BBFs for "
        "Tailward and Earthward BBFs."
    )
    parser.add_argument(
        "--path", "-p", type=str, default=os.path.join(os.getcwd(), "data")
    )
    main(parser.parse_args())
