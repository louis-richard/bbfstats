#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import argparse
import os

# 3rd party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from pyrfu import pyrf

from bbfsdfs.utils import calc_pdf_err, fairfield

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2022"
__license__ = "Apache 2.0"

plt.style.use("classic")
plt.rcParams["mathtext.sf"] = "sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"


def main(args):
    # Load BBFs time intervals (.csv)
    bbfs = pd.read_csv(os.path.join(args.path, "mms_bbfsdb_2017-2021.csv"), header=None)

    # Load saved quantities (.nc)
    data = xr.load_dataset(os.path.join(args.path, "mms_bbfsdb_2017-2021.nc"))

    # Load saved spacecraft location and magnetic field for all tail seasons
    r_tm = xr.load_dataset(os.path.join(args.path, "mms_r_gsm_betai_2017-2021.nc"))
    r_sc = r_tm.r_gsm[
        np.logical_and(
            np.abs(r_tm.r_gsm.data[:, 1]) / 6371 < 12, r_tm.beta_p.data > 0.5
        )
    ]

    # Load saved spacecraft location and magnetic field for all tail seasons
    b_sc = xr.load_dataarray(os.path.join(args.path, "mms_b_gsm_2017-2021.nc"))

    # Converts locations to Re
    data.x.data /= 6371
    data.y.data /= 6371
    data.z.data /= 6371
    r_sc.data /= 6371

    # Selection criteria:
    # |Y_{GSM}| < 12 Re
    cond_0 = np.abs(data.y.data) < 12.0
    # Central Plasma Sheet: \beta_i > 0.5
    cond_1 = np.logical_and(data.beta.data > 0.5, cond_0)
    # Earthward flows
    cond_ew = np.logical_and(data.v_x.data >= 0, cond_1)
    # Tailward flows
    cond_tw = np.logical_and(data.v_x.data < 0, cond_1)

    # Compute bins edges of distance to Earth
    x_edges = np.histogram_bin_edges(data.x.data[cond_0], bins="fd")

    cnts_r_ew = pyrf.histogram(data.x.data[cond_ew], bins=x_edges, density=False)
    cnts_r_tw = pyrf.histogram(data.x.data[cond_tw], bins=x_edges, density=False)
    cnts_r_sc = pyrf.histogram(r_sc.data, bins=x_edges, density=False)
    cnts_r_sc.data = cnts_r_sc.data.astype(float)
    cnts_r_sc.data *= 4.5 / 3600

    # Compute distribution of the duration of the BBFs
    delta_t = np.squeeze(np.diff(pyrf.iso86012datetime64(bbfs.values), axis=1))
    delta_t = pyrf.ts_scalar(data.time.data, delta_t.astype(np.int64) / 60e9)
    percentiles = np.quantile(data.x.data, [0.25, 0.5, 0.75])
    cond_d0_ew = np.logical_and(delta_t < 3 * 60, cond_ew)
    # Inner magnetosphere
    cond_d25_ew = np.logical_and(cond_d0_ew, data.x.data > percentiles[2])
    # Outer magnetosphere
    cond_d50_ew = np.logical_and(
        cond_d0_ew,
        np.logical_and(data.x.data < percentiles[2], data.x.data > percentiles[1]),
    )
    # Near Earth X-line
    cond_d75_ew = np.logical_and(
        cond_d0_ew,
        np.logical_and(data.x.data < percentiles[1], data.x.data > percentiles[0]),
    )
    # Mid-tail
    cond_d100_ew = np.logical_and(cond_d0_ew, data.x.data < percentiles[0])

    means_ew = np.array(
        [
            np.mean(delta_t[cond_d100_ew].data),  # Mid-tail
            np.mean(delta_t[cond_d75_ew].data),  # NE X-line
            np.mean(delta_t[cond_d50_ew].data),  # Outer MSPH
            np.mean(delta_t[cond_d25_ew].data),  # Inner MSPH
            np.mean(delta_t[cond_d25_ew].data),
        ]
    )

    cond_d0_tw = np.logical_and(delta_t < 3 * 60, cond_tw)
    # Inner magnetosphere
    cond_d25_tw = np.logical_and(cond_d0_tw, data.x.data > percentiles[2])
    # Outer magnetosphere
    cond_d50_tw = np.logical_and(
        cond_d0_tw,
        np.logical_and(data.x.data < percentiles[2], data.x.data > percentiles[1]),
    )
    # Near Earth X-line
    cond_d75_tw = np.logical_and(
        cond_d0_tw,
        np.logical_and(data.x.data < percentiles[1], data.x.data > percentiles[0]),
    )
    # Mid-tail
    cond_d100_tw = np.logical_and(cond_d0_tw, data.x.data < percentiles[0])

    means_tw = np.array(
        [
            np.mean(delta_t[cond_d100_tw].data),  # Mid-tail
            np.mean(delta_t[cond_d75_tw].data),  # NE X-line
            np.mean(delta_t[cond_d50_tw].data),  # Outer MSPH
            np.mean(delta_t[cond_d25_tw].data),  # Inner MSPH
            np.mean(delta_t[cond_d25_tw].data),
        ]
    )

    hist_bx_ew = pyrf.histogram(data.b_x[cond_ew], bins="fd", density=True)
    cnts_bx_ew = pyrf.histogram(data.b_x[cond_ew], bins="fd", density=False)
    errs_bx_ew = calc_pdf_err(cnts_bx_ew.bins.data, cnts_bx_ew.data)

    hist_bx_tw = pyrf.histogram(data.b_x[cond_tw], bins="fd", density=True)
    cnts_bx_tw = pyrf.histogram(data.b_x[cond_tw], bins="fd", density=False)
    errs_bx_tw = calc_pdf_err(cnts_bx_tw.bins.data, cnts_bx_tw.data)

    bx_edges = np.histogram_bin_edges(data.b_x[cond_1], bins="fd")
    hist_bx_sc = pyrf.histogram(b_sc[:, 0], bins=bx_edges, density=True)

    # Compute distance to neutral sheet using Fairfield, 1980 model
    # Earthward and tailward flows in the |Y| < 12 Re CPS region
    dz_ew = fairfield(
        data.time.data[cond_ew],
        data.x.data[cond_ew] * 6371,
        data.y.data[cond_ew] * 6371,
        data.z.data[cond_ew] * 6371,
    )
    dz_tw = fairfield(
        data.time.data[cond_tw],
        data.x.data[cond_tw] * 6371,
        data.y.data[cond_tw] * 6371,
        data.z.data[cond_tw] * 6371,
    )
    dz_tt = fairfield(
        data.time.data[cond_1],
        data.x.data[cond_1] * 6371,
        data.y.data[cond_1] * 6371,
        data.z.data[cond_1] * 6371,
    )

    # MMS orbit
    dz_mms = fairfield(
        r_sc.time.data,
        r_sc.data[:, 0] * 6371,
        r_sc.data[:, 1] * 6371,
        r_sc.data[:, 2] * 6371,
    )

    # Compute PDF of distance to neutral and the associated error
    hist_dz_ew = pyrf.histogram(dz_ew[~np.isnan(dz_ew)], bins="fd", density=True)
    cnts_dz_ew = pyrf.histogram(dz_ew[~np.isnan(dz_ew)], bins="fd", density=False)
    errs_dz_ew = calc_pdf_err(cnts_dz_ew.bins.data, cnts_dz_ew.data)

    hist_dz_tw = pyrf.histogram(dz_tw[~np.isnan(dz_tw)], bins="fd", density=True)
    cnts_dz_tw = pyrf.histogram(dz_tw[~np.isnan(dz_tw)], bins="fd", density=False)
    errs_dz_tw = calc_pdf_err(cnts_dz_tw.bins.data, cnts_dz_tw.data)

    dz_edges = np.histogram_bin_edges(dz_tt[~np.isnan(dz_tt)], bins="fd")
    hist_dz_mms = pyrf.histogram(dz_mms[~np.isnan(dz_mms)], bins=dz_edges, density=True)

    f, axs = plt.subplots(2, 2, figsize=(12, 9))
    f.subplots_adjust(bottom=0.07, top=0.95, left=0.1, right=0.9, wspace=0.4)
    axs[0, 0].errorbar(
        cnts_r_sc.bins,
        cnts_r_ew / cnts_r_sc,
        np.sqrt(cnts_r_ew) / cnts_r_sc.data,
        color="tab:blue",
        drawstyle="steps-mid",
        capsize=4,
        linestyle="-",
        marker="o",
        label="Earthward BBFs",
    )
    axs[0, 0].errorbar(
        cnts_r_sc.bins,
        cnts_r_tw / cnts_r_sc,
        np.sqrt(cnts_r_tw) / cnts_r_sc.data,
        color="tab:green",
        drawstyle="steps-mid",
        capsize=4,
        linestyle="-",
        marker="o",
        label="Tailward BBFs",
    )

    axs[0, 0].set_xlabel("$X_{GSM}$ [$R_E$]")
    axs[0, 0].set_ylabel("events/hour")
    axs[0, 0].set_xlim([-29.5, -5])
    axs[0, 0].set_ylim(bottom=0)
    axs[0, 0].legend(loc="upper right", frameon=True)
    axs[0, 0].text(0.025, 0.94, "(a)", transform=axs[0, 0].transAxes)
    print(means_ew)
    axs[0, 1].step(
        np.hstack([-30.0, percentiles, 0]),
        means_ew,
        where="post",
        color="tab:blue",
        linestyle="-",
        label="Earthward BBFs",
    )
    axs[0, 1].step(
        np.hstack([-30.0, percentiles, 0]),
        means_tw,
        where="post",
        color="tab:green",
        linestyle="-",
        label="Tailward BBFs",
    )
    axs[0, 1].set_xlabel("$X_{GSM}$ [$R_E$]")
    axs[0, 1].set_ylabel("$\\langle \\Delta t\\rangle $ [min.]")
    axs[0, 1].set_xlim([-29.5, -5])
    axs[0, 1].set_ylim(top=4.3)
    axs[0, 1].legend(loc="upper right", frameon=True)
    axs[0, 1].text(0.025, 0.94, "(b)", transform=axs[0, 1].transAxes)

    axs[1, 0].errorbar(
        hist_bx_ew.bins,
        hist_bx_ew.data,
        errs_bx_ew.data,
        drawstyle="steps-mid",
        color="tab:blue",
        linestyle="-",
        label="Earthward BBFs",
        marker="o",
        capsize=4,
    )
    axs[1, 0].errorbar(
        hist_bx_tw.bins,
        hist_bx_tw.data,
        errs_bx_tw.data,
        ds="steps-mid",
        color="tab:green",
        linestyle="-",
        label="Tailward BBFs",
        marker="o",
        capsize=4,
    )

    axs[1, 0].step(
        hist_bx_sc.bins,
        hist_bx_sc.data,
        where="mid",
        color="k",
        linestyle="-",
        label="MMS orbit coverage",
    )

    axs[1, 0].legend(loc="upper right", frameon=True)
    axs[1, 0].set_ylim([0, 0.059])
    axs[1, 0].axvspan(-10, 10, color="lightgrey", alpha=0.5, ec="k", linestyle="--")
    axs[1, 0].set_ylabel("PDF")
    axs[1, 0].set_xlim([-65, 65])
    axs[1, 0].set_xlabel("$B_x$ [nT]")
    axs[1, 0].text(0.025, 0.94, "(c)", transform=axs[1, 0].transAxes)

    axs[1, 1].errorbar(
        hist_dz_ew.bins,
        hist_dz_ew.data,
        errs_dz_ew,
        color="tab:blue",
        label="Earthward BBFs",
        drawstyle="steps-mid",
        marker="o",
        capsize=4,
    )
    axs[1, 1].errorbar(
        hist_dz_tw.bins,
        hist_dz_tw.data,
        errs_dz_tw,
        color="tab:green",
        label="Tailward BBFs",
        drawstyle="steps-mid",
        marker="o",
        capsize=4,
    )
    axs[1, 1].step(
        hist_dz_mms.bins,
        hist_dz_mms.data,
        color="k",
        label="MMS orbit coverage",
        where="mid",
    )
    axs[1, 1].set_xlabel("$\\delta Z_{GSM}$ [$R_E$]")
    axs[1, 1].set_ylabel("PDF")
    axs[1, 1].set_xlim([-6.5, 6.5])
    axs[1, 1].set_ylim([0, 0.3])
    axs[1, 1].legend(loc="upper right", frameon=True)
    axs[1, 1].text(0.025, 0.94, "(d)", transform=axs[1, 1].transAxes)

    f.savefig("../figures/figure_2.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Plot the 1D projection of the "
        "distribution of Central Plasma Sheet "
        "fast flows together with the average "
        "duration of the flows with the 25th, "
        "50th, 75th and 100th percentiles."
    )
    parser.add_argument(
        "--path", "-p", type=str, default=os.path.join(os.getcwd(), "data")
    )
    main(parser.parse_args())
