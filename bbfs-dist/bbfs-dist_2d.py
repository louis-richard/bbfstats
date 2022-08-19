#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pdb

# Built-in imports
import argparse
import os

# 3rd party imports
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from bbfsdfs.utils import gaussian, estimate_mp
from matplotlib.patches import Wedge
from pyrfu import pyrf
from pyrfu.plot import plot_spectr
from scipy import optimize

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2022"
__license__ = "Apache 2.0"

plt.style.use("classic")
plt.rcParams["mathtext.sf"] = "sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"


def _add_earth(ax=None, **kwargs):
    theta1, theta2 = 90.0, 270.0
    nightside_ = Wedge((0.0, 0.0), 1.0, theta1, theta2, fc="k", ec="k", **kwargs)
    dayside_ = Wedge((0.0, 0.0), 1.0, theta2, theta1, fc="w", ec="k", **kwargs)
    for wedge in [nightside_, dayside_]:
        ax.add_artist(wedge)
    return [nightside_, dayside_]


def main(args):
    # Load saved quantities (.nc)
    data = xr.load_dataset(os.path.join(args.path, "mms_bbfsdb_2017-2021.nc"))

    # Load saved spacecraft location and magnetic field for all tail seasons
    r_tm = xr.load_dataset(os.path.join(args.path, "mms_r_gsm_betai_2017-2021.nc"))
    r_sc = r_tm.r_gsm[
        np.logical_and(
            np.abs(r_tm.r_gsm.data[:, 1]) / 6371 < 12, r_tm.beta_p.data > 0.5
        )
    ]

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

    cond_v1 = np.logical_and(np.abs(data.v_x.data) >= 400.0, cond_1)
    cond_v2 = np.logical_and(np.abs(data.v_x.data) >= 500.0, cond_1)

    print(
        f"V_x = 300 km/s: Tot.: {np.sum(cond_1)}, "
        f"E.F.: {np.sum(cond_ew)}, T.F.: {np.sum(cond_tw)}"
    )
    print(f"V_x = 400 km/s: Tot.: {np.sum(cond_v1)}")
    print(f"V_x = 500 km/s: Tot.: {np.sum(cond_v2)}")

    x_edges = np.histogram_bin_edges(data.x.data[cond_1], bins="fd")
    y_edges = np.histogram_bin_edges(data.y.data[cond_1], bins="fd")
    z_edges = np.histogram_bin_edges(data.z.data[cond_1], bins="fd")
    print(
        f"h_x = {np.median(np.diff(x_edges)):3.2f} R_E, "
        f"h_y = {np.median(np.diff(y_edges)):3.2f} R_E, "
        f"h_z = {np.median(np.diff(z_edges)):3.2f} R_E"
    )

    # 2d histogram
    # Earthward flows within |Y GSM| < 12 Re in the CPS (\beta_i > 0.5)
    cnts_xy_ew = pyrf.histogram2d(
        data.x[cond_ew], data.y[cond_ew], bins=(x_edges, y_edges), density=False
    )
    cnts_xz_ew = pyrf.histogram2d(
        data.x[cond_ew], data.z[cond_ew], bins=(x_edges, z_edges), density=False
    )
    cnts_xy_ew.data[cnts_xy_ew.data == 0] = np.nan
    cnts_xz_ew.data[cnts_xz_ew.data == 0] = np.nan

    # Tailward flows within |Y GSM| < 12 Re in the CPS (\beta_i > 0.5)
    cnts_xy_tw = pyrf.histogram2d(
        data.x[cond_tw], data.y[cond_tw], bins=(x_edges, y_edges), density=False
    )
    cnts_xz_tw = pyrf.histogram2d(
        data.x[cond_tw], data.z[cond_tw], bins=(x_edges, z_edges), density=False
    )
    cnts_xy_tw.data[cnts_xy_tw.data == 0] = np.nan
    cnts_xz_tw.data[cnts_xz_tw.data == 0] = np.nan

    # MMS orbit
    cnts_xy_sc = pyrf.histogram2d(
        r_sc[:, 0], r_sc[:, 1], bins=(x_edges, y_edges), density=False
    )
    cnts_xz_sc = pyrf.histogram2d(
        r_sc[:, 0], r_sc[:, 2], bins=(x_edges, z_edges), density=False
    )
    cnts_xy_sc.data[cnts_xy_sc.data == 0] = np.nan
    cnts_xy_sc.data *= 4.5 / 3600
    cnts_xz_sc.data[cnts_xz_sc.data == 0] = np.nan
    cnts_xz_sc.data *= 4.5 / 3600

    # Computes counts per hour of coverage
    ctph_xy_ew = cnts_xy_ew / cnts_xy_sc.data
    ctph_xz_ew = cnts_xz_ew / cnts_xz_sc.data

    ctph_xy_tw = cnts_xy_tw / cnts_xy_sc.data
    ctph_xz_tw = cnts_xz_tw / cnts_xz_sc.data

    # 1d histogram
    # Earthward flows within |Y GSM| < 12 Re

    cnts_y_ew = pyrf.histogram(data.y[cond_ew], bins=y_edges, density=False)
    cnts_z_ew = pyrf.histogram(data.z[cond_ew], bins=z_edges, density=False)

    cnts_y_tw = pyrf.histogram(data.y[cond_tw], bins=y_edges, density=False)
    cnts_z_tw = pyrf.histogram(data.z[cond_tw], bins=z_edges, density=False)

    cnts_y_sc = pyrf.histogram(r_sc[:, 1], bins=y_edges, density=False)
    cnts_z_sc = pyrf.histogram(r_sc[:, 2], bins=z_edges, density=False)

    cnts_y_sc.data = cnts_y_sc.data.astype(float)
    cnts_z_sc.data = cnts_z_sc.data.astype(float)

    cnts_y_sc.data *= 4.5 / 3600
    cnts_z_sc.data *= 4.5 / 3600

    # Compute counts per hour of coverage
    ctph_y_ew = cnts_y_ew / cnts_y_sc.data
    ctph_z_ew = cnts_z_ew / cnts_z_sc.data

    ctph_y_tw = cnts_y_tw / cnts_y_sc.data
    ctph_z_tw = cnts_z_tw / cnts_z_sc.data

    sigma_y = np.sqrt(cnts_y_ew.data) / cnts_y_sc.data
    fit_y_ew, cov_y_ew = optimize.curve_fit(
        gaussian,
        cnts_y_sc.bins.data[ctph_y_ew.data != 0],
        ctph_y_ew.data[ctph_y_ew.data != 0],
        sigma=sigma_y[ctph_y_ew.data != 0],
    )
    sigma_z = np.sqrt(cnts_z_ew.data) / cnts_z_sc.data
    fit_z_ew, cov_z_ew = optimize.curve_fit(
        gaussian,
        cnts_z_sc.bins.data[ctph_z_ew.data != 0],
        ctph_z_ew.data[ctph_z_ew.data != 0],
        sigma=sigma_z[ctph_z_ew.data != 0],
    )

    sigma_y = np.sqrt(cnts_y_tw.data) / cnts_y_sc.data
    fit_y_tw, cov_y_tw = optimize.curve_fit(
        gaussian,
        cnts_y_sc.bins.data[ctph_y_tw.data != 0],
        ctph_y_tw.data[ctph_y_tw.data != 0],
        sigma=sigma_y[ctph_y_tw.data != 0],
    )
    sigma_z = np.sqrt(cnts_z_tw.data) / cnts_z_sc.data
    fit_z_tw, cov_z_tw = optimize.curve_fit(
        gaussian,
        cnts_z_sc.bins.data[ctph_z_tw.data != 0],
        ctph_z_tw.data[ctph_z_tw.data != 0],
        sigma=sigma_z[ctph_z_tw.data != 0],
    )

    print(
        f"E.F. : <Y> = "
        f"{fit_y_ew[1]:<4.2f}\\pm {np.sqrt(np.diag(cov_y_ew))[1]:<4.2f} "
        f"R_E, <Z> = "
        f"{fit_z_ew[1]:<4.2f}\\pm {np.sqrt(np.diag(cov_z_ew))[1]:<4.2f} R_E"
    )
    print(
        f"E.F. : std Y = "
        f"{fit_y_ew[2]:<4.2f}\\pm {np.sqrt(np.diag(cov_y_ew))[2]:<4.2f} "
        f"R_E, std Z = "
        f"{fit_z_ew[2]:<4.2f}\\pm {np.sqrt(np.diag(cov_z_ew))[2]:<4.2f} R_E"
    )

    print(
        f"T.F. : <Y> = "
        f"{fit_y_tw[1]:<4.2f}\\pm {np.sqrt(np.diag(cov_y_tw))[1]:<4.2f} "
        f"R_E, <Z> = "
        f"{fit_z_tw[1]:<4.2f}\\pm {np.sqrt(np.diag(cov_z_tw))[1]:<4.2f} R_E"
    )
    print(
        f"T.F. : std Y = "
        f"{fit_y_tw[2]:<4.2f}\\pm {np.sqrt(np.diag(cov_y_tw))[2]:<4.2f} "
        f"R_E, std Z = "
        f"{fit_z_tw[2]:<4.2f}\\pm {np.sqrt(np.diag(cov_z_tw))[2]:<4.2f} R_E"
    )

    mp_bounds = estimate_mp(np.unique(r_sc.time.data.astype("<M8[D]")))

    f, axs = plt.subplots(3, 2, figsize=(12, 9))
    f.subplots_adjust(top=0.92, bottom=0.07, wspace=0.3, hspace=0.4)
    axs[0, 0].plot(mp_bounds[0][0, :], mp_bounds[0][1, :], "k--")
    axs[0, 0].plot(mp_bounds[1][0, :], mp_bounds[1][1, :], "k-")
    axs[0, 0].plot(mp_bounds[2][0, :], mp_bounds[2][1, :], "k--")
    _add_earth(axs[0, 0])
    axs[0, 0], caxs00 = plot_spectr(axs[0, 0], cnts_xy_sc, cmap="Greys")
    axs[0, 0].invert_xaxis()
    axs[0, 0].set_xlim([-29.5, 15])
    axs[0, 0].set_xticks([-20, -10, 0, 10])
    axs[0, 0].set_ylim([-20, 20])
    axs[0, 0].set_aspect("equal")
    axs[0, 0].set_xlabel("$X_{GSM}$ [$R_E$]")
    axs[0, 0].set_yticklabels([])
    caxs00.set_axisbelow(False)
    axs[0, 0].set_title("MMS orbital coverage")
    axs[0, 0].text(0.02, 0.91, "(b)", transform=axs[0, 0].transAxes)

    axs[1, 0].plot(mp_bounds[0][0, :], mp_bounds[0][1, :], "k--")
    axs[1, 0].plot(mp_bounds[1][0, :], mp_bounds[1][1, :], "k-")
    axs[1, 0].plot(mp_bounds[2][0, :], mp_bounds[2][1, :], "k--")
    _add_earth(axs[1, 0])
    axs[1, 0], caxs10 = plot_spectr(axs[1, 0], ctph_xy_ew, cmap="Blues", clim=[0, 5])
    axs[1, 0].invert_xaxis()
    axs[1, 0].set_xlim([-29.5, 15])
    axs[1, 0].set_xticks([-20, -10, 0, 10])
    axs[1, 0].set_ylim([-20, 20])
    axs[1, 0].set_aspect("equal")
    axs[1, 0].set_xlabel("$X_{GSM}$ [$R_E$]")
    axs[1, 0].set_yticklabels([])
    caxs10.set_axisbelow(False)
    axs[1, 0].set_title("Earthward BBFs")
    axs[1, 0].text(0.02, 0.91, "(f)", transform=axs[1, 0].transAxes)

    axs[2, 0].plot(mp_bounds[0][0, :], mp_bounds[0][1, :], "k--")
    axs[2, 0].plot(mp_bounds[1][0, :], mp_bounds[1][1, :], "k-")
    axs[2, 0].plot(mp_bounds[2][0, :], mp_bounds[2][1, :], "k--")
    _add_earth(axs[2, 0])
    axs[2, 0], caxs20 = plot_spectr(axs[2, 0], ctph_xy_tw, cmap="Greens", clim=[0, 1.5])
    axs[2, 0].invert_xaxis()
    axs[2, 0].set_xlim([-29.5, 15])
    axs[2, 0].set_xticks([-20, -10, 0, 10])
    axs[2, 0].set_ylim([-20, 20])
    axs[2, 0].set_aspect("equal")
    axs[2, 0].set_xlabel("$X_{GSM}$ [$R_E$]")
    axs[2, 0].set_yticklabels([])
    caxs20.set_axisbelow(False)
    axs[2, 0].set_title("Tailward BBFs")
    axs[2, 0].text(0.02, 0.91, "(j)", transform=axs[2, 0].transAxes)

    axs[0, 1].plot(mp_bounds[0][0, :], mp_bounds[0][1, :], "k--")
    axs[0, 1].plot(mp_bounds[1][0, :], mp_bounds[1][1, :], "k-")
    axs[0, 1].plot(mp_bounds[2][0, :], mp_bounds[2][1, :], "k--")
    _add_earth(axs[0, 1])
    axs[0, 1], caxs01 = plot_spectr(axs[0, 1], cnts_xz_sc, cmap="Greys")
    axs[0, 1].invert_xaxis()
    axs[0, 1].set_xlim([-29.5, 15])
    axs[0, 1].set_xticks([-20, -10, 0, 10])
    axs[0, 1].set_ylim([-20, 20])
    axs[0, 1].set_aspect("equal")
    axs[0, 1].set_xlabel("$X_{GSM}$ [$R_E$]")
    axs[0, 1].set_yticklabels([])
    caxs01.set_axisbelow(False)
    axs[0, 1].set_title("MMS orbital coverage")
    axs[0, 1].text(0.02, 0.91, "(d)", transform=axs[0, 1].transAxes)

    axs[1, 1].plot(mp_bounds[0][0, :], mp_bounds[0][1, :], "k--")
    axs[1, 1].plot(mp_bounds[1][0, :], mp_bounds[1][1, :], "k-")
    axs[1, 1].plot(mp_bounds[2][0, :], mp_bounds[2][1, :], "k--")
    _add_earth(axs[1, 1])
    axs[1, 1], caxs11 = plot_spectr(axs[1, 1], ctph_xz_ew, cmap="Blues", clim=[0, 5])
    axs[1, 1].invert_xaxis()
    axs[1, 1].set_xlim([-29.5, 15])
    axs[1, 1].set_xticks([-20, -10, 0, 10])
    axs[1, 1].set_ylim([-20, 20])
    axs[1, 1].set_aspect("equal")
    axs[1, 1].set_xlabel("$X_{GSM}$ [$R_E$]")
    axs[1, 1].set_yticklabels([])
    caxs11.set_axisbelow(False)
    axs[1, 1].set_title("Earthward BBFs")
    axs[1, 1].text(0.02, 0.91, "(h)", transform=axs[1, 1].transAxes)

    axs[2, 1].plot(mp_bounds[0][0, :], mp_bounds[0][1, :], "k--")
    axs[2, 1].plot(mp_bounds[1][0, :], mp_bounds[1][1, :], "k-")
    axs[2, 1].plot(mp_bounds[2][0, :], mp_bounds[2][1, :], "k--")
    _add_earth(axs[2, 1])
    axs[2, 1], caxs21 = plot_spectr(axs[2, 1], ctph_xz_tw, cmap="Greens", clim=[0, 1.5])
    axs[2, 1].invert_xaxis()
    axs[2, 1].set_xlim([-29.5, 15])
    axs[2, 1].set_xticks([-20, -10, 0, 10])
    axs[2, 1].set_ylim([-20, 20])
    axs[2, 1].set_aspect("equal")
    axs[2, 1].set_xlabel("$X_{GSM}$ [$R_E$]")
    axs[2, 1].set_yticklabels([])
    caxs21.set_axisbelow(False)
    axs[2, 1].set_title("Tailward BBFs")
    axs[2, 1].text(0.02, 0.91, "(l)", transform=axs[2, 1].transAxes)

    pos00 = axs[0, 0].get_position()
    caxs00.set_position([pos00.x0 + pos00.width + 0.01, pos00.y0, 0.015, pos00.height])
    axs00b = f.add_axes([pos00.x0 - 0.1, pos00.y0, 0.1, pos00.height])
    axs00b.plot(cnts_y_sc.data, cnts_y_sc.bins, color="k")
    axs00b.set_xticks([0, 100, 200])
    axs00b.set_xlim([0, 250])
    axs00b.set_ylim([-20, 20])
    axs00b.set_xlabel("hours")
    axs00b.set_ylabel("$Y_{GSM}$ [$R_E$]")
    caxs00.set_ylabel("hours")
    axs00b.invert_xaxis()
    axs00b.text(0.03, 0.91, "(a)", transform=axs00b.transAxes)

    pos01 = axs[0, 1].get_position()
    caxs01.set_position([pos01.x0 + pos01.width + 0.01, pos01.y0, 0.015, pos01.height])
    axs01b = f.add_axes([pos01.x0 - 0.1, pos01.y0, 0.1, pos01.height])
    axs01b.plot(cnts_z_sc.data, cnts_z_sc.bins, color="k")
    axs01b.set_xticks([0, 100, 200])
    axs01b.set_xlim(list(axs00b.get_xlim()))
    axs01b.set_ylim([-20, 20])
    axs01b.set_xlabel("hours")
    axs01b.set_ylabel("$Z_{GSM}$ [$R_E$]")
    caxs01.set_ylabel("hours")
    # axs01b.invert_xaxis()
    axs01b.text(0.03, 0.91, "(c)", transform=axs01b.transAxes)

    pos10 = axs[1, 0].get_position()
    caxs10.set_position([pos10.x0 + pos10.width + 0.01, pos10.y0, 0.015, pos10.height])
    axs10b = f.add_axes([pos10.x0 - 0.1, pos10.y0, 0.1, pos10.height])
    axs10b.errorbar(
        ctph_y_ew,
        ctph_y_ew.bins,
        xerr=np.sqrt(cnts_y_ew.data) / cnts_y_sc.data,
        color="tab:blue",
        capsize=3,
    )

    axs10b.plot(
        gaussian(np.linspace(-20, 20, 100), fit_y_ew[0], fit_y_ew[1], fit_y_ew[2]),
        np.linspace(-20, 20, 100),
        linestyle="--",
        color="k",
        zorder=3,
    )

    axs10b.set_xticks([0.0, 0.5, 1.0, 1.5])
    axs10b.set_xlim([0, 1.7])
    axs10b.set_ylim([-20, 20])
    axs10b.set_xlabel("events/hour")
    axs10b.set_ylabel("$Y_{GSM}$ [$R_E$]")
    caxs10.set_ylabel("events/hour")
    axs10b.invert_xaxis()
    axs10b.text(0.03, 0.91, "(e)", transform=axs10b.transAxes)

    axs10b.text(
        0.025,
        0.07,
        f"$\\mu={fit_y_ew[1]:3.2f}\\pm" f" {np.sqrt(np.diag(cov_y_ew))[1]:3.2f}~R_E$",
        transform=axs10b.transAxes,
        fontsize=8,
    )
    pos11 = axs[1, 1].get_position()
    caxs11.set_position([pos11.x0 + pos11.width + 0.01, pos11.y0, 0.015, pos11.height])
    axs11b = f.add_axes([pos11.x0 - 0.1, pos11.y0, 0.1, pos11.height])
    axs11b.errorbar(
        ctph_z_ew,
        ctph_z_ew.bins,
        xerr=np.sqrt(cnts_z_ew.data) / cnts_z_sc.data,
        color="tab:blue",
        capsize=3,
    )
    axs11b.plot(
        gaussian(np.linspace(-20, 20, 100), fit_z_ew[0], fit_z_ew[1], fit_z_ew[2]),
        np.linspace(-20, 20, 100),
        linestyle="--",
        color="k",
        zorder=3,
    )
    axs11b.set_xticks([0.0, 0.5, 1.0, 1.5])
    axs11b.set_xlim(list(axs10b.get_xlim()))
    axs11b.set_ylim([-20, 20])
    axs11b.set_xlabel("events/hour")
    axs11b.set_ylabel("$Z_{GSM}$ [$R_E$]")
    caxs11.set_ylabel("events/hour")
    axs11b.text(0.03, 0.91, "(g)", transform=axs11b.transAxes)

    axs11b.text(
        0.025,
        0.07,
        f"$\\mu={fit_z_ew[1]:3.2f}\\pm" f" {np.sqrt(np.diag(cov_z_ew))[1]:3.2f}~R_E$",
        transform=axs11b.transAxes,
        fontsize=8,
    )

    pos20 = axs[2, 0].get_position()
    caxs20.set_position([pos20.x0 + pos20.width + 0.01, pos20.y0, 0.015, pos20.height])
    axs20b = f.add_axes([pos20.x0 - 0.1, pos20.y0, 0.1, pos20.height])
    axs20b.errorbar(
        ctph_y_tw,
        ctph_y_tw.bins,
        xerr=np.sqrt(cnts_y_tw.data) / cnts_y_sc.data,
        color="tab:green",
        capsize=3,
    )

    axs20b.plot(
        gaussian(np.linspace(-20, 20, 100), fit_y_tw[0], fit_y_tw[1], fit_y_tw[2]),
        np.linspace(-20, 20, 100),
        linestyle="--",
        color="k",
    )

    axs20b.set_xticks([0, 0.1, 0.2])
    axs20b.set_xlim([0, 0.27])
    axs20b.set_ylim([-20, 20])
    axs20b.set_xlabel("events/hour")
    axs20b.set_ylabel("$Y_{GSM}$ [$R_E$]")
    caxs20.set_ylabel("events/hour")
    axs20b.invert_xaxis()
    axs20b.text(0.03, 0.91, "(i)", transform=axs20b.transAxes)

    axs20b.text(
        0.025,
        0.07,
        f"$\\mu={fit_y_tw[1]:3.2f}\\pm" f" {np.sqrt(np.diag(cov_y_tw))[1]:3.2f}~R_E$",
        transform=axs20b.transAxes,
        fontsize=8,
    )

    pos21 = axs[2, 1].get_position()
    caxs21.set_position([pos21.x0 + pos21.width + 0.01, pos21.y0, 0.015, pos21.height])
    axs21b = f.add_axes([pos21.x0 - 0.1, pos21.y0, 0.1, pos21.height])
    axs21b.errorbar(
        ctph_z_tw,
        ctph_z_tw.bins,
        xerr=np.sqrt(cnts_z_tw.data) / cnts_z_sc.data,
        color="tab:green",
        capsize=3,
    )

    axs21b.plot(
        gaussian(np.linspace(-20, 20, 100), fit_z_tw[0], fit_z_tw[1], fit_z_tw[2]),
        np.linspace(-20, 20, 100),
        linestyle="--",
        color="k",
    )

    axs21b.set_xticks([0, 0.1, 0.2])
    axs21b.set_xlim(list(axs20b.get_xlim()))
    axs21b.set_ylim([-20, 20])
    axs21b.set_xlabel("events/hour")
    axs21b.set_ylabel("$Z_{GSM}$ [$R_E$]")
    caxs21.set_ylabel("events/hour")
    axs21b.text(0.03, 0.91, "(k)", transform=axs21b.transAxes)

    axs21b.text(
        0.025,
        0.07,
        f"$\\mu={fit_z_tw[1]:3.2f}\\pm" f" {np.sqrt(np.diag(cov_z_tw))[1]:3.2f}~R_E$",
        transform=axs21b.transAxes,
        fontsize=8,
    )

    f.suptitle(f"2017 - 2021 ({np.sum(cond_1):d} events)")

    f.savefig("../figures/figure_1.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Plot the projection of the distribution "
        "of Central Plasma Sheet fast flows onto "
        "the equatorial and meridional plane, "
        "and their 1D dawn-dusk and North-South "
        "projections"
    )
    parser.add_argument(
        "--path", "-p", type=str, default=os.path.join(os.getcwd(), "data")
    )
    main(parser.parse_args())
