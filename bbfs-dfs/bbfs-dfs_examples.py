#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import argparse
import os

# 3rd party imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from pyrfu import pyrf, mms
from pyrfu.plot import plot_line, plot_spectr, zoom

# Local imports
from bbfsdfs.load import load_fields, load_fpi
from bbfsdfs.utils import fit_df

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2022"
__license__ = "Apache 2.0"

# Setup matplotlib style
plt.style.use("classic")
plt.rcParams["mathtext.sf"] = "sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"
color = ["tab:blue", "tab:green", "tab:red", "k"]
plt.rc("axes", prop_cycle=mpl.cycler(color=color))

# Setup path to MMS data
mms.db_init("/Volumes/PHD/bbfs/data")


def main(args):
    bbfs = pd.read_csv(os.path.join(args.path, "mms_bbfsdb_2017-2021.csv"), header=None)
    t_ids = [872, 798]
    t_ids = [3647, 872]
    tints = bbfs.values[t_ids]
    tints = [pyrf.extend_tint(tint, [-53, 53]) for tint in tints]

    # Load data
    fields_e_b_q, fields_e_b_t = [load_fields(tint) for tint in tints]
    moms_fpi_i_q, moms_fpi_i_t = [load_fpi(tint) for tint in tints]

    start = pyrf.iso86012datetime64(np.array(tints[0]))[0]
    end = pyrf.iso86012datetime64(np.array(tints[0]))[0]
    end += np.timedelta64(3, "m")
    tint_sel = list(pyrf.datetime642iso8601(np.array([start, end])))
    b_gsm_r = pyrf.time_clip(fields_e_b_q[0], tint_sel)
    t_df_q, b_syn_q, a_q, b_q, sigma_q = fit_df(b_gsm_r)

    start = pyrf.iso86012datetime64(np.array(tints[1]))[0]
    end = pyrf.iso86012datetime64(np.array(tints[1]))[0]
    end += np.timedelta64(3, "m")
    tint_sel = list(pyrf.datetime642iso8601(np.array([start, end])))
    b_gsm_r = pyrf.time_clip(fields_e_b_t[0], tint_sel)
    t_df_t, b_syn_t, a_t, b_t, sigma_t = fit_df(b_gsm_r)

    tint_df_q = [
        (t_df_q - np.timedelta64(30, "s")).astype(str),
        (t_df_q + np.timedelta64(15, "s")).astype(str),
    ]
    tint_df_t = [
        (t_df_t - np.timedelta64(30, "s")).astype(str),
        (t_df_t + np.timedelta64(15, "s")).astype(str),
    ]

    fig = plt.figure(figsize=(13, 12))
    gsp1 = fig.add_gridspec(
        13, 2, top=0.95, bottom=0.05, left=0.1, right=0.9, hspace=0.1, wspace=0.5
    )

    gsp10 = gsp1[:-3, 0].subgridspec(5, 1, hspace=0)
    gsp11 = gsp1[-2:, 0].subgridspec(1, 1, hspace=0)
    gsp20 = gsp1[:-3, 1].subgridspec(5, 1, hspace=0)
    gsp21 = gsp1[-2:, 1].subgridspec(1, 1, hspace=0)

    # Create axes in the grid spec
    axs10 = [fig.add_subplot(gsp10[i]) for i in range(5)]
    axs11 = [fig.add_subplot(gsp11[i]) for i in range(1)]
    axs20 = [fig.add_subplot(gsp20[i]) for i in range(5)]
    axs21 = [fig.add_subplot(gsp21[i]) for i in range(1)]

    axs10[0], caxs100 = plot_spectr(
        axs10[0], moms_fpi_i_q[3], yscale="log", cscale="log", cmap="Spectral_r"
    )
    axs10[0].set_ylim(moms_fpi_i_q[3].energy.data[[0, -1]])
    axs10[0].set_ylabel("$K_p$" + "\n" + "[eV]")
    caxs100.set_ylabel("DEF" + "\n" + "[(cm$^{2}$ s sr)$^{-1}$]")

    axs20[0], caxs200 = plot_spectr(
        axs20[0], moms_fpi_i_t[3], yscale="log", cscale="log", cmap="Spectral_r"
    )
    axs20[0].set_ylim(moms_fpi_i_t[3].energy.data[[0, -1]])
    axs20[0].set_ylabel("$K_p$" + "\n" + "[eV]")
    caxs200.set_ylabel("DEF" + "\n" + "[(cm$^{2}$ s sr)$^{-1}$]")

    plot_line(axs10[1], fields_e_b_q[0])
    plot_line(axs10[1], pyrf.norm(fields_e_b_q[0]), zorder=0)
    axs10[1].set_ylim([-11, 22])
    axs10[1].set_ylabel("$B$" + "\n" + "[nT]")
    axs10[1].legend(
        ["$B_x$", "$B_y$", "$B_z$", "$|B|$"],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        handlelength=1.5,
    )

    plot_line(axs20[1], fields_e_b_t[0])
    plot_line(axs20[1], pyrf.norm(fields_e_b_t[0]), zorder=0)
    axs20[1].set_ylim([-9, 12])
    axs20[1].set_ylabel("$B$" + "\n" + "[nT]")
    axs20[1].legend(
        ["$B_x$", "$B_y$", "$B_z$", "$|B|$"],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        handlelength=1.5,
    )

    plot_line(axs10[2], moms_fpi_i_q[0])
    axs10[2].set_ylim(bottom=0.07)
    axs10[2].set_ylabel("$n_i$" + "\n" + "[cm$^{-3}$]")

    plot_line(axs20[2], moms_fpi_i_t[0])
    axs20[2].set_ylim([0.07, 0.32])
    axs20[2].set_ylabel("$n_i$" + "\n" + "[cm$^{-3}$]")

    plot_line(axs10[3], moms_fpi_i_q[1])
    axs10[3].set_ylim([-300, 990])
    axs10[3].set_ylabel("$V_i$" + "\n" + "[km s$^{-1}$]")
    axs10[3].legend(
        ["$V_{ix}$", "$V_{iy}$", "$V_{iz}$"],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        handlelength=1.5,
    )

    plot_line(axs20[3], moms_fpi_i_t[1])
    axs20[3].set_ylim([-300, 690])
    axs20[3].set_ylabel("$V_i$" + "\n" + "[km s$^{-1}$]")
    axs20[3].legend(
        ["$V_{ix}$", "$V_{iy}$", "$V_{iz}$"],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        handlelength=1.5,
    )

    plot_line(axs10[4], moms_fpi_i_q[2] / 1e3)
    axs10[4].set_ylim([2.5, 7.5])
    axs10[4].set_ylabel("$T_i$" + "\n" + "[keV]")

    plot_line(axs20[4], moms_fpi_i_t[2] / 1e3)
    axs20[4].set_ylim([2.5, 7.5])
    axs20[4].set_ylabel("$T_i$" + "\n" + "[keV]")

    for ax1, ax2 in zip(axs10, axs20):
        ax1.axvline(t_df_q, color="tab:orange", linestyle="--")
        ax1.axvspan(
            mdates.datestr2num(tint_df_q[0]),
            mdates.datestr2num(tint_df_q[1]),
            facecolor="none",
            linestyle="--",
            edgecolor="k",
        )
        ax2.axvline(t_df_t, color="tab:orange", linestyle="--")
        ax2.axvspan(
            mdates.datestr2num(tint_df_t[0]),
            mdates.datestr2num(tint_df_t[1]),
            facecolor="none",
            linestyle="--",
            edgecolor="k",
        )

    plot_line(axs11[0], fields_e_b_q[0][:, 2], color="tab:red", label="$B_z$")
    plot_line(axs11[0], b_syn_q, color="tab:orange", label="$B_{fit}$")
    axs11[0].axvline(t_df_q, color="tab:orange", linestyle="--")
    axs11[0].set_ylabel("$B_z$" + "\n" + "[nT]")
    axs11[0].set_title(
        f"$a$={a_q:3.2f} nT, $b$={b_q:3.2f} s, $\\sigma$={sigma_q:3.2f} nT"
    )
    axs11[0].legend(
        frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), handlelength=1.5
    )

    plot_line(axs21[0], fields_e_b_t[0][:, 2], color="tab:red", label="$B_z$")
    plot_line(axs21[0], b_syn_t, color="tab:orange", label="$B_{fit}$")
    axs21[0].axvline(t_df_t, color="tab:orange", linestyle="--")
    axs21[0].set_ylabel("$B_z$" + "\n" + "[nT]")
    axs21[0].set_title(
        f"$a$={a_t:3.2f} nT, $b$={b_t:3.2f} s, $\\sigma$={sigma_t:3.2f} nT"
    )
    axs21[0].legend(
        frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), handlelength=1.5
    )

    axs10[-1].get_shared_x_axes().join(*axs10)
    axs20[-1].get_shared_x_axes().join(*axs20)

    fig.align_ylabels(axs10)
    fig.align_ylabels([*axs20, *axs21])

    for ax1, ax2 in zip(axs10[:-1], axs20[:-1]):
        ax1.xaxis.set_ticklabels([])
        ax2.xaxis.set_ticklabels([])

    axs10[-1].set_xlim(tints[0])
    axs20[-1].set_xlim(tints[1])
    axs11[0].set_xlim(tint_df_q)
    axs21[0].set_xlim(tint_df_t)
    zoom(axs11[0], axs10[-1])
    zoom(axs21[0], axs20[-1])
    from pyrfu.plot import make_labels

    make_labels(axs10, [0.016, 0.881], pad=0)
    make_labels(axs11, [0.016, 0.881], pad=5)
    make_labels(axs20, [0.016, 0.881], pad=6)
    make_labels(axs21, [0.016, 0.881], pad=11)

    fig.savefig("../figures/figure_3.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Plot 2 examples of Central Plasma "
        "Sheet fast flows with dipolarization "
        "front associated."
    )
    parser.add_argument(
        "--path", "-p", type=str, default=os.path.join(os.getcwd(), "data")
    )
    main(parser.parse_args())
