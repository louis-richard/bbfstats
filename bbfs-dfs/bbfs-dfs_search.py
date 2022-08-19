#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pdb

# 3rd party imports
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from pyrfu import pyrf, mms
from pyrfu.plot import plot_spectr, plot_line, zoom

# Local imports
from bbfsdfs.load import load_fpi
from bbfsdfs.utils import fit_df

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2022"
__license__ = "Apache 2.0"


def _generate_tints(start, end):
    r"""Construct series of 3 minutes time intervals that span the input time
    interval.

    Parameters
    ----------
    tint : list
        Time interval to span.

    Returns
    -------
    tints : list
        List of 3 minutes time intervals that span the input time interval.

    """

    # Unpack start and end of the time interval
    start, end = [t_.astype("<M8[ns]").astype(np.int64) for t_ in [start, end]]

    # Estimate the number of time intervals required to span
    n_tints = (end - start) // 90e9

    # Create starts of the time intervals (from left to right)
    starts_l = start + np.arange(n_tints) * 90e9

    # Create starts of the time intervals (from right to left)
    starts_r = end - np.flip(np.arange(n_tints) * 90e9) - 180e9

    # For the left side of the time interval use time intervals created from
    # left to right and for the right side use time intervals created from
    # right to left.
    starts = np.hstack([starts_l[: int(n_tints // 2)], starts_r[int(n_tints // 2) :]])

    # Create time interval (starts, starts + 3 minutes)
    starts = starts.astype("<M8[ns]")
    ends = starts + np.timedelta64(3, "m")
    tints = [list(t_) for t_ in list(np.stack([starts, ends]).T.astype(str))]

    return tints


def main():
    data = xr.load_dataset("./data/mms_bbfsdb_2017-2021.nc")
    bbfs = pd.read_csv("./data/mms_bbfsdb_2017-2021.csv", header=None)
    duration = np.diff(pyrf.iso86012datetime64(bbfs.values), axis=1)
    duration = np.squeeze(duration).astype(np.int64) / 1e9
    dur_plus = (3.0 * 60 - duration) / 2 + 1
    dur_plus[dur_plus < 0] = 0

    for j in range(3550, len(bbfs.values)):
        print(
            f"it: {j:d}, time interval: {list(bbfs.values[int(j), :])[0]}"
            + f"-> {list(bbfs.values[int(j), :])[1]}"
        )
        if data.beta.data[j] < 0.5 or np.abs(data.y.data[j] / 6371) > 12:
            # results.append([False, np.nan])
            continue

        tint = pyrf.extend_tint(
            list(bbfs.values[int(j), :]), [-dur_plus[int(j)], dur_plus[int(j)]]
        )
        b_gsm = mms.get_data("b_gsm_fgm_srvy_l2", tint, 1)
        e_gsm = mms.get_data("e_gse_edp_fast_l2", tint, 1)
        n_i, v_gsm_i, t_i, def_i = load_fpi(tint)
        idx_max = np.min(
            [np.nanargmax(np.abs(v_gsm_i.data[:, 0])) + 1, len(v_gsm_i.time.data) - 1]
        )
        end = v_gsm_i.time.data[idx_max]
        tints = _generate_tints(tint[0], tint[1])

        n_e = mms.get_data("ne_fpi_fast_l2", tint, 1)
        v_gse_e = mms.get_data("ve_gse_fpi_fast_l2", tint, 1)
        v_gsm_e = pyrf.cotrans(v_gse_e, "gse>gsm")
        t_gse_e = mms.get_data("te_gse_fpi_fast_l2", tint, 1)
        t_fac_e = mms.rotate_tensor(
            t_gse_e, "fac", pyrf.cotrans(b_gsm, "gsm>gse"), "pp"
        )
        t_e = pyrf.trace(t_fac_e) / 3
        def_e = mms.get_data("defe_fpi_fast_l2", tint, 1)

        dt_max = np.max(np.diff(n_e.time.data).astype(np.int64) / 1e9)

        if dt_max > 2 * pyrf.calc_dt(n_e):
            continue

        any_df = []
        ts_dfs = []
        res = []
        for i in range(len(tints)):
            b_gsm_r = pyrf.time_clip(b_gsm, tints[i])

            # Unpack magnetic field components
            b_x, b_y, b_z = [b_gsm_r.data[:, i] for i in range(3)]

            # Compute quantities used in Schmid et al. 2011 criteria.
            # Magnetic field jump (\Delta B_z > 4 nT)
            delta_bz = np.max(b_z) - np.min(b_z)

            # Elevation angle max(\theta) > 45 deg. and \Delta \theta > 11 deg.
            theta = np.arctan(
                b_gsm_r.data[:, 2] / np.linalg.norm(b_gsm_r.data[:, :2], axis=1)
            )
            theta = np.rad2deg(theta)
            delta_theta = theta[np.argmax(b_z)] - theta[np.argmin(b_z)]
            theta_max = np.max(theta)

            # Check if fulfill Schmid et al. 2011 criteria
            criteria_schmid = delta_bz > 4 and delta_theta > 11 and theta_max > 45

            t_df, b_syn, a, b, sigma = fit_df(b_gsm_r)

            # Create time interval from 1min before the DF to 15 after.
            tint_df = [
                (t_df - np.timedelta64(30, "s")).astype(str),
                (t_df + np.timedelta64(15, "s")).astype(str),
            ]

            is_df = (
                a * b > 0
                and np.abs(a) > 4
                and np.abs(b) < 8
                and np.abs(sigma) < 0.5 * np.abs(a)
            )
            res.append([delta_bz, delta_theta, theta_max, t_df, end, a, b, sigma])
            if is_df and criteria_schmid:
                any_df.append(True)
                ts_dfs.append(t_df)
            elif criteria_schmid:
                any_df.append(True)
                ts_dfs.append(np.nan)
            else:
                any_df.append(False)
                ts_dfs.append(np.nan)

            fig = plt.figure(figsize=(12, 17.2))
            gsp1 = fig.add_gridspec(
                19, 1, top=0.95, bottom=0.05, left=0.1, right=0.9, hspace=0.1
            )

            gsp10 = gsp1[:16].subgridspec(8, 1, hspace=0)
            gsp11 = gsp1[17:].subgridspec(1, 1, hspace=0)

            # Create axes in the grid spec
            axs10 = [fig.add_subplot(gsp10[i]) for i in range(8)]
            axs11 = [fig.add_subplot(gsp11[i]) for i in range(1)]

            plot_line(axs10[0], b_gsm)
            axs10[0].legend(
                ["$B_x$", "$B_y$", "$B_z$"],
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
            )
            axs10[0].set_ylabel("$B$" + "\n" + "[nT]")

            plot_line(axs10[1], e_gsm)
            axs10[1].legend(
                ["$E_x$", "$E_y$", "$E_z$"],
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
            )
            axs10[1].set_ylabel("$E$" + "\n" + "[mV m$^{-1}$]")

            plot_line(axs10[2], v_gsm_i)
            axs10[2].legend(
                ["$V_{ix}$", "$V_{iy}$", "$V_{iz}$"],
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
            )
            axs10[2].set_ylabel("$V_i$" + "\n" + "[km s$^{-1}$]")

            plot_line(axs10[3], v_gsm_e)
            axs10[3].legend(
                ["$V_{ex}$", "$V_{ey}$", "$V_{ez}$"],
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
            )
            axs10[3].set_ylabel("$V_e$" + "\n" + "[km s$^{-1}$]")

            plot_line(axs10[4], n_i)
            plot_line(axs10[4], n_e)
            axs10[4].legend(
                ["$n_{i}$", "$n_{e}$"],
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
            )
            axs10[4].set_ylabel("$n$" + "\n" + "[cm$^{-3}$]")

            plot_line(axs10[5], t_i)
            plot_line(axs10[5], t_e)
            axs10[5].legend(
                ["$T_{i}$", "$T_{e}$"],
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
            )
            axs10[5].set_ylabel("$T$" + "\n" + "[cm$^{-3}$]")

            axs10[6], caxs106 = plot_spectr(axs10[6], def_i, yscale="log", cscale="log")
            axs10[6].set_ylabel("$K_p$" + "\n" + "eV")
            caxs106.set_ylabel("DEF" + "\n" + "[(cm$^{2}$ s sr)$^{-1}$]")
            axs10[7], caxs107 = plot_spectr(axs10[7], def_e, yscale="log", cscale="log")
            axs10[7].set_ylabel("$K_e$" + "\n" + "eV")
            caxs107.set_ylabel("DEF" + "\n" + "[(cm$^{2}$ s sr)$^{-1}$]")

            for ax in axs10:
                ax.axvline(t_df)

            axs10[-1].get_shared_x_axes().join(*axs10)

            fig.align_ylabels(axs10)

            for ax in axs10[:-1]:
                ax.xaxis.set_ticklabels([])

            axs10[-1].set_xlim(mdates.datestr2num(tint))

            plot_line(axs11[0], b_gsm[:, 2])
            plot_line(axs11[0], b_syn)
            axs11[0].legend(
                ["$B_z$", "$B_{fit}$"],
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
            )
            axs11[0].set_ylabel("$B_z$" + "\n" + "[nT]")
            axs11[0].set_xlim(tint_df)
            zoom(axs11[0], axs10[-1])

            fig.suptitle(
                f"$\\Delta B_z=${delta_bz:3.2f} nT,"
                f"$\\Delta \\theta_{{max}}=${delta_theta:3.2f}"
                f"$^{{\\circ}}$"
                f"$\\theta_{{max}}=${theta_max:3.2f}$^{{\\circ}}$"
                f", $a$={a:3.2f}, $b$={b:3.2f}, "
                f"$\\sigma$={sigma:3.2f}"
            )
            fig_name = f"{pyrf.date_str([t_[:-3] for t_ in tints[i]], 3)}_fits"
            fig.savefig(f"/Volumes/PHD/bbfs/bbfs-dfs-fits/{fig_name}.png", dpi=300)
            plt.close("all")

        f_name = f"{pyrf.date_str([t_[:-3] for t_ in bbfs.values[int(j), :]], 3)}.npy"
        np.save(f"./data/{f_name}", np.vstack(res))


if __name__ == "__main__":
    main()
