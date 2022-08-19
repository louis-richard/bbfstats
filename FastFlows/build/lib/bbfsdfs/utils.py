#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Library of routines to propagate errors,  """

# 3rd party imports
import numpy as np
import openturns as ot
import pandas as pd

from pyrfu.mms import get_data
from pyrfu.pyrf import (ts_vec_xyz, ts_scalar, cotrans, datetime642iso8601,
                        iso86012datetime64, histogram, get_omni_data)
from scipy import optimize, signal
from uncertainties import ufloat, unumpy

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2022"
__license__ = "Apache 2.0"

__all__ = ["find_bbfs", "calc_pdf_err", "gaussian", "fairfield",
           "estimate_mp", "bz_model", "fit_df"]


def find_bbfs(inp, thresh, direction):
    r"""Algorithm to find intervals of fast flows following the criteria
    described in the paper i.e., ion bulk velocity above the threshold,
    100 km/s wide intervals and clustered with the 1 minute neighbours.

    Parameters
    ----------
    inp : xr.DataArray
        Time series of the ion bulk velocity.
    thresh : float
        Ion bulk velocity threshold.
    direction : str, {"earthward", "tailward"}
        Direction of the flow.

    Returns
    -------
    times : np.ndarray
        Times above the threshold
    tints : list
        Interval of clusters of points above the threshold (i.e., interval
        of fast flows).
    """

    if direction.lower() == "earthward":
        thresh_bvel = thresh
        indices = np.where(inp.data[:, 0] > thresh_bvel)[0]
    else:
        thresh_bvel = -thresh
        indices = np.where(inp.data[:, 0] < thresh_bvel)[0]

    times = np.vstack([inp.time.data[indices[:-1]],
                       inp.time.data[indices[:-1] + 1]]).T

    times = [list(t_) for t_ in np.atleast_2d(datetime642iso8601(times))]
    times_d64 = iso86012datetime64(np.array(times))

    if times_d64.size == 0:
        return None, None

    if len(times_d64) > 1 and any(np.diff(times_d64[:, 0]).astype(int) > 60e9):
        idx = np.where(np.diff(times_d64[:, 0]).astype(int) > 60e9)[0] + 1

        times_clusters = [times_d64[:idx[0], :]]
        for i in range(len(idx) - 1):
            times_clusters.append(times_d64[idx[i]:idx[i + 1], :])

        times_clusters.append(
            iso86012datetime64(np.array(times))[idx[-1]:, :])
    else:
        times_clusters = [iso86012datetime64(np.array(times_d64))]

    tints = []
    for t_ in times_clusters:
        idx = [np.where(inp.time.data == t_[0][0])[0][0],
               np.where(inp.time.data == t_[-1][1])[0][0]]
        if direction.lower() == "earthward":
            if any(inp.data[:idx[0], 0] < 100.):
                idx_left = np.where(inp.data[:idx[0], 0] < 100.)[0][-1]
            else:
                idx_left = int(0)

            if any(inp.data[idx[-1]:, 0] < 100.):
                idx_righ = np.where(inp.data[idx[-1]:, 0] < 100.)[0][0]
                idx_righ += idx[-1]
            else:
                idx_righ = int(-1)
        elif direction.lower() == "tailward":
            if any(inp.data[:idx[0], 0] > -100.):
                idx_left = np.where(inp.data[:idx[0], 0] > -100.)[0][-1]  #
            else:
                idx_left = int(0)

            if any(inp.data[idx[-1]:, 0] > -100.):
                idx_righ = np.where(inp.data[idx[-1]:, 0] > -100.)[0][0]
                idx_righ += idx[-1]
            else:
                idx_righ = int(-1)

            tints.append([inp.time.data[idx_left], inp.time.data[idx_righ]])
        else:
            raise ValueError

        tints.append([inp.time.data[idx_left], inp.time.data[idx_righ]])

    tints = np.array(tints)
    tints = [list(t_) for t_ in np.atleast_2d(datetime642iso8601(tints))]
    tints = list(pd.DataFrame(tints).drop_duplicates().values)

    return times, tints


def calc_pdf_err(bins, counts):
    r"""Routine to propagate the standard Poisson statistical uncertainty of
    each bin count to the probability density function.

    Parameters
    ----------
    bins : array_like
        Bin (or edges) used to compute the distribution.
    counts : array_like
        Counts per bin.

    Returns
    -------
    d_pdf : np.ndarray
        Uncertainty on the probability density function.

    """

    # Bin width
    w = np.diff(bins)[0]

    # Counts and standard Poisson statistical uncertainty of each bin count.
    counts = unumpy.uarray(counts, np.sqrt(counts))

    # Propagate to Probability Density Function
    pdf_n = counts / np.sum(counts * w)
    d_pdf = unumpy.std_devs(pdf_n)

    return d_pdf


def gaussian(x, a, x0, sigma):
    r"""Compute Gaussian distribution with mean ``x0``, amplitude ``a`` and
    standard deviation ``sigma``. Only used to fit the dawn-dusk and
    North-South distributions.

    Parameters
    ----------
    x : array_like
        Values to evaluate the Gaussian distribution.
    a : float
        Amplitude (e.g., area) of the distribution
    x0 : float
        Mean of the distribution.
    sigma : float
        Standard deviation.

    Returns
    -------
    y : np.ndarray
        Gaussian distribution.

    """

    y = a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    return y


def fairfield(time, x, y, z):
    r"""Computes the distance from spacecraft to neutral sheet using the
    model of [1]_ .

    .. math::
        \delta z = \left [ (D + H_0) * \left ( 1 - \frac{Y^2}{Y_0^2}\right
        )^{1/2} - D \right ] \sin \chi

    Parameters
    ----------
    time : array_like
        Times to compute the model neutral sheet (to get dipole tilt)
    x : array_like
        Location of the spacecraft along the Earth-tail direction
    y : array_like
        Location of the spacecraft along the dawn-dusk direction
    z : array_like
        Location of the spacecraft along the North-South direction

    Returns
    -------
    delta_z : xr.DataArray
        Time series of the distance to neutral sheet.

    References
    ----------
    .. [1]  Fairfield, D. H. (1980), A Statistical Determination of the
            Shape and Position of the Geomagnetic Neutral Sheet, Journal of
            Geophysical Research, 85 (A2).

    """
    r_gsm = ts_vec_xyz(time, np.transpose(np.stack([x, y, z])))
    dipole_sm_ = ts_vec_xyz(r_gsm.time.data, np.tile([0, 0, 1], (
        len(r_gsm.time.data), 1)))
    dipole_gsm = cotrans(dipole_sm_, "sm>gsm")
    chi = np.arctan(dipole_gsm[:, 0] / dipole_gsm[:, 2])

    h_0, d, y_0 = [10.5 * 6371, 22.5 * 6371, 14 * 6371]
    delta_z = ((h_0 + d) * np.sqrt(
        1 - r_gsm.data[:, 1] ** 2 / y_0 ** 2) - d) * np.sin(chi)
    delta_z = ts_scalar(r_gsm.time.data, delta_z / 6371)
    return delta_z


def _rzero(d_p, b_z):
    return (10.22 + 1.29 * np.tanh(0.184 * (b_z + 8.14))) * d_p ** (- 1 / 6.6)


def _alpha(d_p, b_z):
    return (0.58 - 0.007 * b_z) * (1 + 0.024 * np.log(d_p))


def _shue_mc(r_z_mc, alp_mc):
    theta_ = np.linspace(0, np.pi, int(np.pi / .1))
    with np.errstate(divide='ignore'):
        r_ = r_z_mc * (2. / (1 + np.cos(theta_))) ** alp_mc

    x_ = r_ * np.cos(theta_)
    y_ = r_ * np.sin(theta_)
    y_mp = y_[abs(x_) < 100]
    x_mp = x_[abs(x_) < 100]
    return x_mp, y_mp


def _make_sample(inp):
    n = 20.000
    h = histogram(inp, bins="fd")
    x_axis, y_axis = h.bins.data, h.data
    n_times = [int(y_axis[i] * n) for i in range(len(y_axis))]
    s = np.repeat(x_axis, n_times)
    sample = ot.Sample([[p] for p in s])
    return sample


def estimate_mp(days):
    r"""Compute the 1 sigma bounds of the magnetopause location using Shue
    model. The standard deviation is estimated as the Monte-Carlo propagated
    standard deviation of the dynamical pressure and IMF Bz"""

    days = days.astype("<M8[ns]")
    beg = days + np.timedelta64(0, "ns")
    beg = beg.astype(str)
    end = days + np.timedelta64(1, "D") - np.timedelta64(1, "s")
    end = end.astype(str)

    b_z, d_p = [], []
    print("Loading OMNI data...")
    for b, e in zip(beg, end):
        tint = [b, e]
        omni_data = get_omni_data(["p", "bzgsm"], tint)
        b_z.append(float(omni_data.bzgsm.mean("time").data))
        d_p.append(float(omni_data.p.mean("time").data))

    b_z = ts_scalar(days, np.array(b_z))
    d_p = ts_scalar(days, np.array(d_p))

    fitdist_d_p = ot.LogNormalFactory().buildAsLogNormal(_make_sample(d_p))
    fitdist_b_z = ot.NormalFactory().buildAsNormal(_make_sample(b_z))
    samples_d_p = np.squeeze(np.array(fitdist_d_p.getSample(100000)))
    samples_b_z = np.squeeze(np.array(fitdist_b_z.getSample(100000)))

    r_zero = _rzero(samples_d_p, samples_b_z)
    alpha_ = _alpha(samples_d_p, samples_b_z)
    fitdist_rz = ot.NormalFactory().buildAsNormal([[p] for p in r_zero])
    fitdist_al = ot.NormalFactory().buildAsNormal([[p] for p in alpha_])

    r_z_mc = ufloat(fitdist_rz.getMean()[0],
                    fitdist_rz.getStandardDeviation()[0])
    alp_mc = ufloat(fitdist_al.getMean()[0],
                    fitdist_al.getStandardDeviation()[0])

    x_mp, y_mp = _shue_mc(r_z_mc, alp_mc)

    x_mp_avg = unumpy.nominal_values(x_mp)
    y_mp_avg = unumpy.nominal_values(y_mp)
    x_mp_min = unumpy.nominal_values(x_mp) - unumpy.std_devs(x_mp)
    y_mp_min = unumpy.nominal_values(y_mp) - unumpy.std_devs(y_mp)
    x_mp_plu = unumpy.nominal_values(x_mp) + unumpy.std_devs(x_mp)
    y_mp_plu = unumpy.nominal_values(y_mp) + unumpy.std_devs(y_mp)

    bnds_avg = np.vstack([np.hstack([x_mp_avg, np.flip(x_mp_avg)]),
                          np.hstack([y_mp_avg, np.flip(-y_mp_avg)])])
    bnds_plu = np.vstack([np.hstack([x_mp_plu, np.flip(x_mp_plu)]),
                          np.hstack([y_mp_plu, np.flip(-y_mp_plu)])])
    bnds_min = np.vstack([np.hstack([x_mp_min, np.flip(x_mp_min)]),
                          np.hstack([y_mp_min, np.flip(-y_mp_min)])])

    return bnds_min, bnds_avg, bnds_plu


def bz_model(delta_t, a, b, c):
    r"""Model of the dipolarization front North-South magnetic field as an
    hyperbolic tangent.

    Parameters
    ----------
    delta_t : np.ndarray
        Time line to fit the magnetic field.
    a : float
        Amplitude of the dipolarization front.
    b : float
        Time scale of the dipolarization front.
    c : float
        Offset of the dipolarization front.

    Returns
    -------
    b_mod : np.ndarray
        Synthetic dipolarization front magnetic field.

    """

    b_mod = (0.5 * a) * np.tanh(delta_t / (0.5 * b)) + (c + a / 2)

    return b_mod


def fit_df(b_gsm):
    r"""Fit North-South magnetic field using the method described in [1]_
    in order to identify dipolarization fronts in the time series.

    Parameters
    ----------
    b_gsm : xarray.DataArray
        Time series of the magnetic field in GSM coordinates.

    Returns
    -------
    t_df : np.datetime64
        Time of the candidate DF.
    b_syn : xarray.DataArray
        Time series of the fit of the North-South magnetic field in GSM
        coordinates.
    a : float
        Amplitude of the hyperbolic tangent [nT].
    b : float
        Time scale of DF [s].
    sigma : float
        Root-mean square of the fit residual [nT].

    References
    ----------
    .. [1]  Fu, H. S., Y. V. Khotyaintsev, A. Vaivads, M. André, and S. Y.
            Huang (2012), Occurrence rate of earthward-propagating
            dipolarization fronts, Geophys. Res. Lett., 39, L10101,
            doi:10.1029/2012GL051784.

    """

    # Unpack magnetic field components
    b_smooth = ts_scalar(b_gsm.time.data,
                         signal.savgol_filter(b_gsm.data[:, 2], 161, 5))
    t_df = b_gsm.time.data[np.argmax(np.diff(b_smooth))]
    # Create time interval from 1min before the DF to 15 after.
    tint_df = [(t_df - np.timedelta64(30, "s")).astype(str),
               (t_df + np.timedelta64(15, "s")).astype(str)]

    # Load magnetic field associated with the time interval
    b_df = get_data("b_gsm_fgm_srvy_l2", tint_df, 1, verbose=False)
    b_filt = ts_scalar(b_df.time.data,
                       signal.savgol_filter(b_df.data[:, 2], 161, 5))

    b_z = b_filt
    p_, c_ = optimize.curve_fit(bz_model,
                                np.linspace(-30, 15, len(b_z)),
                                b_z)
    a, b, c = p_

    # Estimate the RMS residual
    b_syn = bz_model(np.linspace(-30, 15, len(b_z)), a, b, c)

    sigma = np.sqrt(np.mean((b_z - b_syn) ** 2)).data
    b_syn = ts_scalar(b_df.time.data, b_syn)
    return t_df, b_syn, a, b, sigma
