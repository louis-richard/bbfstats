#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Library of routines to load easily the electromagnetic fields, the ion
moments and the ion energy spectra.

Note:
    All the data are loaded in Fast Survey mode for MMS 1.
    
"""

# 3rd party imports
from pyrfu.mms import (get_data, remove_idist_background, rotate_tensor,
                       db_init)
from pyrfu.pyrf import cotrans, trace
from scipy import constants

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2022"
__license__ = "Apache 2.0"

# Setup path to MMS data (default)
#db_init("/Volumes/PHD/bbfs/data")

__all__ = ["load_fields", "load_fpi"]


def load_fields(tint):
    r"""Load the Fast Survey electromagnetic field from FGM and EDP.

    Parameters
    ----------
    tint : list
        Time interval.

    Returns
    -------
    b_gsm : xr.DataArray
        Time series of the magnetic field in geocentric solar magnetospheric
        coordinates.
    e_gsm : xr.DataArray
        Time series of the electric field in geocentric solar magnetospheric
        coordinates.

    """
    b_gsm = get_data("b_gsm_fgm_srvy_l2", tint, 1)
    e_gse = get_data("e_gse_edp_fast_l2", tint, 1)
    e_gsm = cotrans(e_gse, "gse>gsm")

    return b_gsm, e_gsm


def load_fpi(tint):
    r"""Load Fast Survey ion moments from FPI-DIS, and remove background
    using the method described in [1]_.

    Parameters
    ----------
    tint : list
        Time interval.

    Returns
    -------
    n_i : xr.DataArray
        Ion number density
    v_gsm_i : xr.DataArray
        Ion bulk velocity in geocentric solar magnetospheric coordinates.
    t_i : xr.DataArray
        Ion temperature
    def_i : xr.DataArray
        Ion differential energy flux.

    References
    ----------
    .. [1]  Gershman, D. J., Dorelli, J. C., Avanov, L. A., Gliese, U.,
            Barrie, A., Schiff, C., et al. (2019). Systematic uncertainties
            in plasma parameters reported by the fast plasma investigation on
            NASA's magnetospheric multiscale mission.
            Journal of Geophysical Research: Space Physics, 124, 10,345–10,359
            https://doi.org/10.1029/2019JA026980

    """

    b_gse = get_data("b_gse_fgm_srvy_l2", tint, 1)

    n_i = get_data("ni_fpi_fast_l2", tint, 1)
    v_gse_i = get_data("vi_gse_fpi_fast_l2", tint, 1)
    t_gse_i = get_data("ti_gse_fpi_fast_l2", tint, 1)

    #
    p_gse_i = n_i.data[:, None, None] * t_gse_i
    p_gse_i.data *= 1e15 * constants.elementary_charge

    # Background radiation
    nbg_i = get_data(f"nbgi_fpi_fast_l2", tint, 1)
    pbg_i = get_data(f"pbgi_fpi_fast_l2", tint, 1)

    # Remove penetrating radiations
    moms_clean = remove_idist_background(n_i, v_gse_i, p_gse_i, nbg_i, pbg_i)
    n_i_clean, v_gse_i_clean, p_gse_i_clean = moms_clean

    # Compute scalar temperature from pressure tensor
    t_gse_i_clean = p_gse_i_clean / n_i_clean.data[:, None, None]
    t_gse_i_clean.data /= 1e15 * constants.elementary_charge

    # Remove extremely low density points
    v_gse_i = v_gse_i_clean[n_i_clean > .005, ...]
    v_gsm_i = cotrans(v_gse_i, "gse>gsm")
    t_gse_i = t_gse_i_clean[n_i_clean > .005, ...]
    n_i = n_i_clean[n_i_clean > .005]
    t_fac_i = rotate_tensor(t_gse_i, "fac", b_gse, "pp")
    t_i = trace(t_fac_i) / 3

    def_i = get_data("defi_fpi_fast_l2", tint, 1)

    return n_i, v_gsm_i, t_i, def_i
