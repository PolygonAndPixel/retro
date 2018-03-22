# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Function to generate the funciton for finding expected number of photons to
survive from a 5D CLSim table.
"""

from __future__ import absolute_import, division, print_function

__all__ = '''
    MACHINE_EPS
    generate_pexp_5d_function
'''.split()

__author__ = 'P. Eller, J.L. Lanfranchi'
__license__ = '''Copyright 2017 Philipp Eller and Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

from collections import OrderedDict
import math
from os.path import abspath, dirname
import sys

import numpy as np
import numba

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import DFLT_NUMBA_JIT_KWARGS, numba_jit
from retro.const import PI
from retro.hypo.discrete_hypo import SRC_CKV_BETA1, SRC_OMNI
from retro.utils.ckv import (
    survival_prob_from_cone, survival_prob_from_smeared_cone
)
from retro.utils.geom import infer_power

MACHINE_EPS = 1e-16


def generate_pexp_5d_function(
        table, table_kind, compute_t_indep_exp, use_directionality,
        num_phi_samples=None, ckv_sigma_deg=None, template_library=None,
    ):
    """Generate a numba-compiled function for computing expected photon counts
    at a DOM, where the table's binning info is used to pre-compute various
    constants for the compiled function to use.

    Parameters
    ----------
    table : mapping
        As returned by `load_clsim_table_minimal`

    table_kind : str in {'raw_uncompr', 'ckv_uncompr', 'ckv_templ_compr'}

    compute_t_indep_exp : bool

    use_directionality : bool
        If the source photons have directionality, use it in computing photon
        expectations at the DOM.

    num_phi_samples : int
        Number of samples in the phi_dir to average over bin counts.
        (Irrelevant if `use_directionality` is False or if you use a Cherenkov
        table, which already has this parameter integrated into it.)

    ckv_sigma_deg : float
        Standard deviation in degrees for Cherenkov angle. (Irrelevant if
        `use_directionality` is False or if you use a Cherenkov table, which
        already has this parameter integrated into it.)

    template_library : shape-(n_templates, n_dir_theta, n_dir_deltaphi) array
        Containing the directionality templates for compressed tables


    Returns
    -------
    pexp_5d : callable
        Function usable to extract photon expectations from a table of
        `table_kind` and with the binning of `table`. Note that this returns
        two values (photon expectation at hit time and time-independent photon
        expectation) even if `compute_t_indep_exp` is False (whereupon the
        latter number should be ignored.

    meta : OrderedDict
        Paramters, including the binning, that uniquely identify what the
        capabilities of the returned `pexp_5d`. (Use this to eliminate
        redundant pexp_5d functions.)

    """
    tbl_is_raw = table_kind in ['raw_uncompr', 'raw_templ_compr']
    tbl_is_ckv = table_kind in ['ckv_uncompr', 'ckv_templ_compr']
    tbl_is_templ_compr = table_kind in ['raw_templ_compr', 'ckv_templ_compr']
    assert tbl_is_raw or tbl_is_ckv

    meta = OrderedDict(
        table_kind=table_kind,
        compute_t_indep_exp=compute_t_indep_exp,
        use_directionality=use_directionality,
        num_phi_samples=None if tbl_is_ckv or not use_directionality else num_phi_samples,
        ckv_sigma_deg=None if tbl_is_ckv or not use_directionality else ckv_sigma_deg,
    )

    if num_phi_samples is None:
        num_phi_samples = 0
    if ckv_sigma_deg is None:
        ckv_sigma_deg = 0

    r_min = np.min(table['r_bin_edges'])

    # Ensure r_min is zero; this removes need for lower-bound checks and a
    # subtraction each time computing bin index
    assert r_min == 0

    r_max = np.max(table['r_bin_edges'])
    rsquared_max = r_max*r_max
    r_power = infer_power(table['r_bin_edges'])
    inv_r_power = 1 / r_power
    n_r_bins = len(table['r_bin_edges']) - 1
    table_dr_pwr = (r_max - r_min)**inv_r_power / n_r_bins

    n_costheta_bins = len(table['costheta_bin_edges']) - 1
    table_dcostheta = 2 / n_costheta_bins

    t_min = np.min(table['t_bin_edges'])

    # Ensure t_min is zero; this removes need for lower-bound checks and a
    # subtraction each time computing bin index
    assert t_min == 0

    t_max = np.max(table['t_bin_edges'])
    n_t_bins = len(table['t_bin_edges']) - 1
    table_dt = (t_max - t_min) / n_t_bins

    assert table['costhetadir_bin_edges'][0] == -1
    assert table['costhetadir_bin_edges'][-1] == 1
    n_costhetadir_bins = len(table['costhetadir_bin_edges']) - 1
    table_dcosthetadir = 2 / n_costhetadir_bins
    assert np.allclose(np.diff(table['costhetadir_bin_edges']), table_dcosthetadir)
    last_costhetadir_bin_idx = n_costhetadir_bins - 1

    assert table['deltaphidir_bin_edges'][0] == 0
    assert np.isclose(table['deltaphidir_bin_edges'][-1], PI)
    n_deltaphidir_bins = len(table['deltaphidir_bin_edges']) - 1
    table_dphidir = PI / n_deltaphidir_bins
    assert np.allclose(np.diff(table['deltaphidir_bin_edges']), table_dphidir)
    last_deltaphidir_bin_idx = n_deltaphidir_bins - 1

    binning_info = dict(
        r_min=r_min, r_max=r_max, n_r_bins=n_r_bins, r_power=r_power,
        n_costheta_bins=n_costheta_bins,
        t_min=t_min, t_max=t_max, n_t_bins=n_t_bins,
        n_costhetadir_bins=n_costhetadir_bins,
        n_deltaphidir_bins=n_deltaphidir_bins,
        deltaphidir_one_sided=True
    )
    meta['binning_info'] = binning_info

    random_delta_thetas = np.array([])
    if tbl_is_raw and use_directionality and ckv_sigma_deg > 0:
        rand = np.random.RandomState(0)
        random_delta_thetas = rand.normal(
            loc=0,
            scale=np.deg2rad(ckv_sigma_deg),
            size=num_phi_samples
        )

    empty_1d_array = np.array([], dtype=np.float32).reshape((0,))
    empty_2d_array = np.array([], dtype=np.float32).reshape((0,)*2)
    empty_4d_array = np.array([], dtype=np.float32).reshape((0,)*4)

    docstr = """For a set of generated photons `sources`, compute the expected
        photons in a particular DOM at `hit_time` and the total expected
        photons, independent of time.

        This function utilizes the relative space-time coordinates _and_
        directionality of the generated photons (via "raw" 5D CLSim tables) to
        determine how many photons are expected to arrive at the DOM.

        Retro DOM tables applied to the generated photon info `sources`,
        and the total expected photon count (time integrated) -- the
        normalization of the pdf.

        Parameters
        ----------
        sources : shape (num_sources,) array of dtype SRC_DTYPE
            A discrete sequence of points describing expected sources of
            photons that result from a hypothesized event.

        hit_times : shape (num_hits,) array of dtype float64, units of ns
            Time at which the DOM recorded a hit (or multiple simultaneous
            hits). Use np.nan to indicate no hit occurred.

        dom_coord : shape (3,) array
            DOM (x, y, z) coordinate in meters (in terms of the IceCube
            coordinate system).

        quantum_efficiency : float in (0, 1]
            Scale factor that reduces detected photons due to average quantum
            efficiency of the DOM.

        table : array
            Time-dependent photon survival probability table. If using an
            uncompressed table, this will have shape
                (n_r, n_costheta, n_t, n_costhetadir, n_deltaphidir)
            while if you use a template-compressed table, this will have shape
                (n_templates, n_costhetadir, n_deltaphidir)

        table_norm : shape (n_r, n_t) array
            Normalization to apply to `table`, which is assumed to depend on
            both r- and t-dimensions and therefore is an array.

        table_map : shape (n_templates, n_costhetadir, n_deltaphidir) array, optional
            Only used if `table_kind` is template-compressed

        t_indep_table : array, optional
            Time-independent photon survival probability table. If using an
            uncompressed table, this will have shape
                (n_r, n_costheta, n_costhetadir, n_deltaphidir)
            while if using a

        t_indep_table_norm : array, optional
            r-dependent normalization (any t-dep normalization is assumed to
            already have been applied to generate the t_indep_table).

        t_indep_table_map : array, optional
            Only used if `table_kind` is template-compressed.

        Returns
        -------
        exp_p_at_all_times : float64
            If `compute_t_indep_exp` is True, return the total photons due to
            the hypothesis expected to arrive at the specified DOM for _all_
            times. If `compute_t_indep_exp` is False, return value is 0.0.

        exp_p_at_hit_times : array of shape (num_hits,), dtype float64
            Total photons due to the hypothesis expected to arrive at the
            specified DOM at the times the DOM recorded the hit.

        """

    if tbl_is_templ_compr:
        @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
        def table_lookup_mean(table, r_bin_idx, costheta_bin_idx, t_bin_idx):
            '''
            helper function for directionality averaged table lookup
            '''
            templ = table[
                r_bin_idx,
                costheta_bin_idx,
                t_bin_idx
            ]
            return templ['weight'] / template_library[templ['index']].size

        @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
        def table_lookup(table, r_bin_idx, costheta_bin_idx, t_bin_idx, costhetadir_bin_idx, deltaphidir_bin_idx):
            '''
            helper function for table lookup
            '''
            templ = table[
                r_bin_idx,
                costheta_bin_idx,
                t_bin_idx
            ]
            return templ['weight'] * template_library[templ['index'], costhetadir_bin_idx, deltaphidir_bin_idx]

    else:
        @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
        def table_lookup_mean(table, r_bin_idx, costheta_bin_idx, t_bin_idx):
            '''
            helper function for directionality averaged table lookup
            '''
            return np.mean(table[r_bin_idx, costheta_bin_idx, t_bin_idx, :, :])

        @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
        def table_lookup(table, r_bin_idx, costheta_bin_idx, t_bin_idx, costhetadir_bin_idx, deltaphidir_bin_idx):
            '''
            helper function for table lookup
            '''
            return table[r_bin_idx, costheta_bin_idx, t_bin_idx, costhetadir_bin_idx, deltaphidir_bin_idx]

    from retro.retro_types import DOM_INFO
    from retro.hypo.discrete_hypo import SRC_DTYPE
    sources_dtype = SRC_DTYPE
    dom_info_dtype = DOM_INFO
    tabledt = np.dtype([('index', np.uint16), ('weight', np.float32)])


    #@numba_jit((
    #            numba.from_dtype(sources_dtype)[:],
    #            numba.float32[:,:],
    #            numba.from_dtype(dom_info_dtype),
    #            numba.float64,
    #            numba.from_dtype(tabledt)[:,:,:],
    #            numba.float64[:,:],
    #            numba.float32[:,:,:,:],
    #            numba.float64[:]
    #            ),
    #            **DFLT_NUMBA_JIT_KWARGS)
    @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
    def pexp_5d(
            sources,
            hits,
            dom_info,
            time_window,
            table,
            table_norm,
            t_indep_table=empty_4d_array,
            t_indep_table_norm=empty_1d_array,
        ):

        #return np.random.rand(), np.random.rand()

        num_hits = hits.shape[1]

        if not dom_info['operational']:
            return np.float64(0), np.float64(0)

        # Initialize accumulators (using double precision)
        exp_p_at_all_times = np.float64(0.0)
        exp_p_at_hit_times = np.zeros(num_hits, dtype=np.float64)

        # Extract the components of the DOM coordinate
        dom_x = dom_info['x']
        dom_y = dom_info['y']
        dom_z = dom_info['z']

        # Loop over the entries (one per row)
        for source in sources:
            dx = dom_x - source['x']
            dy = dom_y - source['y']
            dz = dom_z - source['z']

            rhosquared = dx*dx + dy*dy
            rsquared = rhosquared + dz*dz

            # Continue if photon is outside the radial binning limits
            if rsquared >= rsquared_max:
                continue

            r = math.sqrt(rsquared)
            r_bin_idx = int(r**inv_r_power / table_dr_pwr)
            costheta_bin_idx = int((1 - dz/r) / table_dcostheta)

            source_kind = source['kind']

            if source_kind == SRC_OMNI and compute_t_indep_exp:
                t_indep_surv_prob = np.mean(
                    t_indep_table[r_bin_idx, costheta_bin_idx, :, :]
                )

            elif source_kind == SRC_CKV_BETA1:
                # Note that for these tables, we have to invert the photon
                # direction relative to the vector from the DOM to the photon's
                # vertex since simulation has photons going _away_ from the DOM
                # that in reconstruction will hit the DOM if they're moving
                # _towards_ the DOM.

                # Zenith angle is indep. of photon position relative to DOM
                pdir_costheta = source['dir_costheta']

                rho = math.sqrt(rhosquared)

                # \Delta\phi depends on photon position relative to the DOM...

                # Below is the projection of pdir into the (x, y) plane and the
                # projection of that onto the vector in that plane connecting
                # the photon source to the DOM. We get the cosine of the angle
                # between these vectors by solving the identity
                #   `a dot b = |a| |b| cos(deltaphi)`
                # for cos(deltaphi), where the `a` and `b` vectors are the
                # projections of the aforementioned vectors onto the xy-plane.

                if rho <= MACHINE_EPS:
                    pdir_cosdeltaphi = 1.0
                    pdir_sindeltaphi = 0.0
                else:
                    pdir_cosdeltaphi = (
                        source['dir_cosphi'] * dx/rho + source['dir_sinphi'] * dy/rho
                    )
                    # Note that the max and min here here in case numerical
                    # precision issues cause the dot product to blow up.
                    pdir_cosdeltaphi = min(1, max(-1, pdir_cosdeltaphi))
                    if tbl_is_raw:
                        pdir_sindeltaphi = math.sqrt(1 - pdir_cosdeltaphi*pdir_cosdeltaphi)

                if tbl_is_raw:
                    pdir_sintheta = source['dir_sintheta']

                    # Cherenkov angle is encoded as the projection of a
                    # length-1 vector going in the Ckv direction onto the
                    # charged particle's direction. Ergo, in the length of the
                    # pdir vector is the cosine of the ckv angle.
                    ckv_costheta = source['ckv_costheta']
                    ckv_theta = source['ckv_theta']

                    if ckv_sigma_deg > 0:
                        if compute_t_indep_exp:
                            t_indep_surv_prob, _a, _b = survival_prob_from_smeared_cone( # pylint: disable=unused-variable, invalid-name
                                theta=ckv_theta,
                                num_phi=num_phi_samples,
                                rot_costheta=pdir_costheta,
                                rot_sintheta=pdir_sintheta,
                                rot_cosphi=pdir_cosdeltaphi,
                                rot_sinphi=pdir_sindeltaphi,
                                directional_survival_prob=(
                                    t_indep_table[r_bin_idx, costheta_bin_idx, :, :]
                                ),
                                num_costheta_bins=n_costhetadir_bins,
                                num_deltaphi_bins=n_deltaphidir_bins,
                                random_delta_thetas=random_delta_thetas
                            )
                    else:
                        ckv_sintheta = source['ckv_sintheta']
                        if compute_t_indep_exp:
                            t_indep_surv_prob, _a, _b = survival_prob_from_cone( # pylint: disable=unused-variable, invalid-name
                                costheta=ckv_costheta,
                                sintheta=ckv_sintheta,
                                num_phi=num_phi_samples,
                                rot_costheta=pdir_costheta,
                                rot_sintheta=pdir_sintheta,
                                rot_cosphi=pdir_cosdeltaphi,
                                rot_sinphi=pdir_sindeltaphi,
                                directional_survival_prob=(
                                    t_indep_table[r_bin_idx, costheta_bin_idx, :, :]
                                ),
                                num_costheta_bins=n_costhetadir_bins,
                                num_deltaphi_bins=n_deltaphidir_bins,
                            )

                else: # tbl_is_ckv
                    costhetadir_bin_idx = int((pdir_costheta + 1.0) / table_dcosthetadir)

                    # Make upper edge inclusive
                    if costhetadir_bin_idx > last_costhetadir_bin_idx:
                        costhetadir_bin_idx = last_costhetadir_bin_idx

                    pdir_deltaphi = abs(math.acos(pdir_cosdeltaphi))
                    deltaphidir_bin_idx = int(pdir_deltaphi / table_dphidir)

                    # Make upper edge inclusive
                    if deltaphidir_bin_idx > last_deltaphidir_bin_idx:
                        deltaphidir_bin_idx = last_deltaphidir_bin_idx

                    t_indep_surv_prob = t_indep_table[
                        r_bin_idx,
                        costheta_bin_idx,
                        costhetadir_bin_idx,
                        deltaphidir_bin_idx
                    ]
                    if t_indep_surv_prob == 0:
                        print((r_bin_idx, costheta_bin_idx, costhetadir_bin_idx, deltaphidir_bin_idx))
                        raise ValueError('xyz')

            else:
                raise NotImplementedError('Source kind not implmented')

            source_photons = source['photons']

            if compute_t_indep_exp:
                ti_norm = t_indep_table_norm[r_bin_idx]
                if not ti_norm > 0:
                    print(r_bin_idx, dom_info, ti_norm)
                    raise ValueError('time indep norm is not > 0!')
                exp_p_at_all_times += (
                    source_photons * ti_norm * t_indep_surv_prob
                )

            for hit_t_idx in range(num_hits):
                hit_time = hits[0, hit_t_idx]

                # Causally impossible? (Note the comparison is written such that it
                # will evaluate to True if hit_time is NaN.)
                source_t = source['t']
                if not source_t <= hit_time:
                    continue

                # A photon that starts immediately in the past (before the DOM
                # was hit) will show up in the Retro DOM tables in bin 0; the
                # further in the past the photon started, the higher the time
                # bin index. Therefore, subract source time from hit time.
                dt = hit_time - source_t

                # Is relative time outside binning?
                if dt >= t_max:
                    continue

                t_bin_idx = int(dt / table_dt)

                r_t_bin_norm = table_norm[r_bin_idx, t_bin_idx]

                if source_kind == SRC_OMNI:
                    surv_prob_at_hit_t = table_lookup_mean(table, r_bin_idx, costheta_bin_idx, t_bin_idx)

                elif source_kind == SRC_CKV_BETA1:
                #    if tbl_is_raw and not tbl_is_templ_compr:
                #        if ckv_sigma_deg > 0:
                #            surv_prob_at_hit_t, _c, _d = survival_prob_from_smeared_cone( # pylint: disable=unused-variable, invalid-name
                #                theta=ckv_theta,
                #                num_phi=num_phi_samples,
                #                rot_costheta=pdir_costheta,
                #                rot_sintheta=pdir_sintheta,
                #                rot_cosphi=pdir_cosdeltaphi,
                #                rot_sinphi=pdir_sindeltaphi,
                #                directional_survival_prob=(
                #                    table[r_bin_idx, costheta_bin_idx, t_bin_idx, :, :]
                #                ),
                #                num_costheta_bins=n_costhetadir_bins,
                #                num_deltaphi_bins=n_deltaphidir_bins,
                #                random_delta_thetas=random_delta_thetas
                #            )
                #        else:
                #            surv_prob_at_hit_t, _c, _d = survival_prob_from_cone( # pylint: disable=unused-variable, invalid-name
                #                costheta=ckv_costheta,
                #                sintheta=ckv_sintheta,
                #                num_phi=num_phi_samples,
                #                rot_costheta=pdir_costheta,
                #                rot_sintheta=pdir_sintheta,
                #                rot_cosphi=pdir_cosdeltaphi,
                #                rot_sinphi=pdir_sindeltaphi,
                #                directional_survival_prob=(
                #                    table[r_bin_idx, costheta_bin_idx, t_bin_idx, :, :]
                #                ),
                #                num_costheta_bins=n_costhetadir_bins,
                #                num_deltaphi_bins=n_deltaphidir_bins,
                #            )

                #    else: # tbl_is_ckv
                    surv_prob_at_hit_t = table_lookup(table, 
                                                          r_bin_idx,
                                                          costheta_bin_idx,
                                                          t_bin_idx,
                                                          costhetadir_bin_idx,
                                                          deltaphidir_bin_idx
                                                          )

                else:
                    raise NotImplementedError('Source kind not implemented')

                exp_p_at_hit_times[hit_t_idx] += source_photons * r_t_bin_norm * surv_prob_at_hit_t

        quantum_efficiency = dom_info['quantum_efficiency']
        noise_rate_per_ns = dom_info['noise_rate_per_ns']

        sum_log_at_hit_times = np.float64(0.0)
        for hit_idx in range(num_hits):
            ep_at_ht = exp_p_at_hit_times[hit_idx]
            hit_mult = hits[1, hit_idx]
            if hit_mult == 0:
                print(dom_info, hit_idx, hits[:, hit_idx])
            sum_log_at_hit_times += (
                hit_mult * math.log(quantum_efficiency * ep_at_ht + noise_rate_per_ns)
            )
        #exp_p_at_hit_times = quantum_efficiency * exp_p_at_hit_times
        exp_p_at_all_times = (
            exp_p_at_all_times * quantum_efficiency
            + noise_rate_per_ns * time_window
        )

        #if sum_log_at_hit_times == 0 and num_hits > 0:
        #    print(hits)
        #    print(dom_info)
        #    print(sum_log_at_hit_times)
        #    print(exp_p_at_all_times)

        if np.isinf(sum_log_at_hit_times):
            print(dom_info, sum_log_at_hit_times)
            raise ValueError('sum_log_at_hit_times is inf!')

        return exp_p_at_all_times, sum_log_at_hit_times


    #@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
    #def pexp_5d_templ_compr(
    #        sources,
    #        hits,
    #        dom_info,
    #        time_window,
    #        table,
    #        table_norm,
    #        t_indep_table=empty_4d_array,
    #        t_indep_table_norm=empty_1d_array,
    #    ):
    #    num_hits = hits.shape[1]

    #    if not dom_info['operational']:
    #        return np.float64(0), np.float64(0)

    #    # Initialize accumulators (using double precision)
    #    exp_p_at_all_times = np.float64(0.0)
    #    exp_p_at_hit_times = np.zeros(num_hits, dtype=np.float64)

    #    # Extract the components of the DOM coordinate
    #    dom_x = dom_info['x']
    #    dom_y = dom_info['y']
    #    dom_z = dom_info['z']

    #    # Loop over the entries (one per row)
    #    for source in sources:
    #        dx = dom_x - source['x']
    #        dy = dom_y - source['y']
    #        dz = dom_z - source['z']

    #        rhosquared = dx*dx + dy*dy
    #        rsquared = rhosquared + dz*dz

    #        # Continue if photon is outside the radial binning limits
    #        if rsquared >= rsquared_max:
    #            continue

    #        r = math.sqrt(rsquared)
    #        r_bin_idx = int(r**inv_r_power / table_dr_pwr)
    #        costheta_bin_idx = int((1 - dz/r) / table_dcostheta)

    #        source_kind = source['kind']

    #        if source_kind == SRC_OMNI and compute_t_indep_exp:
    #            t_indep_surv_prob = np.mean(
    #                t_indep_table[r_bin_idx, costheta_bin_idx, :, :]
    #            )

    #        elif source_kind == SRC_CKV_BETA1:
    #            # Note that for these tables, we have to invert the photon
    #            # direction relative to the vector from the DOM to the photon's
    #            # vertex since simulation has photons going _away_ from the DOM
    #            # that in reconstruction will hit the DOM if they're moving
    #            # _towards_ the DOM.

    #            # Zenith angle is indep. of photon position relative to DOM
    #            pdir_costheta = source['dir_costheta']

    #            rho = math.sqrt(rhosquared)

    #            # \Delta\phi depends on photon position relative to the DOM...

    #            # Below is the projection of pdir into the (x, y) plane and the
    #            # projection of that onto the vector in that plane connecting
    #            # the photon source to the DOM. We get the cosine of the angle
    #            # between these vectors by solving the identity
    #            #   `a dot b = |a| |b| cos(deltaphi)`
    #            # for cos(deltaphi).
    #            #
    #            if rho <= MACHINE_EPS:
    #                pdir_cosdeltaphi = 1.0
    #                pdir_sindeltaphi = 0.0
    #            else:
    #                pdir_cosdeltaphi = (
    #                    source['dir_cosphi'] * dx/rho + source['dir_sinphi'] * dy/rho
    #                )
    #                # Note that the max and min here here in case numerical
    #                # precision issues cause the dot product to blow up.
    #                pdir_cosdeltaphi = min(1, max(-1, pdir_cosdeltaphi))
    #                if tbl_is_raw:
    #                    pdir_sindeltaphi = math.sqrt(1 - pdir_cosdeltaphi*pdir_cosdeltaphi)

    #            if tbl_is_raw:
    #                pdir_sintheta = source['dir_sintheta']

    #                # Cherenkov angle is encoded as the projection of a
    #                # length-1 vector going in the Ckv direction onto the
    #                # charged particle's direction. Ergo, in the length of the
    #                # pdir vector is the cosine of the ckv angle.
    #                ckv_costheta = source['ckv_costheta']
    #                ckv_theta = source['ckv_theta']

    #                if ckv_sigma_deg > 0:
    #                    if compute_t_indep_exp:
    #                        t_indep_surv_prob, _a, _b = survival_prob_from_smeared_cone( # pylint: disable=unused-variable, invalid-name
    #                            theta=ckv_theta,
    #                            num_phi=num_phi_samples,
    #                            rot_costheta=pdir_costheta,
    #                            rot_sintheta=pdir_sintheta,
    #                            rot_cosphi=pdir_cosdeltaphi,
    #                            rot_sinphi=pdir_sindeltaphi,
    #                            directional_survival_prob=(
    #                                t_indep_table[r_bin_idx, costheta_bin_idx, :, :]
    #                            ),
    #                            num_costheta_bins=n_costhetadir_bins,
    #                            num_deltaphi_bins=n_deltaphidir_bins,
    #                            random_delta_thetas=random_delta_thetas
    #                        )
    #                else:
    #                    ckv_sintheta = source['ckv_sintheta']
    #                    if compute_t_indep_exp:
    #                        t_indep_surv_prob, _a, _b = survival_prob_from_cone( # pylint: disable=unused-variable, invalid-name
    #                            costheta=ckv_costheta,
    #                            sintheta=ckv_sintheta,
    #                            num_phi=num_phi_samples,
    #                            rot_costheta=pdir_costheta,
    #                            rot_sintheta=pdir_sintheta,
    #                            rot_cosphi=pdir_cosdeltaphi,
    #                            rot_sinphi=pdir_sindeltaphi,
    #                            directional_survival_prob=(
    #                                t_indep_table[r_bin_idx, costheta_bin_idx, :, :]
    #                            ),
    #                            num_costheta_bins=n_costhetadir_bins,
    #                            num_deltaphi_bins=n_deltaphidir_bins,
    #                        )

    #            else: # tbl_is_ckv
    #                costhetadir_bin_idx = int((pdir_costheta + 1.0) / table_dcosthetadir)

    #                # Make upper edge inclusive
    #                if costhetadir_bin_idx > last_costhetadir_bin_idx:
    #                    costhetadir_bin_idx = last_costhetadir_bin_idx

    #                pdir_deltaphi = math.acos(pdir_cosdeltaphi)
    #                deltaphidir_bin_idx = int(pdir_deltaphi / table_dphidir)

    #                # Make upper edge inclusive
    #                if deltaphidir_bin_idx > last_deltaphidir_bin_idx:
    #                    deltaphidir_bin_idx = last_deltaphidir_bin_idx

    #                t_indep_surv_prob = t_indep_table[
    #                    r_bin_idx,
    #                    costheta_bin_idx,
    #                    costhetadir_bin_idx,
    #                    deltaphidir_bin_idx
    #                ]
    #                if t_indep_surv_prob == 0:
    #                    print((r_bin_idx, costheta_bin_idx, costhetadir_bin_idx, deltaphidir_bin_idx))
    #                    raise ValueError('xyz')

    #        else:
    #            raise NotImplementedError('Source kind not implmented')

    #        source_photons = source['photons']

    #        if compute_t_indep_exp:
    #            ti_norm = t_indep_table_norm[r_bin_idx]
    #            if not ti_norm > 0:
    #                print(r_bin_idx, dom_info, ti_norm)
    #                raise ValueError('time indep norm is not > 0!')
    #            exp_p_at_all_times += (
    #                source_photons * ti_norm * t_indep_surv_prob
    #            )

    #        for hit_t_idx in range(num_hits):
    #            hit_time = hits[0, hit_t_idx]

    #            # Causally impossible? (Note the comparison is written such that it
    #            # will evaluate to True if hit_time is NaN.)
    #            source_t = source['t']
    #            if not source_t <= hit_time:
    #                continue

    #            # A photon that starts immediately in the past (before the DOM
    #            # was hit) will show up in the Retro DOM tables in bin 0; the
    #            # further in the past the photon started, the higher the time
    #            # bin index. Therefore, subract source time from hit time.
    #            dt = hit_time - source_t

    #            # Is relative time outside binning?
    #            if dt >= t_max:
    #                continue

    #            t_bin_idx = int(dt / table_dt)

    #            r_t_bin_norm = table_norm[r_bin_idx, t_bin_idx]

    #            templ = table[
    #                r_bin_idx,
    #                costheta_bin_idx,
    #                t_bin_idx
    #            ]
    #            if source_kind == SRC_OMNI:
    #                surv_prob_at_hit_t = np.mean(
    #                    templ['weight'] *
    #                    template_library[templ['index'],:,:]
    #                )

    #            else: # tbl_is_ckv
    #                surv_prob_at_hit_t = (templ['weight'] * 
    #                    template_library[templ['index'],
    #                    costhetadir_bin_idx,
    #                    deltaphidir_bin_idx
    #                    ])


    #            exp_p_at_hit_times[hit_t_idx] += source_photons * r_t_bin_norm * surv_prob_at_hit_t

    #    quantum_efficiency = dom_info['quantum_efficiency']
    #    noise_rate_per_ns = dom_info['noise_rate_per_ns']

    #    sum_log_at_hit_times = np.float64(0.0)
    #    for hit_idx in range(num_hits):
    #        ep_at_ht = exp_p_at_hit_times[hit_idx]
    #        hit_mult = hits[1, hit_idx]
    #        if hit_mult == 0:
    #            print(dom_info, hit_idx, hits[:, hit_idx])
    #        sum_log_at_hit_times += (
    #            hit_mult * math.log(quantum_efficiency * ep_at_ht + noise_rate_per_ns)
    #        )
    #    #exp_p_at_hit_times = quantum_efficiency * exp_p_at_hit_times
    #    exp_p_at_all_times = (
    #        exp_p_at_all_times * quantum_efficiency
    #        + noise_rate_per_ns * time_window
    #    )

    #    #if sum_log_at_hit_times == 0 and num_hits > 0:
    #    #    print(hits)
    #    #    print(dom_info)
    #    #    print(sum_log_at_hit_times)
    #    #    print(exp_p_at_all_times)

    #    if np.isinf(sum_log_at_hit_times):
    #        print(dom_info, sum_log_at_hit_times)
    #        raise ValueError('sum_log_at_hit_times is inf!')

    #    return exp_p_at_all_times, sum_log_at_hit_times

    pexp_5d.__doc__ = docstr

    return pexp_5d, meta
