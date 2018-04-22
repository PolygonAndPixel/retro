#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, redefined-outer-name, range-builtin-not-iterating, too-many-locals

"""
Instantiate Retro tables and find the max over the log-likelihood space.
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'CUBE_DIMS',
    'PRI_UNIFORM',
    'PRI_LOG_UNIFORM',
    'PRI_LOG_NORMAL',
    'PRI_COSINE',
    'PRI_GAUSSIAN',
    'PRI_SPEFIT2',
    'PRI_CAUCHY',
    'run_multinest',
    'reco',
    'parse_args'
]

__author__ = 'J.L. Lanfranchi, P. Eller'
__license__ = '''Copyright 2017 Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

from argparse import ArgumentParser
from collections import OrderedDict
from math import acos, exp

from os.path import abspath, dirname, join
import pickle
import sys
import time

import numpy as np
from scipy import stats

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import HYPO_PARAMS_T, LLHP_T, init_obj
from retro.const import TWO_PI, ALL_STRS_DOMS_SET, NUM_DOMS_TOT
from retro.retro_types import HypoParams8D, HypoParams10D
from retro.utils.misc import expand, mkdir, sort_dict
#from retro.likelihood import get_llh


T = 'time'
X = 'x'
Y = 'y'
Z = 'z'
TRCK_ZEN = 'track_zenith'
TRCK_AZ = 'track_azimuth'
CSCD_ZEN = 'cascade_zenith'
CSCD_AZ = 'cascade_azimuth'
CSCD_ENERGY = 'cascade_energy'
TRCK_ENERGY = 'track_energy'
ENERGY = 'energy'
TRCK_FRAC = 'track_fraction'


if HYPO_PARAMS_T is HypoParams8D:
    #CUBE_DIMS = [X, Y, Z, T, TRCK_ZEN, TRCK_AZ, ENERGY, TRCK_FRAC]
    CUBE_DIMS = [X, Y, Z, T, TRCK_ZEN, TRCK_AZ, CSCD_ENERGY, TRCK_ENERGY]
elif HYPO_PARAMS_T is HypoParams10D:
    #CUBE_DIMS = [X, Y, Z, T, TRCK_ZEN, TRCK_AZ, ENERGY, TRCK_FRAC, CSCD_ZEN, CSCD_AZ]
    CUBE_DIMS = [X, Y, Z, T, TRCK_ZEN, TRCK_AZ, CSCD_ENERGY, TRCK_ENERGY, CSCD_ZEN, CSCD_AZ]
else:
    raise NotImplementedError(str(HYPO_PARAMS_T))


CUBE_T_IDX = CUBE_DIMS.index(T)
CUBE_X_IDX = CUBE_DIMS.index(X)
CUBE_Y_IDX = CUBE_DIMS.index(Y)
CUBE_Z_IDX = CUBE_DIMS.index(Z)
CUBE_TRACK_ZEN_IDX = CUBE_DIMS.index(TRCK_ZEN)
CUBE_TRACK_AZ_IDX = CUBE_DIMS.index(TRCK_AZ)
#CUBE_ENERGY_IDX = CUBE_DIMS.index(ENERGY)
#CUBE_TRACK_FRAC_IDX = CUBE_DIMS.index(TRCK_FRAC)
CUBE_CSCD_ENERGY_IDX = CUBE_DIMS.index(CSCD_ENERGY)
CUBE_TRCK_ENERGY_IDX = CUBE_DIMS.index(TRCK_ENERGY)

if HYPO_PARAMS_T is HypoParams8D:
    CUBE_CSCD_ZEN_IDX = None
    CUBE_CSCD_AZ_IDX = None
else:
    CUBE_CSCD_ZEN_IDX = CUBE_DIMS.index(CSCD_ZEN)
    CUBE_CSCD_AZ_IDX = CUBE_DIMS.index(CSCD_AZ)


PRI_UNIFORM = 'uniform'
PRI_LOG_UNIFORM = 'log_uniform'
PRI_LOG_NORMAL = 'log_normal'
PRI_COSINE = 'cosine'
PRI_GAUSSIAN = 'gaussian'
PRI_SPEFIT2 = 'spefit2'
PRI_CAUCHY = 'cauchy'


def run_multinest(
        outdir,
        event_idx,
        event,
        dom_tables,
        hypo_handler,
        priors,
        importance_sampling,
        max_modes,
        const_eff,
        n_live,
        evidence_tol,
        sampling_eff,
        max_iter,
        seed,
    ):
    """Setup and run MultiNest on an event.

    See the README file from MultiNest for greater detail on parameters
    specific to to MultiNest (parameters from `importance_sampling` on).

    Parameters
    ----------
    outdir
    event_idx
    event
    dom_tables,
    hypo_handler,
    priors : mapping
    importance_sampling
    max_modes
    const_eff
    n_live
    evidence_tol
    sampling_eff
    max_iter
        Note that this limit is the maximum number of sample replacements and
        _not_ max number of likelihoods evaluated. A replacement only occurs
        when a likelihood is found that exceeds the minimum likelihood among
        the live points.
    seed

    Returns
    -------
    llhp : shape (num_llh,) structured array of dtype retro.LLHP_T
        LLH and the corresponding parameter values.

    mn_meta : OrderedDict
        Metadata used for running MultiNest, including priors, parameters, and
        the keyword args used to invoke the `pymultinest.run` function.

    """
    # pylint: disable=missing-docstring
    # Import pymultinest here; it's a less common dependency, so other
    # functions / constants in this module will still be import-able w/o it.
    import pymultinest

    hits = event['hits']
    hits_indexer = event['hits_indexer']
    hits_summary = event['hits_summary']

    priors_used = OrderedDict()

    prior_funcs = []
    for dim_num, dim_name in enumerate(CUBE_DIMS):
        prior_kind, prior_params = priors[dim_name]
        if prior_kind is PRI_UNIFORM:
            # Time is special since prior is relative to hits in the event
            if dim_name == T:
                prior_params = (
                    hits_summary['earliest_hit_time'] + prior_params[0],
                    hits_summary['latest_hit_time'] + prior_params[1]
                )
            priors_used[dim_name] = (prior_kind, prior_params)

            if prior_params == (0, 1):
                continue
                #def prior_func(cube): # pylint: disable=unused-argument
                #    pass
            elif np.min(prior_params[0]) == 0:
                maxval = np.max(prior_params)
                def prior_func(cube, n=dim_num, maxval=maxval):
                    cube[n] = cube[n] * maxval
            else:
                minval = np.min(prior_params)
                width = np.max(prior_params) - minval
                def prior_func(cube, n=dim_num, width=width, minval=minval):
                    cube[n] = cube[n] * width + minval

        elif prior_kind == PRI_LOG_UNIFORM:
            priors_used[dim_name] = (prior_kind, prior_params)
            log_min = np.log(np.min(prior_params))
            log_width = np.log(np.max(prior_params) / np.min(prior_params))
            def prior_func(cube, n=dim_num, log_width=log_width, log_min=log_min):
                cube[n] = exp(cube[n] * log_width + log_min)

        elif prior_kind == PRI_COSINE:
            priors_used[dim_name] = (prior_kind, prior_params)
            cos_min = np.min(prior_params)
            cos_width = np.max(prior_params) - cos_min
            def prior_func(cube, n=dim_num, cos_width=cos_width, cos_min=cos_min):
                cube[n] = acos(cube[n] * cos_width + cos_min)

        elif prior_kind == PRI_GAUSSIAN:
            priors_used[dim_name] = (prior_kind, prior_params)
            mean, stddev = prior_params
            norm = 1 / (stddev * np.sqrt(TWO_PI))
            def prior_func(cube, n=dim_num, norm=norm, mean=mean, stddev=stddev):
                cube[n] = norm * exp(-((cube[n] - mean) / stddev)**2)

        elif prior_kind == PRI_LOG_NORMAL:
            priors_used[dim_name] = (prior_kind, prior_params)
            shape, loc, scale, low, high = prior_params
            lognorm = stats.lognorm(shape, loc, scale)
            def prior_func(cube, lognorm=lognorm, n=dim_num, low=low, high=high):
                cube[n] = np.clip(lognorm.isf(cube[n]), a_min=low, a_max=high)

        elif prior_kind == PRI_SPEFIT2:
            spe_fit_val = event['recos']['SPEFit2'][dim_name]
            rel_loc, scale, low, high = prior_params
            loc = spe_fit_val + rel_loc
            cauchy = stats.cauchy(loc=loc, scale=scale)
            if dim_name == T:
                low += hits_summary['time_window_start']
                high += hits_summary['time_window_stop']
            priors_used[dim_name] = (PRI_CAUCHY, (loc, scale, low, high))
            def prior_func(cube, cauchy=cauchy, n=dim_num, low=low, high=high):
                cube[n] = np.clip(cauchy.isf(cube[n]), a_min=low, a_max=high)

        else:
            raise NotImplementedError('Prior "{}" not implemented.'
                                      .format(prior_kind))
            assert np.isfinite(cube[n]), 'cube dim %u not finite %s'%(n, c[n])

        prior_funcs.append(prior_func)

    param_values = []
    log_likelihoods = []
    t_start = []

    report_after = 1

    def prior(cube, ndim, nparams): # pylint: disable=unused-argument
        """Function for pymultinest to translate the hypercube MultiNest uses
        (each value is in [0, 1]) into the dimensions of the parameter space.

        Note that the cube dimension names are defined in module variable
        `CUBE_DIMS` for reference elsewhere.

        """
        try:
            for prior_func in prior_funcs:
                prior_func(cube)
        except Exception as e:
            print('Exception in prior call:')
            print(e.__doc__)
            print(e.message)
            raise 

    get_llh = dom_tables._get_llh # pylint: disable=protected-access
    dom_info = dom_tables.dom_info
    tables = dom_tables.tables
    table_norm = dom_tables.table_norm
    t_indep_tables = dom_tables.t_indep_tables
    t_indep_table_norm = dom_tables.t_indep_table_norm
    sd_idx_table_indexer = dom_tables.sd_idx_table_indexer
    time_window = np.float32(
        hits_summary['time_window_stop'] - hits_summary['time_window_start']
    )
    print('time window = %s'%time_window)
    # TODO: implement logic allowing for not all DOMs to be used
    #hit_sd_indices = np.array(
    #    sorted(dom_tables.use_sd_indices_set.union(hits_indexer['sd_idx'])),
    #    dtype=np.uint32
    #)
    hit_sd_indices = hits_indexer['sd_idx']
    unhit_sd_indices = np.array(
        sorted(ALL_STRS_DOMS_SET.difference(hit_sd_indices)),
        dtype=np.uint32
    )

    assert(len(hit_sd_indices) + len(unhit_sd_indices) == NUM_DOMS_TOT)

    # DEBUG
    #table_indices = []
    #t_indep_indices = []

    def loglike(cube, ndim, nparams): # pylint: disable=unused-argument
        """Function pymultinest calls to get llh values.

        Note that this is called _after_ `prior` has been called, so `cube`
        alsready contains the parameter values scaled to be in their physical
        ranges.

        """
        try:
            if not t_start:
                t_start.append(time.time())

            t0 = time.time()

            #total_energy = cube[CUBE_ENERGY_IDX]
            #track_fraction = cube[CUBE_TRACK_FRAC_IDX]

            if HYPO_PARAMS_T is HypoParams8D:
                hypo = HYPO_PARAMS_T(
                    time=cube[CUBE_T_IDX],
                    x=cube[CUBE_X_IDX],
                    y=cube[CUBE_Y_IDX],
                    z=cube[CUBE_Z_IDX],
                    track_zenith=cube[CUBE_TRACK_ZEN_IDX],
                    track_azimuth=cube[CUBE_TRACK_AZ_IDX],
                    #cascade_energy=total_energy * (1 - track_fraction),
                    #track_energy=total_energy * track_fraction
                    cascade_energy=cube[CUBE_CSCD_ENERGY_IDX],
                    track_energy=cube[CUBE_TRCK_ENERGY_IDX],
                )
            else:
                hypo = HYPO_PARAMS_T(
                    time=cube[CUBE_T_IDX],
                    x=cube[CUBE_X_IDX],
                    y=cube[CUBE_Y_IDX],
                    z=cube[CUBE_Z_IDX],
                    track_zenith=cube[CUBE_TRACK_ZEN_IDX],
                    track_azimuth=cube[CUBE_TRACK_AZ_IDX],
                    #cascade_energy=total_energy * (1 - track_fraction),
                    #track_energy=total_energy * track_fraction,
                    cascade_energy=cube[CUBE_CSCD_ENERGY_IDX],
                    track_energy=cube[CUBE_TRCK_ENERGY_IDX],
                    cascade_zenith=cube[CUBE_CSCD_ZEN_IDX],
                    cascade_azimuth=cube[CUBE_CSCD_AZ_IDX]
                )
            sources = hypo_handler.get_sources(hypo)
            llh = get_llh(
                sources=sources,
                hits=hits,
                hits_indexer=hits_indexer,
                unhit_sd_indices=unhit_sd_indices,
                sd_idx_table_indexer=sd_idx_table_indexer,
                time_window=time_window,
                dom_info=dom_info,
                tables=tables,
                table_norm=table_norm,
                t_indep_tables=t_indep_tables,
                t_indep_table_norm=t_indep_table_norm,
                # DEBUG
                #table_indices=table_indices,
                #t_indep_indices=t_indep_indices
            )
            # DEBUG
            #print('')
            #with open('/tmp/get_llh.asm', 'w') as f:
            #print(get_llh.inspect_asm(get_llh.signatures[0]))
            #print('number of signatures:', len(get_llh.signatures))
            #print('')
            #raise Exception()

            t1 = time.time()

            param_values.append(hypo)
            log_likelihoods.append(llh)

            n_calls = len(log_likelihoods)

            if n_calls % report_after == 0:
                t_now = time.time()
                best_idx = np.argmax(log_likelihoods)
                best_llh = log_likelihoods[best_idx]
                best_p = param_values[best_idx]
                this_llh = log_likelihoods[-1]
                this_p = param_values[-1]
                print('')
                if HYPO_PARAMS_T is HypoParams8D:
                    print('tot sources = %s, tot photons = %s'%(len(sources), np.sum(sources['photons'])))
                    print('total noise hits = %s'%(np.sum(np.nan_to_num(dom_info['noise_rate_per_ns']))*time_window))
                    truth = event['truth']
                    print(('            truth = '
                           '(t={:+.1f}, x={:+.1f}, y={:+.1f}, z={:+.1f},'
                           ' zen={:.1f} deg, az={:.1f} deg, Etrk={:.1f}, Ecscd={:.1f})')
                          .format(
                              truth['time'], truth['x'], truth['y'], truth['z'],
                              np.rad2deg(np.arccos(truth['coszen'])),
                              np.rad2deg(truth['azimuth']),
                              truth['highest_energy_daughter_energy'],
                              truth['cascade_energy']))
                    print(('best llh = {:.3f} @ '
                           '(t={:+.1f}, x={:+.1f}, y={:+.1f}, z={:+.1f},'
                           ' zen={:.1f} deg, az={:.1f} deg, Etrk={:.1f}, Ecscd={:.1f})')
                          .format(
                              best_llh, best_p.time, best_p.x, best_p.y, best_p.z,
                              np.rad2deg(best_p.track_zenith),
                              np.rad2deg(best_p.track_azimuth),
                              best_p.track_energy,
                              best_p.cascade_energy))
                    print(('this llh = {:.3f} @ '
                           '(t={:+.1f}, x={:+.1f}, y={:+.1f}, z={:+.1f},'
                           ' zen={:.1f} deg, az={:.1f} deg, Etrk={:.1f}, Ecscd={:.1f})')
                          .format(
                              this_llh, this_p.time, this_p.x, this_p.y, this_p.z,
                              np.rad2deg(this_p.track_zenith),
                              np.rad2deg(this_p.track_azimuth),
                              this_p.track_energy,
                              this_p.cascade_energy))
                else:
                    print(('best llh = {:.3f} @'
                           ' (t={:+.1f}, x={:+.1f}, y={:+.1f}, z={:+.1f},'
                           ' zen_trk={:.1f} deg, zen_csc={:.1f} deg,'
                           ' az_trk={:.1f}, az_csc={:.1f},'
                           ' Etrk={:.1f}, Ecscd={:.1f})')
                          .format(
                              best_llh, best_p.time, best_p.x, best_p.y, best_p.z,
                              np.rad2deg(best_p.track_zenith),
                              np.rad2deg(best_p.cascade_zenith),
                              np.rad2deg(best_p.track_azimuth),
                              np.rad2deg(best_p.cascade_azimuth),
                              best_p.track_energy,
                              best_p.cascade_energy))
                print('{} LLH computed'.format(n_calls))
                print('avg time per llh: {:.3f} ms'.format((t_now - t_start[0])/n_calls*1000))
                print('this llh took:    {:.3f} ms'.format((t1 - t0)*1000))
                print('')

            assert np.isfinite(llh), 'llh not finite! %s'%llh
            return llh
        except Exception as e:
            print('Exception in LLH call:')
            print(e.__doc__)
            print(e.message)
            raise 


    n_dims = len(HYPO_PARAMS_T._fields)
    mn_kw = OrderedDict([
        ('n_dims', n_dims),
        ('n_params', n_dims),
        ('n_clustering_params', n_dims),
        ('wrapped_params', [int('azimuth' in p.lower()) for p in CUBE_DIMS]),
        ('importance_nested_sampling', importance_sampling),
        ('multimodal', max_modes > 1),
        ('const_efficiency_mode', const_eff),
        ('n_live_points', n_live),
        ('evidence_tolerance', evidence_tol),
        ('sampling_efficiency', sampling_eff),
        ('null_log_evidence', -1e90),
        ('max_modes', max_modes),
        ('mode_tolerance', -1e90),
        ('seed', seed),
        ('log_zero', -1e100),
        ('max_iter', max_iter),
    ])

    mn_meta = OrderedDict([
        ('params', CUBE_DIMS),
        ('original_prior_specs', priors),
        ('priors_used', priors_used),
        ('time_window', time_window),
        ('kwargs', sort_dict(mn_kw)),
    ])

    outdir = expand(outdir)
    mkdir(outdir)

    out_prefix = join(outdir, 'evt{}-'.format(event_idx))
    print('Output files prefix: "{}"\n'.format(out_prefix))

    print('Runing MultiNest...')
    t0 = time.time()
    pymultinest.run(
        LogLikelihood=loglike,
        Prior=prior,
        verbose=True,
        outputfiles_basename=out_prefix,
        resume=False,
        write_output=False,
        n_iter_before_update=5000,
        **mn_kw
    )
    t1 = time.time()

    llhp = np.empty(shape=len(param_values), dtype=LLHP_T)
    llhp['llh'] = log_likelihoods
    llhp[list(HYPO_PARAMS_T._fields)] = param_values

    llhp_outf = out_prefix + 'llhp.npy'
    print('Saving llhp to "{}"...'.format(llhp_outf))
    np.save(llhp_outf, llhp)

    mn_meta['num_llhp'] = len(param_values)
    mn_meta['run_time'] = t1 - t0
    mn_meta_outf = out_prefix + 'multinest_meta.pkl'
    print('Saving MultiNest metadata to "{}"'.format(mn_meta_outf))
    pickle.dump(
        mn_meta,
        open(mn_meta_outf, 'wb'),
        protocol=pickle.HIGHEST_PROTOCOL
    )

    # DEBUG
    #table_indices_outf = out_prefix + 'table_indices.pkl'
    #pickle.dump(table_indices, open(table_indices_outf, 'wb'),
    #            protocol=pickle.HIGHEST_PROTOCOL)
    #t_indep_table_indices_outf = out_prefix + 't_indep_table_indices.pkl'
    #pickle.dump(t_indep_indices, open(t_indep_table_indices_outf, 'wb'),
    #            protocol=pickle.HIGHEST_PROTOCOL)

    return llhp, mn_meta


def reco(dom_tables_kw, hypo_kw, events_kw, reco_kw):
    """Script "main" function"""
    t00 = time.time()

    dom_tables = init_obj.setup_dom_tables(**dom_tables_kw)
    hypo_handler = init_obj.setup_discrete_hypo(**hypo_kw)
    events_iterator = init_obj.get_events(**events_kw)

    print('Running reconstructions...')

    spatial_prior_orig = reco_kw.pop('spatial_prior').strip()
    spatial_prior_name = spatial_prior_orig.lower()
    if spatial_prior_name == 'ic':
        x_prior = (PRI_UNIFORM, (-860, 870))
        y_prior = (PRI_UNIFORM, (-780, 770))
        z_prior = (PRI_UNIFORM, (-780, 790))
    elif spatial_prior_name == 'dc':
        x_prior = (PRI_UNIFORM, (-150, 270))
        y_prior = (PRI_UNIFORM, (-210, 150))
        z_prior = (PRI_UNIFORM, (-770, 760))
    elif spatial_prior_name == 'dc_subdust':
        x_prior = (PRI_UNIFORM, (-150, 270))
        y_prior = (PRI_UNIFORM, (-210, 150))
        z_prior = (PRI_UNIFORM, (-610, -60))
    elif spatial_prior_name == 'spefit2':
        x_prior = (
            PRI_SPEFIT2,
            (
                # scipy.stats.cauchy loc, scale parameters
                -0.19687812829978152, 14.282171566308806,
                # Hard limits
                -600, 750
            )
        )
        y_prior = (
            PRI_SPEFIT2,
            (
                # scipy.stats.cauchy loc, scale parameters
                -0.2393645701205161, 15.049528023495354,
                # Hard limits
                -750, 650
            )
        )
        z_prior = (
            PRI_SPEFIT2,
            (
                # scipy.stats.cauchy loc, scale parameters
                -5.9170661027492546, 12.089399308036718,
                # Hard limits
                -1200, 200
            )
        )
    else:
        raise ValueError('Spatial prior "{}" not recognized'
                         .format(spatial_prior_orig))

    temporal_prior_orig = reco_kw.pop('temporal_prior').strip()
    temporal_prior_name = temporal_prior_orig.lower()
    if temporal_prior_name == PRI_UNIFORM:
        time_prior = (PRI_UNIFORM, (-4e3, 0.0))
    elif temporal_prior_name == PRI_SPEFIT2:
        time_prior = (
            PRI_SPEFIT2,
            (
                # scipy.stats.cauchy loc (rel to SPEFit2 time), scale
                -82.631395081663754, 75.619895703067343,
                # Hard limits (relative to left, right edges of window,
                # respectively)
                -4e3, 0.0
            )
        )
    else:
        raise ValueError('Temporal prior "{}" not recognized'
                         .format(temporal_prior_orig))

    cscd_energy_prior_name = reco_kw.pop('cscd_energy_prior')
    cscd_energy_lims = reco_kw.pop('cscd_energy_lims')
    if cscd_energy_prior_name == PRI_UNIFORM:
        cscd_energy_prior = (PRI_UNIFORM, (np.min(cscd_energy_lims), np.max(cscd_energy_lims)))
    elif cscd_energy_prior_name == PRI_LOG_UNIFORM:
        cscd_energy_prior = (PRI_LOG_UNIFORM, (np.min(cscd_energy_lims), np.max(cscd_energy_lims)))
    elif cscd_energy_prior_name == PRI_LOG_NORMAL:
        cscd_energy_prior = (
            PRI_LOG_NORMAL,
            (
                # scipy.stats.lognorm 3 paramters
                0.96251341305506233, 0.4175592980195757, 17.543915051586644,
                # hard limits
                np.min(cscd_energy_lims), np.max(cscd_energy_lims)
            )
        )
    else:
        raise ValueError(str(cscd_energy_prior_name))

    trck_energy_prior_name = reco_kw.pop('trck_energy_prior')
    trck_energy_lims = reco_kw.pop('trck_energy_lims')
    if trck_energy_prior_name == PRI_UNIFORM:
        trck_energy_prior = (PRI_UNIFORM, (np.min(trck_energy_lims), np.max(trck_energy_lims)))
    elif trck_energy_prior_name == PRI_LOG_UNIFORM:
        trck_energy_prior = (PRI_LOG_UNIFORM, (np.min(trck_energy_lims), np.max(trck_energy_lims)))
    elif trck_energy_prior_name == PRI_LOG_NORMAL:
        trck_energy_prior = (
            PRI_LOG_NORMAL,
            (
                # scipy.stats.lognorm 3 paramters
                0.96251341305506233, 0.4175592980195757, 17.543915051586644,
                # hard limits
                np.min(trck_energy_lims), np.max(trck_energy_lims)
            )
        )
    else:
        raise ValueError(str(trck_energy_prior_name))

    priors = OrderedDict([
        ('time', time_prior),
        ('x', x_prior),
        ('y', y_prior),
        ('z', z_prior),
        ('track_zenith', (PRI_COSINE, (-1, 1))),
        ('track_azimuth', (PRI_UNIFORM, (0, TWO_PI))),
        ('cascade_energy', cscd_energy_prior),
        ('track_energy', trck_energy_prior),
        ('track_fraction', (PRI_UNIFORM, (0, 1)))
    ])
    if HYPO_PARAMS_T is HypoParams10D:
        priors['cascade_zenith'] = (PRI_COSINE, (-1, 1))
        priors['cascade_azimuth'] = (PRI_UNIFORM, (0, TWO_PI))

    for event_idx, event in events_iterator: # pylint: disable=unused-variable
        t1 = time.time()
        if 'mc_truth' in event:
            print(event['mc_truth'])
        llhp, _ = run_multinest(
            event_idx=event_idx,
            event=event,
            dom_tables=dom_tables,
            hypo_handler=hypo_handler,
            priors=priors,
            **reco_kw
        )
        dt = time.time() - t1
        n_points = llhp.size
        print('  ---> {:.3f} s, {:d} points ({:.3f} ms per LLH)'
              .format(dt, n_points, dt / n_points * 1e3))

    print('Total script run time is {:.3f} s'.format(time.time() - t00))


def parse_args(description=__doc__):
    """Parse command-line arguments.

    Returns
    -------
    split_kwargs : dict of dicts
        Contains keys "dom_tables_kw", "hypo_kw", "events_kw", and "reco_kw",
        where values are kwargs dicts usable to instantiate or call each of the
        corresponding objects or functions.

    """
    parser = ArgumentParser(description=description)

    parser.add_argument(
        '--outdir', required=True
    )

    group = parser.add_argument_group(
        title='Hypothesis parameter priors',
    )

    group.add_argument(
        '--spatial-prior',
        choices='dc dc_subdust ic SPEFit2'.split(),
        required=True,
        help='''Choose a prior for choosing spatial samples. "dc", "dc_subdust"
        and "ic" are uniform priors with hard cut-offs at the extents of the
        respective volumes, while "SPEFit2" samples from Cauchy distributions
        around the SPEFit2 (x, y, z) best-fit values.'''
    )
    group.add_argument(
        '--temporal-prior',
        choices='uniform SPEFit2'.split(),
        required=True,
        help='''Choose a prior for choosing temporal samples. "uniform" chooses
        uniformly from 4000 ns prior to the first hit up to the last hit, while
        "SPEFit2" samples from a Cauchy distribution near (not *at* due to
        bias) the SPEFit2 time best-fit value.'''
    )
    group.add_argument(
        '--cscd-energy-prior',
        choices=[PRI_UNIFORM, PRI_LOG_UNIFORM, PRI_LOG_NORMAL],
        required=True,
        help='''Prior to put on cascade event energy. Must specify
        --cscd-energy-lims.'''
    )
    group.add_argument(
        '--trck-energy-prior',
        choices=[PRI_UNIFORM, PRI_LOG_UNIFORM, PRI_LOG_NORMAL],
        required=True,
        help='''Prior to put on track event energy. Must specify
        --trck-energy-lims.'''
    )
    group.add_argument(
        '--cscd-energy-lims', nargs='+',
        required=True,
        help='''Lower and upper energy limits, in GeV. E.g.: --cscd-energy-lims=1,100
        Required if --cscd-energy-prior is {}, {}, or {}'''
        .format(PRI_UNIFORM, PRI_LOG_UNIFORM, PRI_LOG_NORMAL)
    )
    group.add_argument(
        '--trck-energy-lims', nargs='+',
        required=True,
        help='''Lower and upper energy limits, in GeV. E.g.: --trck-energy-lims=1,100
        Required if --trck-energy-prior is {}, {}, or {}'''
        .format(PRI_UNIFORM, PRI_LOG_UNIFORM, PRI_LOG_NORMAL)
    )

    group = parser.add_argument_group(
        title='MultiNest parameters',
    )

    group.add_argument(
        '--importance-sampling', action='store_true',
        help='''Importance nested sampling (INS) mode. Could be more efficient,
        but also can be unstable. Does not work with multimodal.'''
    )
    group.add_argument(
        '--max-modes', type=int, required=True,
        help='''Set to 1 to disable multi-modal search. Must be 1 if --importance-sampling is
        specified.'''
    )
    group.add_argument(
        '--const-eff', action='store_true',
        help='''Constant efficiency mode.'''
    )
    group.add_argument(
        '--n-live', type=int, required=True
    )
    group.add_argument(
        '--evidence-tol', type=float, required=True
    )
    group.add_argument(
        '--sampling-eff', type=float, required=True
    )
    group.add_argument(
        '--max-iter', type=int, required=True,
        help='''Note that iterations of the MultiNest algorithm are _not_ the
        number of likelihood evaluations. An iteration comes when one live
        point is discarded by finding a sample with higher likelihood than at
        least one other live point. Such a point can take many likelihood
        evaluatsions to find.'''
    )
    group.add_argument(
        '--seed', type=int, required=True,
        help='''Integer seed for MultiNest's random number generator.'''
    )

    split_kwargs = init_obj.parse_args(
        dom_tables=True, hypo=True, events=True, parser=parser
    )

    split_kwargs['reco_kw'] = reco_kw = split_kwargs.pop('other_kw')

    if reco_kw['cscd_energy_prior'] in [PRI_UNIFORM, PRI_LOG_UNIFORM, PRI_LOG_NORMAL]:
        assert reco_kw['cscd_energy_lims'] is not None
        cscd_elims = ''.join(reco_kw['cscd_energy_lims'])
        cscd_elims = [float(l) for l in cscd_elims.split(',')]
        reco_kw['cscd_energy_lims'] = cscd_elims
    elif reco_kw['cscd_energy_lims'] is not None:
        raise ValueError('--cscd-energy-limits not used with energy prior {}'
                         .format(reco_kw['cscd_energy_prior']))
    if reco_kw['trck_energy_prior'] in [PRI_UNIFORM, PRI_LOG_UNIFORM, PRI_LOG_NORMAL]:
        assert reco_kw['trck_energy_lims'] is not None
        trck_elims = ''.join(reco_kw['trck_energy_lims'])
        trck_elims = [float(l) for l in trck_elims.split(',')]
        reco_kw['trck_energy_lims'] = trck_elims
    elif reco_kw['trck_energy_lims'] is not None:
        raise ValueError('--trck-energy-limits not used with energy prior {}'
                         .format(reco_kw['trck_energy_prior']))

    return split_kwargs


if __name__ == '__main__':
    reco(**parse_args())
