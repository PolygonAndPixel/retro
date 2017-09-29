#!/usr/bin/env python
# coding: utf-8
# pylint: disable=wrong-import-position, invalid-name

"""
Create time- and DOM-independent "whole-detector" Cartesian-binned retro table.

The generated table is useful for computing the total charge expected to be
deposited by a hypothesis across the entire detector (i.e., independent of time
and DOM).

Define a Cartesian grid that covers all of the IceCube fiducial volume, then
tabulate for each voxel the survival probability for photons coming from any
DOM at any time to reach that voxel. Also, tabulate the "average surviving
photon," defined by its x, y, and z components (which differs from the original
time- and DOM-dependent retro tables, wherein length, theta, and deltaphi are
used to characterize the average surviving photon).

Note that the length of the average surviving photon vector can be interpreted
as a measure of the directionality required for a photon to reach a DOM. I.e.,
if its length is 1, then only photons going exactly opposite that direction
will make it to a DOM (to within statistical and bin-size uncertainties used to
arrive at the average photon. If the length is _less_ than 1, then other
directions besides the average photon direction will be accepted, with
increasing likelihood as that length decreases towards 0.

The new table is in (x, y, z)--independent of time and DOM--and can be used to
scale the photons expected to reach any DOM at any time due to a hypothesis
that generates some number of photons (with an average direction / length) in
any of the voxel(s) of this table.
"""


from __future__ import absolute_import, division, print_function

from argparse import ArgumentParser
from collections import OrderedDict
from copy import deepcopy
import cPickle as pickle
import os
from os.path import abspath, dirname, isdir, isfile, join
import re
import sys
import time

import numpy as np
import pyfits

os.sys.path.append(dirname(dirname(abspath('__file__'))))
from retro import (IC_QUANT_EFF, DC_QUANT_EFF, POL_TABLE_NRBINS,
                   POL_TABLE_NTBINS, POL_TABLE_NTHETABINS, POL_TABLE_RMAX,
                   POL_TABLE_RPWR)
from retro import powerspace, spherical_volume, extract_photon_info
from retro.shift_and_bin import shift_and_bin
from retro.sphbin2cartbin import sphbin2cartbin

from pisa.utils.hash import hash_obj
from pisa.utils.format import hrlist2list, list2hrlist
from pisa.utils.timing import timediffstamp


TDI_TABLE_FNAME_PROTO = (
    'retro_tdi_table'
    '_{tdi_hash:s}'
    '_binmap_{binmap_hash:s}'
    '_geom_{geom_hash:s}'
    '_domtbl_{dom_tables_hash:s}'
    '_times_{times_str:s}'
    '_x{x_min:.3f}_{x_max:.3f}'
    '_y{y_min:.3f}_{y_max:.3f}'
    '_z{z_min:.3f}_{z_max:.3f}'
    '_bw{binwidth:.9f}'
    '_anisot_{anisotropy_str:s}'
    '_ics{ic_scale:.5f}'
    '_dcs{dc_scale:.5f}'
    '_{table_name:s}'
    '.fits'
)

TDI_TABLE_FNAME_RE = re.compile(
    r'^retro_tdi_table'
    r'_(?P<tdi_hash>[^_]+)'
    r'_binmap_(?P<binmap_hash>[^_]+)'
    r'_geom_(?P<geom_hash>[^_]+)'
    r'_domtbl_(?P<dom_tables_hash>[^_]+)'
    r'_times_(?P<times_str>[^_]+)'
    r'_x(?P<x_min>[^_]+)_(?P<x_max>[^_]+)'
    r'_y(?P<y_min>[^_]+)_(?P<y_max>[^_]+)'
    r'_z(?P<z_min>[^_]+)_(?P<z_max>[^_]+)'
    r'_bw(?P<binwidth>[^_]+)'
    r'_anisot_(?P<anisotropy>.+?)'
    r'_ics(?P<ic_scale>.+?)'
    r'_dcs(?P<dc_scale>.+?)'
    r'_(?P<table_name>(avg_photon_x|avg_photon_y|avg_photon_z|survival_prob))'
    r'\.fits$'
    , re.IGNORECASE
)


def get_anisotropy_str(anisotropy):
    if anisotropy is None:
        anisotropy_str = 'none'
    else:
        anisotropy_str = '_'.join(str(param) for param in anisotropy)
    return anisotropy_str


# TODO: does anisotropy need to be considered here?
def get_binmap_meta(r_max, r_power, n_rbins, n_costhetabins, n_phibins,
                    cart_binwidth, oversample, antialias):
    """Get metadata for spherical to Cartesian bin mapping, including the file
    name, hash string, and a dict with all of the parameters that contributed
    to these which can be passed via ``**binmap_kw`` to the `sphbin2cartbin`
    function.

    Parameters
    ----------
    r_max : float
        Maximum radius in Retro (t,r,theta)-binned DOM table (meters)

    r_power : float
        Binning in radial direction is regular in the inverse of this power.
        I.e., every element of `np.diff(r**(1/r_power))` is equal.

    n_rbins, n_costhetabins, n_phibins : int

    cart_binwidth : float
        Cartesian bin widths, same in x, y, and z (meters)

    oversample : int
        Oversample factor, same in x, y, and z

    antialias : int
        Antialias factor

    Returns
    -------
    metadata : dict
        Contains following items:
            'fname' : string
                File name for the specified bin mapping
            'hash' : length-8 string
                Hex digits represented as a string.
            'kwargs' : OrderedDict
                The keyword args used for the hash.

    """
    kwargs = OrderedDict([
        ('r_max', r_max),
        ('r_power', r_power),
        ('n_rbins', n_rbins),
        ('n_costhetabins', n_costhetabins),
        ('n_phibins', n_phibins),
        ('cart_binwidth', cart_binwidth),
        ('oversample', oversample),
        ('antialias', antialias)
    ])

    binmap_hash = hash_obj(kwargs, hash_to='hex', full_hash=True)

    print('kwargs:', kwargs)

    fname = (
        'sph2cart_binmap'
        '_%s'
        '_nr{n_rbins:d}_ncostheta{n_costhetabins:d}_nphi{n_phibins:d}'
        '_rmax{r_max:f}_rpwr{r_power}'
        '_bw{cart_binwidth:.6f}'
        '_os{oversample:d}'
        '_aa{antialias:d}'
        '.pkl'.format(**kwargs)
    ) % binmap_hash

    metadata = OrderedDict([
        ('fname', fname),
        ('hash', binmap_hash),
        ('kwargs', kwargs)
    ])

    return metadata


def get_geom_meta(geom):
    """Hash on geometry. Note that the values are rounded to the nearest
    centimeter for hashing purposes. (Also, the values are converted to
    integers at this precision to eliminate any possible float32 / float64
    issues that could cause discrepancies in hash values for what we consider
    to be equal geometries.)

    Parameters
    ----------
    geom : shape (n_strings, n_depths, 3) numpy ndarray, dtype float{32,64}

    Returns
    -------
    metadata : OrderedDict
        Contains the item:
            'hash' : length-8 str
                Hex characters convert to a string of length 8

    """
    assert len(geom.shape) == 3
    assert geom.shape[2] == 3
    rounded_ints = np.round(geom * 100).astype(np.int)
    geom_hash = hash_obj(rounded_ints, hash_to='hex', full_hash=True)
    return OrderedDict([('hash', geom_hash)])


def get_tdi_table_meta(binmap_hash, geom_hash, dom_tables_hash, times_str,
                       x_min, x_max, y_min, y_max, z_min, z_max, binwidth,
                       anisotropy, ic_scale, dc_scale):
    """Get metadata for a time- and DOM-independent Cartesian (x, y, z)-binned
    table.

    Parameters
    ----------
    binmap_hash : string
    geom_hash : string
    dom_tables_hash : string
    times_str : string
    x_lims, y_lims, z_lims : 2-tuples of floats
    binwidth : float
    anisotropy : None or tuple
    ic_scale : float
    dc_scale : float

    Returns
    -------
    metadata : dict
        Contains keys
            'fbasename' : string
            'hash' : string
            'kwargs' : OrderedDict

    """
    if dom_tables_hash is None:
        dom_tables_hash = 'none'

    kwargs = OrderedDict([
        ('geom_hash', geom_hash),
        ('binmap_hash', binmap_hash),
        ('dom_tables_hash', dom_tables_hash),
        ('times_str', times_str),
        ('x_min', x_min),
        ('x_max', x_max),
        ('y_min', y_min),
        ('y_max', y_max),
        ('z_min', z_min),
        ('z_max', z_max),
        ('binwidth', binwidth),
        ('anisotropy', anisotropy),
        ('ic_scale', ic_scale),
        ('dc_scale', dc_scale)
    ])

    hash_params = deepcopy(kwargs)
    for param in ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max']:
        rounded_int = int(np.round(hash_params[param]*100))
        hash_params[param] = rounded_int
        kwargs[param] = float(rounded_int) / 100
    for param in ['ic_scale', 'dc_scale']:
        rounded_int = int(np.round(hash_params[param]*10000))
        hash_params[param] = rounded_int
        kwargs[param] = float(rounded_int) / 10000
    hash_params['binwidth'] = int(np.round(hash_params['binwidth'] * 1e10))
    tdi_hash = hash_obj(hash_params, hash_to='hex', full_hash=True)

    anisotropy_str = get_anisotropy_str(anisotropy)
    fname = TDI_TABLE_FNAME_PROTO.format(
        tdi_hash=tdi_hash,
        anisotropy_str=anisotropy_str,
        table_name='',
        **kwargs
    )
    fbasename = fname.rsplit('_.fits')[0]

    metadata = OrderedDict([
        ('fbasename', fbasename),
        ('hash', tdi_hash),
        ('kwargs', kwargs)
    ])

    return metadata


# TODO: does anisotropy need to be considered here?
def generate_binmap(r_max, r_power, n_rbins, n_costhetabins, n_phibins,
                    cart_binwidth, oversample, antialias, tables_dir,
                    recompute):
    """Generate mapping from polar binning (assumed to be symmetric about
    Z-axis) to Cartesian 3D binning.

    The heart of the functionality is implemented in
    `retro.sphbin2cartbin.sphbin2cartbin`, while this function implements
    loading already-computed mappings and storing the results to disk.

    Parameters
    ----------
    r_max : float > 0
    r_power : float != 0
    n_rbins, n_costhetabins, n_phibins : int >= 1
    cart_binwidth : float > 0
    oversample : int >= 1
    antialias : int between 1 and 50
    tables_dir : string
    recompute : bool

    Returns
    -------
    ind_arrays
    vol_arrays
    meta
        Output of `get_binmap_meta`

    """
    assert isdir(tables_dir)
    r_edges = powerspace(0, r_max, n_rbins + 1, r_power)
    theta_edges = np.arccos(np.linspace(1, -1, n_costhetabins + 1))

    R, THETA = np.meshgrid(r_edges, theta_edges, indexing='ij')
    exact_vols = []
    for ri in range(n_rbins):
        sub_exact_vols = []
        for ti in range(int(np.ceil(n_costhetabins / 2.0))):
            rs = R[ri:ri+2, ti:ti+2]
            ts = THETA[ri:ri+2, ti:ti+2]
            dcostheta = np.abs(np.diff(np.cos([ts.max(), ts.min()])))
            exact_vol = spherical_volume(rmin=rs.max(), rmax=rs.min(),
                                         dcostheta=dcostheta, dphi=np.pi/2)
            sub_exact_vols.append(exact_vol)
        exact_vols.append(sub_exact_vols)
    exact_vols = np.array(exact_vols)

    meta = get_binmap_meta(
        r_max=r_max, r_power=r_power,
        n_rbins=n_rbins, n_costhetabins=n_costhetabins, n_phibins=n_phibins,
        cart_binwidth=cart_binwidth, oversample=oversample, antialias=antialias
    )
    fpath = join(tables_dir, meta['fname'])

    print('Binmap kwargs:', meta['kwargs'])

    if not recompute and isfile(fpath):
        sys.stdout.write('Loading binmap from file\n  "%s"\n' % fpath)
        sys.stdout.flush()

        t0 = time.time()
        data = pickle.load(file(fpath, 'rb'))
        ind_arrays = data['ind_arrays']
        vol_arrays = data['vol_arrays']
        t1 = time.time()
        print('  Time to load bin mapping from pickle:', timediffstamp(t1 - t0))

    else:
        sys.stdout.write('  Computing bin mapping...\n')
        sys.stdout.flush()

        t0 = time.time()
        ind_arrays, vol_arrays = sphbin2cartbin(**meta['kwargs'])
        t1 = time.time()
        print('    Time to compute bin mapping:', timediffstamp(t1 - t0))

        print('  Writing bin mapping to pickle file\n  "%s"' % fpath)
        data = OrderedDict([
            ('kwargs', meta['kwargs']),
            ('ind_arrays', ind_arrays),
            ('vol_arrays', vol_arrays)
        ])
        pickle.dump(data, file(fpath, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        t2 = time.time()
        print('    Time to pickle bin mapping:', timediffstamp(t2 - t1))
    print('')

    binned_vol = np.sum([va.sum() for va in vol_arrays])
    exact_vol = spherical_volume(rmin=0, rmax=r_max, dcostheta=-1, dphi=np.pi/2)
    print('  Exact vol = %f, binned vol = %f (%e fract error)'
          % (exact_vol, binned_vol, (binned_vol-exact_vol)/exact_vol))

    ind_bin_vols = np.array([va.sum() for va in vol_arrays])
    fract_err = ind_bin_vols/exact_vols.flat - 1
    abs_fract_err = np.abs(fract_err)
    worst_abs_fract_err = np.max(abs_fract_err)
    flat_idx = np.where(abs_fract_err == worst_abs_fract_err)[0][0]
    r_idx, costheta_idx = divmod(flat_idx, int(np.ceil(n_costhetabins/2)))
    print('  Worst single-bin fract err: %e;'
          'r_idx=%d, costheta_idx=%d;'
          'binned vol=%e, exact vol=%e'
          % (worst_abs_fract_err, r_idx, costheta_idx, ind_bin_vols[flat_idx],
             exact_vols[r_idx, costheta_idx]))

    return ind_arrays, vol_arrays, meta


def generate_tdi_table(tables_dir, geom_fpath, dom_tables_hash, n_phibins,
                       x_lims, y_lims, z_lims,
                       binwidth, oversample, antialias, anisotropy,
                       ic_scale, dc_scale,
                       strings=slice(None),
                       depths=slice(None),
                       times=slice(None),
                       recompute_binmap=False,
                       recompute_table=False):
    """Create a time- and DOM-independent Cartesian (x,y,z)-binned Retro
    tables (if it doesn't already exist or if the user requests that it be
    re-computed) and saves the tables to disk.

    The intermediate step of computing a bin mapping from polar (r, theta)
    coordinates for the source (t, r, theta)-binned DOM Retro tables is also
    performed if it hasn't already been saved to disk or if the user forces
    its recomputation; the result is stored to disk for future use.


    Parameters
    ----------
    tables_dir
    geom_fpath
    dom_tables_hash
    n_phibins : int
    x_lims, y_lims, z_lims : 2-tuples of floats
    binwidth : float
    oversample : int
    antialias : int
    anisotropy : int
    ic_scale : float
    dc_scale : float
    strings : int, sequence, slice
        Select only these strings by indexing into the geom array

    depths : int, sequence, slice
        Select only these depth indices by indexing into the geom array

    times : int, sequence, slice
        Sum over only these times

    recompute_binmap : bool
        Force recomputation of bin mapping even if it already exists; existing
        file will be overwritten

    recompute_table : bool
        Force recomputation of table files even if the already exist; existing
        files will be overwritten


    Returns
    -------
    tdi_data : OrderedDict
        Contains following items:
            'binned_sp : shape (nx,ny,nz) numpy ndarray, dtype float32
                Survival probability table
            'binned_px' : shape (nx,ny,nz) numpy ndarray, dtype float32
            'binned_py' : shape (nx,ny,nz) numpy ndarray, dtype float32
            'binned_pz' : shape (nx,ny,nz) numpy ndarray, dtype float32
                Tables with average photon directionality, one each for x, y,
                and z components, respectively
            'ind_arrays'
            'vol_arrays'
            'tdi_meta' : OrderedDict
                Return value from `get_tdi_table_meta`
            'binmap_meta' : OrderedDict
                Return value from `get_binmap_meta`

    """
    assert isdir(tables_dir)
    if dom_tables_hash is None:
        dom_tables_hash = 'none'
        r_max = POL_TABLE_RMAX
        r_power = POL_TABLE_RPWR
        n_rbins = POL_TABLE_NRBINS
        n_costhetabins = POL_TABLE_NTHETABINS
        n_tbins = POL_TABLE_NTBINS
    else:
        raise ValueError('Cannot handle non-None `dom_tables_hash`')

    nx = int(np.round((x_lims[1] - x_lims[0]) / binwidth))
    ny = int(np.round((y_lims[1] - y_lims[0]) / binwidth))
    nz = int(np.round((z_lims[1] - z_lims[0]) / binwidth))
    assert np.abs(x_lims[0] + nx * binwidth - x_lims[1]) < 1e-6
    assert np.abs(y_lims[0] + ny * binwidth - y_lims[1]) < 1e-6
    assert np.abs(z_lims[0] + nz * binwidth - z_lims[1]) < 1e-6

    xyz_shape = (nx, ny, nz)
    print('Generated/loaded TDI Cart table will have shape:', xyz_shape)
    print('')

    geom = np.load(geom_fpath)

    depth_indices = np.atleast_1d(np.arange(60)[depths])
    string_indices = np.atleast_1d(np.arange(87)[strings]) - 1
    string_indices = string_indices[string_indices >= 0]

    subdet_doms = {'ic': [], 'dc': []}
    dc_strings = list(range(79, 86))
    for string_idx in string_indices:
        dom_coords = geom[string_idx:string_idx+1, depths, :]
        if string_idx in dc_strings:
            subdet_doms['dc'].append(dom_coords)
        else:
            subdet_doms['ic'].append(dom_coords)
    for subdet in subdet_doms.keys():
        dom_string_list = subdet_doms[subdet]
        if not dom_string_list:
            subdet_doms.pop(subdet)
        else:
            subdet_doms[subdet] = np.concatenate(dom_string_list, axis=0)
    geom = geom[string_indices, :, :][:, depth_indices, :]
    geom_meta = get_geom_meta(geom)
    print('Geom uses strings %s, depth indices %s for a total of %d DOMs'
          % (list2hrlist([i+1 for i in string_indices]),
             list2hrlist(depth_indices),
             geom.shape[0] * geom.shape[1]))
    print('')

    ind_arrays, vol_arrays, binmap_meta = generate_binmap(
        r_max=r_max, r_power=r_power,
        n_rbins=n_rbins, n_costhetabins=n_costhetabins, n_phibins=n_phibins,
        cart_binwidth=binwidth, oversample=oversample, antialias=antialias,
        tables_dir=tables_dir, recompute=recompute_binmap
    )
    print('')

    # Figure out which time bin(s) to use to reduce source (t,r,theta) tables
    # along time axis (where reduction is one minus product of one minus
    # survival probabilities and average photon directionality)
    all_t_bins = list(range(n_tbins))
    remaining_t_bins = np.array(all_t_bins)[times].tolist()
    if all_t_bins == remaining_t_bins:
        times_str = 'all'
    else:
        times_str = list2hrlist(remaining_t_bins)

    print('Marginalizing over times in source (t,r,theta) DOM Retro tables:',
          times_str)
    print('')

    tdi_meta = get_tdi_table_meta(
        binmap_hash=binmap_meta['hash'],
        geom_hash=geom_meta['hash'],
        dom_tables_hash=None, # TODO: hash for dom tables not yet implemented
        times_str=times_str,
        x_min=x_lims[0], x_max=x_lims[1],
        y_min=y_lims[0], y_max=y_lims[1],
        z_min=z_lims[0], z_max=z_lims[1],
        binwidth=binwidth, anisotropy=anisotropy,
        ic_scale=ic_scale, dc_scale=dc_scale
    )

    print('Generating Cartesian time- and DOM-independent (TDI) Retro table')
    print('tdi_kw:', tdi_meta['kwargs'])

    names = [
        'survival_prob',
        'avg_photon_x',
        'avg_photon_y',
        'avg_photon_z'
    ]
    if not recompute_table:
        for name in names:
            fpath = join(tables_dir,
                         '%s_%s.fits' % (tdi_meta['fbasename'], name))
            if not isfile(fpath):
                print('  Could not find table, will (re)compute\n%s\n' % fpath)
                recompute_table = True
                break

    if not recompute_table:
        print('  Loading (x,y,z)-binned TDI Retro table from disk')
        for name in names:
            fpath = join(tables_dir,
                         tdi_meta['fbasename'] + '_' + name + '.fits')
            with pyfits.open(fpath) as fits_file:
                tmp = fits_file[0].data
            if name == 'survival_prob':
                binned_sp = tmp
            elif name == 'avg_photon_x':
                binned_px = tmp
            elif name == 'avg_photon_y':
                binned_py = tmp
            elif name == 'avg_photon_z':
                binned_pz = tmp
            del tmp
        tdi_data = OrderedDict([
            ('binned_sp', binned_sp),
            ('binned_px', binned_px),
            ('binned_py', binned_py),
            ('binned_pz', binned_pz),
            ('ind_arrays', ind_arrays),
            ('vol_arrays', vol_arrays),
            ('tdi_meta', tdi_meta),
            ('binmap_meta', binmap_meta)
        ])
        return tdi_data

    # Instantiate arrays for aggregation of survival probabilities and
    # averaging photon direction per Cartesian bin. Note that these start as 1D
    # to speed indexing operations, then are reshaped into 3D at the end.
    binned_spv = np.zeros((nx*ny*nz), dtype=np.float64)
    binned_px_spv = np.zeros((nx*ny*nz), dtype=np.float64)
    binned_py_spv = np.zeros((nx*ny*nz), dtype=np.float64)
    binned_pz_spv = np.zeros((nx*ny*nz), dtype=np.float64)
    binned_one_minus_sp = np.ones((nx*ny*nz), dtype=np.float64)

    t00 = time.time()
    for subdet, subdet_dom_coords in subdet_doms.items():
        print('  Subdetector:', subdet)
        print('  -> %d strings with DOM(s) at %d depths'
              % (subdet_dom_coords.shape[0], subdet_dom_coords.shape[1]))
        print('')

        if subdet == 'ic':
            qe_scale = ic_scale
        elif subdet == 'dc':
            qe_scale = dc_scale
        else:
            raise ValueError(str(subdet))

        for rel_idx, depth_idx in enumerate(depth_indices):
            print('    Subdetector: %s, depth_idx: %d' % (subdet, depth_idx))
            dom_coords = subdet_dom_coords[:, rel_idx, :]

            t0 = time.time()
            table_fname = (
                'retro_nevts1000'
                '_{subdet:s}'
                '_DOM{depth_idx:d}'
                '_r_cz_t_angles'
                '.fits'.format(
                    subdet=subdet.upper(), depth_idx=depth_idx
                )
            )
            # TODO: validate that bin edges match spec we're using
            photon_info, _ = extract_photon_info(
                fpath=join(tables_dir, table_fname),
                depth_idx=depth_idx, scale=qe_scale
            )
            t1 = time.time()
            print('    Time to load Retro DOM table:', timediffstamp(t1 - t0))

            sp = photon_info.survival_prob[depth_idx].astype(np.float64)
            plength = photon_info.length[depth_idx].astype(np.float64)
            ptheta = photon_info.theta[depth_idx].astype(np.float64)
            pdeltaphi = photon_info.deltaphi[depth_idx].astype(np.float64)

            plength *= np.cos(pdeltaphi)
            pz = plength * np.cos(ptheta)
            prho = plength * np.sin(ptheta)

            # Marginalize out time, computing the probability of a photon
            # starting at any one time being detected at any other time
            t_indep_sp = 1 - np.prod(1 - sp[times], axis=0)

            mask = t_indep_sp != 0
            scale = 1 / sp.sum(axis=0)[mask]

            t_indep_pz = np.zeros_like(t_indep_sp)
            t_indep_prho = np.zeros_like(t_indep_sp)

            t_indep_pz[mask] = (pz[times] * sp[times]).sum(axis=0)[mask] * scale
            t_indep_prho[mask] = (prho[times] * sp[times]).sum(axis=0)[mask] * scale

            t2 = time.time()
            print("    Time to reduce Retro DOM table's time dimension:",
                  timediffstamp(t2 - t1))

            shift_and_bin(
                ind_arrays=ind_arrays,
                vol_arrays=vol_arrays,
                dom_coords=dom_coords,
                survival_prob=t_indep_sp,
                prho=t_indep_prho,
                pz=t_indep_pz,
                nr=n_rbins,
                ntheta=n_costhetabins,
                r_max=r_max,
                binned_spv=binned_spv,
                binned_px_spv=binned_px_spv,
                binned_py_spv=binned_py_spv,
                binned_pz_spv=binned_pz_spv,
                binned_one_minus_sp=binned_one_minus_sp,
                x_min=x_lims[0],
                y_min=y_lims[0],
                z_min=z_lims[0],
                x_max=x_lims[1],
                y_max=y_lims[1],
                z_max=z_lims[1],
                binwidth=binwidth,
                oversample=oversample,
                anisotropy=None
            )
            print('    %d surv probs are exactly 1' % np.sum(binned_one_minus_sp == 0))
            t3 = time.time()
            print('    Time to shift and bin:', timediffstamp(t3 - t2))
            print('')

    print('Total time to shift and bin:', timediffstamp(t3 - t00))
    print('')

    binned_sp = 1.0 - binned_one_minus_sp
    binned_sp = binned_sp.astype(np.float32).reshape(xyz_shape)
    del binned_one_minus_sp

    mask = binned_spv != 0
    binned_px_spv[mask] /= binned_spv[mask]
    binned_py_spv[mask] /= binned_spv[mask]
    binned_pz_spv[mask] /= binned_spv[mask]
    del mask

    # Rename so as to not mislead
    binned_px = binned_px_spv.astype(np.float32).reshape(xyz_shape)
    binned_py = binned_py_spv.astype(np.float32).reshape(xyz_shape)
    binned_pz = binned_pz_spv.astype(np.float32).reshape(xyz_shape)
    del binned_px_spv, binned_py_spv, binned_pz_spv

    t4 = time.time()
    print('Time to normalize histograms:', timediffstamp(t4 - t3))
    print('')

    arrays_names = [
        (binned_sp, 'survival_prob'),
        (binned_px, 'avg_photon_x'),
        (binned_py, 'avg_photon_y'),
        (binned_pz, 'avg_photon_z')
    ]
    for array, name in arrays_names:
        fname = '%s_%s.fits' % (tdi_meta['fbasename'], name)
        fpath = join(tables_dir, fname)
        hdulist = pyfits.HDUList([
            pyfits.PrimaryHDU(array.astype(np.float32)),
            pyfits.ImageHDU(xyz_shape),
            pyfits.ImageHDU(np.array([x_lims, y_lims, z_lims])),
            pyfits.ImageHDU(geom)
        ])
        print('Saving %s to file\n%s\n' % (name, fpath))
        hdulist.writeto(fpath, clobber=True)
    t5 = time.time()
    print('Time to save tables to disk:', timediffstamp(t5 - t4))
    print('')

    print('TOTAL RUN TIME:', timediffstamp(t5 - t00))

    tdi_data = OrderedDict([
        ('binned_sp', binned_sp),
        ('binned_px', binned_px),
        ('binned_py', binned_py),
        ('binned_pz', binned_pz),
        ('ind_arrays', ind_arrays),
        ('vol_arrays', vol_arrays),
        ('tdi_meta', tdi_meta),
        ('binmap_meta', binmap_meta)
    ])
    return tdi_data


def parse_args(description=__doc__):
    """Parse command line args"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--tables-dir', required=True,
        help='Path to eirectory containing Retro tables'
    )
    parser.add_argument(
        '--geom-fpath', required=True,
        help='Path to geometry NPY file'
    )
    parser.add_argument(
        '--dom-tables-hash', default=None,
        help='Hash ID for source (t,r,theta)-binned DOM Retro tables'
    )
    # TODO: all of the following should be known by passing the hash
    #parser.add_argument(
    #    '--t-max', type=float,
    #    help='''Maximum time bin edge in the source (t,r,theta)-binnned DOM
    #    Retro tables (nanoseconds)'''
    #)
    #parser.add_argument(
    #    '--r-max', type=float,
    #    help='''Maximum radial bin edge in the source (t,r,theta)-binnned DOM
    #    Retro tables (meters)'''
    #)
    #parser.add_argument(
    #    '--r-power', type=float,
    #    help='''Power used for radial power-law binning in source
    #    (t,r,theta)-binned DOM Retro tables'''
    #)
    #parser.add_argument(
    #    '--n-rbins', type=int,
    #    help='''Number of radial bins used in source (t,r,theta)-binned DOM
    #    Retro tables'''
    #)
    #parser.add_argument(
    #    '--n-costhetabins', type=int,
    #    help='''Number of costheta bins used in source (t,r,theta)-binned DOM
    #    Retro tables'''
    #)
    #parser.add_argument(
    #    '--n-tbins', type=int,
    #    help='''Number of time bins used in source (t,r,theta)-binned DOM Retro
    #    tables'''
    #)
    parser.add_argument(
        '--n-phibins', type=int, required=True,
        help='''Number of phi bins to use for rotating the (r,theta) tables
        about the z-axis to for effectively spherical tables'''
    )
    parser.add_argument(
        '--x-lims', nargs=2, type=float, required=True,
        help='''Limits of the produced table in the x-direction (meters)'''
    )
    parser.add_argument(
        '--y-lims', nargs=2, type=float, required=True,
        help='''Limits of the produced table in the y-direction (meters)'''
    )
    parser.add_argument(
        '--z-lims', nargs=2, type=float, required=True,
        help='''Limits of the produced table in the z-direction (meters)'''
    )
    parser.add_argument(
        '--binwidth', type=float, required=True,
        help='''Binwidth in x, y, and z directions (meters). Must divide each
        of --x-lims, --y-lims, and --z-lims into an integral number of bins.'''
    )
    parser.add_argument(
        '--oversample', type=int, required=True,
        help='''Oversampling factor in the x-, y-, and z- directions (int >=
        1).'''
    )
    parser.add_argument(
        '--antialias', type=int, required=True,
        help='''Antialiasing factor (int between 1 and 50).'''
    )
    parser.add_argument(
        '--anisotropy', nargs='+', metavar='ANISOT_PARAM', required=False,
        default=None,
        help='''[NOT IMPLEMENTED] Simple ice anisotropy parameters to use: DIR
        for azimuthal direction of low-scattering axis (radians) and MAG for
        magnitude of anisotropy (unitless). If not specified, no anisotropy is
        modeled.'''
    )
    parser.add_argument(
        '--ic-scale', type=float, default=IC_QUANT_EFF,
        help='''IceCube (non-DeepCore) DOM scaling factor (e.g. for quantum
        efficiency)'''
    )
    parser.add_argument(
        '--dc-scale', type=float, default=DC_QUANT_EFF,
        help='''DeepCore DOM scaling factor (e.g. for quantum efficiency)'''
    )
    parser.add_argument(
        '--strings', type=str, nargs='+', required=False, default=None,
        help='''Only use these strings (indices start at 1, as per the IceCube
        convention). Specify a human-redable string, e.g. "80-86" to include
        only DeepCore strings, or "26-27,35-37,45-46,80-86" to include the
        IceCube strings that are considered to be part of DeepCore as well as
        "DeepCore-proper" strings. Note that spaces are acceptable.'''
    )
    parser.add_argument(
        '--depths', type=str, nargs='+', required=False, default=None,
        help='''Only use these depths, specified as indices with shallowest at
        0 and deepest at 59. Note that the actual depths of the DOMs depends
        upon whether the string is in DeepCore or not. Specify a human-redable
        string, e.g. "50-59" to include depths {50, 51, ..., 59}. Or one
        could specify "4-59:5" to use every fifth DOM on each string. Note that
        spaces are acceptable.'''
    )
    parser.add_argument(
        '--times', type=str, nargs='+', required=False, default=None,
        help='''Only use these times (specified as indices) from the source
        (t,r,theta)-binned Retro DOM tables. Specify as a human-readable
        sequence, similarly to --strings and --depths.'''
    )
    parser.add_argument(
        '--recompute-binmap', action='store_true',
        help='''Recompute the bin mapping even if the file exists; the existing
        file will be overwritten.'''
    )
    parser.add_argument(
        '--recompute-table', action='store_true',
        help='''Recompute the Retro time- and DOM-independent (TDI) table even
        if the corresponding files exist; these files will be overwritten.'''
    )

    kwargs = vars(parser.parse_args())

    for key in ['strings', 'depths', 'times']:
        val = kwargs[key]
        if val is None:
            kwargs[key] = slice(None)
        else:
            kwargs[key] = hrlist2list(','.join(val))

    return kwargs


if __name__ == '__main__':
    tdi_data = generate_tdi_table(**parse_args())
