# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, invalid-name

"""
Statistics
"""

from __future__ import absolute_import, division, print_function

__all__ = '''
    poisson_llh
    partial_poisson_llh
    weighted_average
    estimate
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

from os.path import abspath, dirname
import sys

import numpy as np
from scipy.special import gammaln
from scipy import stats

RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
if __name__ == '__main__' and __package__ is None:
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
import retro


def poisson_llh(expected, observed):
    r"""Compute the log Poisson likelihood.

    .. math::
        {\rm observed} \cdot \log {\rm expected} - {\rm expected} \log \Gamma({\rm observed})

    Parameters
    ----------
    expected
        Expected value(s)

    observed
        Observed value(s)

    Returns
    -------
    llh
        Log likelihood(s)

    """
    llh = observed * np.log(expected) - expected - gammaln(observed + 1)
    return llh


def partial_poisson_llh(expected, observed):
    r"""Compute the log Poisson likelihood _excluding_ subtracting off
    expected. This part, which constitutes an expected-but-not-observed
    penalty, is intended to be taken care of outside this function.

    .. math::
        {\rm observed} \cdot \log {\rm expected} - \log \Gamma({\rm observed})

    Parameters
    ----------
    expected
        Expected value(s)

    observed
        Observed value(s)

    Returns
    -------
    llh
        Log likelihood(s)

    """
    llh = observed * np.log(expected) - expected - gammaln(observed)
    return llh


@retro.numba_jit(**retro.DFLT_NUMBA_JIT_KWARGS)
def weighted_average(x, w):
    """Average of elements in `x` weighted by `w`.

    Parameters
    ----------
    x : numpy.ndarray
        Values to average

    w : numpy.ndarray
        Weights, same shape as `x`

    Returns
    -------
    avg : numpy.ndarray
        Weighted average, same shape as `x`

    """
    sum_xw = 0.0
    sum_w = 0.0
    for x_i, w_i in zip(x, w):
        sum_xw += x_i * w_i
        sum_w += w_i
    return sum_xw / sum_w

def weighted_percentile(data, percentile, weights=None):
    '''
    percenttile (0..100)
    weights specifies the frequency (count) of data.
    '''
    if weights is None:
        return np.percentile(data, percents)
    ind=np.argsort(data)
    d=data[ind]
    w=weights[ind]
    p=1.*w.cumsum()/w.sum()*100
    y=np.interp(percentile, p, d)
    return y

def estimate(llhp, percentile_nd=0.95, meta=None):
    '''
    Evaluate estimator for reconstruction quantities given
    the MultiNest points of LLH space exploration
    
    Paranters:
    llhp : structured nd array with columns `llh` + any reco quantities
    percentile_nd : float
        on what percentile of llh values to base the calculation on
    meta : dict
        meta information from the minimization
    
    Returns : dict of estimated points incluing uncertainties
    '''
    
    columns = list(llhp.dtype.names)
    assert 'llh' in columns, 'llh not in %s'%columns
    columns.remove('llh')
    
    nd = len(columns)
    
    # keep best LLHs
    cut = llhp['llh'] > np.nanmax(llhp['llh']) - stats.chi2.ppf(percentile_nd, nd)
    
    if np.sum(cut) == 0:
        raise IndexError('no points')

    # can throw rest away
    llhp = llhp[cut]

    weights = np.ones(len(llhp))
    # calculate prior weights
    if not meta is None:
        priors = meta['priors_used']

        for dim in columns:
            prior = priors[dim]
            if prior[0] == 'uniform':
                continue
            elif prior[0] == 'spefit2':
                weights /= stats.cauchy.pdf(llhp[dim], *prior[1])
            elif prior[0] == 'lognorm':
                weights /= stats.lognorm.pdf(llhp[dim], *prior[1])
            elif prior[0] == 'cosine':
                weights /= np.clip(np.sin(llhp[dim]), 0.01, None)
            elif prior[0] == 'log_uniform' and dim == 'energy':
                weights *= llhp['track_energy'] + llhp['cascade_energy']
            else:
                raise NotImplementedError('prior %s for dimension %s unknown'%(prior[0], dim))

    estimator = {}
    estimator['mean'] = {}
    estimator['weighted_mean'] = {}
    estimator['median'] = {}
    estimator['weighted_median'] = {}
    estimator['low'] = {}
    estimator['high'] = {}

    # cut away upper and lower 13.35% to arrive at 1 sigma
    percentile = (percentile_nd - 0.682689492137086) / 2. * 100.

    for col in columns:
        var = llhp[col]
        if 'azimuth' in col:
            # azimuth is a cyclic function, so need some special treatement to get correct mean
            mean = stats.circmean(var)
            shifted = (var - mean + np.pi)%(2*np.pi)
            weighted_mean = (np.average(shifted, weights=weights) + mean - np.pi)%(2*np.pi)
            weighted_median = (weighted_percentile(shifted, 50, weights) + mean - np.pi)%(2*np.pi)
            median = (np.median(shifted) + mean - np.pi)%(2*np.pi)
            low = (weighted_percentile(shifted, percentile, weights) + mean - np.pi)%(2*np.pi)
            high = (weighted_percentile(shifted, 100-percentile, weights) + mean - np.pi)%(2*np.pi)
        else:
            mean = np.mean(var)
            weighted_mean = np.average(var, weights=weights)
            weighted_median = weighted_percentile(var, 50, weights)
            median = np.median(var)
            low = weighted_percentile(var, percentile, weights)
            high = weighted_percentile(var, 100-percentile, weights)
        estimator['mean'][col] = mean
        estimator['weighted_mean'][col] = weighted_mean
        estimator['median'][col] = median
        estimator['weighted_median'][col] = weighted_median
        estimator['low'][col] = low
        estimator['high'][col] = high
    return estimator
