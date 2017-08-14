"""
Perform fast(er) exact analytical hypothesis photon expectations.
"""

# pylint: disable=print-statement, wrong-import-position, invalid-name, line-too-long


from __future__ import absolute_import, division

import os
from os.path import abspath, dirname
import time

import numba # pylint: disable=unused-import
import numpy as np

if __name__ == '__main__' and __package__ is None:
    os.sys.path.append(dirname(dirname(abspath(__file__))))
from retro import (BinningCoords, FTYPE, SPEED_OF_LIGHT_M_PER_NS,
                   HYPO_PARAMS_T, hypo_to_track_params, PI_BY_TWO,
                   TimeSpaceCoord, TRACK_M_PER_GEV, TWO_PI)
from sparse import Sparse


__all__ = ['inner_loop', 'Track', 'Hypo']


#@numba.jit(nopython=False, nogil=True, fastmath=True, cache=True, parallel=True)
def inner_loop(z, k, phi_inter, theta_inter_neg, theta_inter_pos, r_inter_neg,
               r_inter_pos, photons_per_meter):
    """Fill in the Z matrix the length of the track if there is overlap of the
    track with a bin ... using ugly (but numba-fiable?) nested loops"""
    for i in range(phi_inter.shape[0]):
        phi = phi_inter[i]
        if phi[0] < 0 and phi[1] < 0:
            continue
        for m in range(theta_inter_neg.shape[0]):
            theta = theta_inter_neg[m]
            if theta[0] < 0 and theta[1] < 0:
                continue
            for j in range(r_inter_neg.shape[0]):
                r = r_inter_neg[j]
                if r[0] < 0 and r[1] < 0:
                    continue
                # interval of r theta and phi
                A = max(r[0], theta[0], phi[0])
                B = min(r[1], theta[1], phi[1])
                if A <= B:
                    length = B - A
                    z[k, j, m, i] += length * photons_per_meter
            for j in range(r_inter_pos.shape[0]):
                r = r_inter_pos[j]
                if r[0] < 0 and r[1] < 0:
                    continue
                # interval of r theta and phi
                A = max(r[0], theta[0], phi[0])
                B = min(r[1], theta[1], phi[1])
                if A <= B:
                    length = B - A
                    z[k, j, m, i] += length * photons_per_meter
        for m in range(theta_inter_pos.shape[0]):
            theta = theta_inter_pos[m]
            if theta[0] < 0 and theta[1] < 0:
                continue
            for j in range(r_inter_neg.shape[0]):
                r = r_inter_neg[j]
                if r[0] < 0 and r[1] < 0:
                    continue
                # interval of r theta and phi
                A = max(r[0], theta[0], phi[0])
                B = min(r[1], theta[1], phi[1])
                if A <= B:
                    length = B - A
                    z[k, j, m, i] += length * photons_per_meter
            for j in range(r_inter_pos.shape[0]):
                r = r_inter_pos[j]
                if r[0] < 0 and r[1] < 0:
                    continue
                # interval of r theta and phi
                A = max(r[0], theta[0], phi[0])
                B = min(r[1], theta[1], phi[1])
                if A <= B:
                    length = B - A
                    z[k, j, m, i] += length * photons_per_meter
    return z


class Track(object):
    """Class for calculating track positons for a given track hypo.

    Initialize with track parameters.

    Parameters
    ----------
    params : TrackParams
    origin : TimeSpaceCoord

    """
    def __init__(self, params, origin=(0,)*4):
        self.params = params

        # Track length is derived from its energy
        self.length = params.energy * TRACK_M_PER_GEV

        # Pre-calculated quantities
        self.dt = self.length / SPEED_OF_LIGHT_M_PER_NS
        self.sinphi = np.sin(self.params.azimuth)
        self.cosphi = np.cos(self.params.azimuth)
        self.tanphi = np.tan(self.params.azimuth)
        self.sintheta = np.sin(self.params.zenith)
        self.costheta = np.cos(self.params.zenith)
        self.origin = None
        self.set_origin(origin)

    def set_origin(self, coord):
        """Displace track

        Parameters
        ----------
        coord : TimeSpaceCoord namedtuple

        """
        if coord == self.origin:
            return

        if not isinstance(coord, TimeSpaceCoord):
            coord = TimeSpaceCoord(*coord)

        self.origin = coord

        # Define relative coordinates

        self.t0 = self.params.t - self.origin.t
        self.x0 = self.params.x - self.origin.x
        self.y0 = self.params.y - self.origin.y
        self.z0 = self.params.z - self.origin.z

    def point(self, t):
        """return point on track for a given time"""
        # make sure time is valid
        assert (self.t0 <= t) and (t <= self.t0 + self.dt)
        dt = (t - self.t0)
        dr = SPEED_OF_LIGHT_M_PER_NS * dt
        x = self.x0 + dr * self.sintheta * self.cosphi
        y = self.y0 + dr * self.sintheta * self.sinphi
        z = self.z0 + dr * self.costheta
        return (x, y, z)

    def extent(self, t_low, t_high):
        """return maximum time extent of track in a given time interval"""
        if (self.t0 + self.dt >= t_low) and (t_high >= self.t0):
            # then we have overlap
            t_0 = max(t_low, self.t0)
            t_1 = min(t_high, self.t0 + self.dt)
            return (t_0, t_1)
        return None

    @property
    def tb(self):
        """closest time to origin (smallest R)"""
        result = (
            self.t0 - (self.x0 * self.sintheta * self.cosphi
                       + self.y0 * self.sintheta * self.sinphi
                       + self.z0 * self.costheta)
        ) / SPEED_OF_LIGHT_M_PER_NS
        return result

    @property
    def ts(self):
        """time with smallest theta"""
        if ((self.x0 == self.y0 == self.z0 == 0)
                or self.params.zenith == 0
                or self.params.zenith == np.pi):
            return self.t0

        rho = ((- self.costheta * (self.x0*self.x0 + self.y0*self.y0)
                + self.sintheta * self.z0 * (self.x0 * self.cosphi + self.y0 * self.sinphi))
               /
               (+ self.sintheta * self.costheta * (self.x0 * self.cosphi + self.y0 * self.sinphi)
                - self.z0 * self.sintheta*self.sintheta))

        return self.t0 + rho / SPEED_OF_LIGHT_M_PER_NS

    def rho_of_t(self, t):
        if t < self.t0 or t > self.t0 + self.dt:
            return -1
        return (t - self.t0) * SPEED_OF_LIGHT_M_PER_NS

    def rho_of_phi(self, phi):
        """track parameter rho for a give phi"""
        sin = np.sin(phi)
        cos = np.cos(phi)
        result = (
            (sin * self.x0 - cos * self.y0)
            / (cos * self.sinphi * self.sintheta - sin * self.cosphi * self.sintheta)
        )
        return result

    def get_M(self, T):
        """helper function"""
        S = (- self.x0*self.x0 * self.sintheta*self.sintheta * self.sinphi*self.sinphi
             - self.y0*self.y0 * self.cosphi*self.cosphi * self.sintheta*self.sintheta
             + 2*self.x0 * self.y0 * self.sinphi * self.sintheta*self.sintheta * self.cosphi
             + np.tan(T)**2 * (+ (self.x0*self.x0 + self.y0*self.y0) * self.costheta*self.costheta
                               + self.z0*self.z0 * self.sintheta*self.sintheta
                               - 2*self.z0 * self.sintheta * self.costheta * (
                                   self.x0 * self.cosphi + self.y0 * self.sinphi
                               )))
        if S < 0:
            return 0.
        return np.sqrt(S)

    def rho_of_theta_neg(self, T):
        """track parameter rho for a given theta, solution 1"""
        M = self.get_M(T)
        d = -self.sintheta*self.sintheta + self.costheta*self.costheta * np.tan(T)**2
        if d == 0:
            return np.inf

        rho = (+ self.x0 * self.sintheta * self.cosphi
               + self.y0 * self.sinphi * self.sintheta
               - self.z0 * self.costheta * np.tan(T)**2
               + M) / d

        return rho

    def rho_of_theta_pos(self, T):
        """track parameter rho for a given theta, solution 2"""
        M = self.get_M(T)
        d = -self.sintheta*self.sintheta + self.costheta*self.costheta * np.tan(T)**2
        if d == 0:
            return np.inf

        rho = (+ self.x0 * self.sintheta * self.cosphi
               + self.y0 * self.sinphi * self.sintheta
               - self.z0 * self.costheta * np.tan(T)**2
               - M) / d
        return rho

    def get_A(self, R):
        """helper function"""
        S = (
            R*R
            + self.x0*self.x0 * (self.cosphi*self.cosphi * self.sintheta*self.sintheta - 1)
            + self.y0*self.y0 * (self.sinphi*self.sinphi * self.sintheta*self.sintheta - 1)
            + self.z0*self.z0 * (self.costheta*self.costheta - 1)
            + 2*self.x0 * self.y0 * self.sinphi * self.sintheta*self.sintheta * self.cosphi
            + 2*self.x0 * self.z0 * self.cosphi * self.sintheta * self.costheta
            + 2*self.y0 * self.z0 * self.sinphi * self.sintheta * self.costheta
        )
        if S < 0:
            return 0.
        return np.sqrt(S)

    def rho_of_r_pos(self, R):
        """track parameter rho for a given r, solution 1"""
        A = self.get_A(R)
        rho = (A
               - self.x0 * self.sintheta * self.cosphi
               - self.y0 * self.sinphi * self.sintheta
               - self.z0 * self.costheta)
        return rho

    def rho_of_r_neg(self, R):
        """track parameter rho for a given r, solution 2"""
        A = self.get_A(R)
        rho = (-A
               - self.x0 * self.sintheta * self.cosphi
               - self.y0 * self.sinphi * self.sintheta
               - self.z0 * self.costheta)
        return rho


class Hypo(object):
    """
    Hypothesis for a given set of parameters from which one can retrieve maps
    (z-matrix) that contain the expected photon sources in each cell for a
    spherical coordinate system that has its origin at the DOM-hit point. The
    cells are divided up according to the binning that must be set.

    The class is instantiated with the hypo vertex (_v), direction, and track
    and cascade energies.

    Parameters
    ----------
    params : HYPO_PARAMS_T

    cascade_e_scale, track_e_scale : float

    """
    def __init__(self, params, cascade_e_scale=1, track_e_scale=1):
        if not isinstance(params, HYPO_PARAMS_T):
            params = HYPO_PARAMS_T(*params)
        self.params = params

        # Convert track energy to length by (ToDo check this)
        track_params = hypo_to_track_params(params)
        self.track = Track(track_params)

        # Precalculate (from nphotons.py) to avoid icetray
        self.photons_per_meter = 2451.4544553 * track_e_scale
        photons_per_gev_cascade = 12805.3383311 * cascade_e_scale
        self.cascade_photons = params.cascade_energy * photons_per_gev_cascade
        self.track_photons = self.track.length * self.photons_per_meter
        self.tot_photons = self.cascade_photons + self.track_photons

        self.bin_edges = None
        self.bin_centers = None
        self.shape = tuple([])

    def set_binning(self, bin_edges):
        """Set the binning of the spherical coordinates with bin_edges.

        Parameters
        ----------
        bin_edges : BinningCoords

        """
        self.bin_edges = bin_edges
        self.bin_centers = BinningCoords(
            t=0.5 * (bin_edges.t[:-1] + bin_edges.t[1:]),
            r=0.5 * (bin_edges.r[:-1] + bin_edges.r[1:]),
            theta=0.5 * (bin_edges.theta[:-1] + bin_edges.theta[1:]),
            phi=0.5 * (bin_edges.phi[:-1] + bin_edges.phi[1:])
        )
        self.shape = BinningCoords(*(len(dim) for dim in self.bin_centers))

    # -- Coordinate transforms -- #

    @staticmethod
    def cr(x, y, z):
        """r in spehrical coord., given x, y, z"""
        return np.sqrt(x*x + y*y + z*z)

    @staticmethod
    def cphi(x, y, z): # pylint: disable=unused-argument
        """phi in spehrical coord., given x, y, z"""
        v = np.arctan2(y, x)
        if v < 0:
            v += TWO_PI
        return v

    @staticmethod
    def ctheta(x, y, z):
        """theta in spehrical coord., given x, y, z"""
        if x == y == z == 0:
            return 0
        return np.arccos(z / np.sqrt(x*x + y*y + z*z))

    # -- Bin correlators -- #

    def correlate_theta(self, bin_edges, t, rho_extent):
        """get the track parameter rho intervals, which lie inside bins"""
        start_t = time.time()
        intervals = []
        for i in range(len(bin_edges) - 1):
            # get interval overlaps
            # from these two intervals:
            if t is None:
                intervals.append([-1., -1.])
                continue
            b = (bin_edges[i], bin_edges[i+1])
            if abs(t[0] - t[1]) < 1e-7 and (b[0] <= t[0]) and (t[0] <= b[1]):
                # along coordinate axis
                intervals.append(rho_extent)
            elif (b[0] <= t[1]) and (t[0] <= b[1]) and (t[0] < t[1]):
                val_high = min(b[1], t[1])
                val_low = max(b[0], t[0])
                if b[0] < PI_BY_TWO:
                    theta_low = self.track.rho_of_theta_pos(val_low)
                    theta_high = self.track.rho_of_theta_pos(val_high)
                else:
                    theta_low = self.track.rho_of_theta_neg(val_low)
                    theta_high = self.track.rho_of_theta_neg(val_high)
                intervals.append(sorted([theta_low, theta_high]))
            elif (b[0] <= t[0]) and (t[1] <= b[1]) and (t[1] < t[0]):
                val_high = min(b[1], t[0])
                val_low = max(b[0], t[1])
                if b[0] < PI_BY_TWO:
                    theta_low = self.track.rho_of_theta_neg(val_low)
                    theta_high = self.track.rho_of_theta_neg(val_high)
                else:
                    theta_low = self.track.rho_of_theta_pos(val_low)
                    theta_high = self.track.rho_of_theta_pos(val_high)
                intervals.append(sorted([theta_low, theta_high]))
            else:
                intervals.append([-1, -1])
        end_t = time.time()
        #print 'corr theta took %.2f'%((end_t-start_t)*1000)
        return np.array(intervals, dtype=FTYPE)

    def correlate_phi(self, bin_edges, t, rho_extent):
        """get the track parameter rho intervals, which lie inside bins"""
        start_t = time.time()
        # phi intervals
        intervals = []
        for i in range(len(bin_edges) - 1):
            # get interval overlaps
            # from these two intervals:
            b = (bin_edges[i], bin_edges[i+1])
            if abs(t[0] - t[1]) < 1e-14 and b[0] <= t[0] and t[0] < b[1]:
                # along coordinate axis
                intervals.append(rho_extent)
            elif t[0] <= t[1] and b[0] <= t[1] and t[0] < b[1]:
                phi_high = min(b[1], t[1])
                phi_low = max(b[0], t[0])
                r_low = self.track.rho_of_phi(phi_low)
                r_high = self.track.rho_of_phi(phi_high)
                intervals.append(sorted([r_low, r_high]))
            # crossing the 0/2pi point
            elif t[1] < t[0]:
                if b[1] >= 0 and t[1] >= b[0]:
                    phi_high = min(b[1], t[1])
                    phi_low = max(b[0], 0)
                elif TWO_PI > b[0] and b[1] >= t[0]:
                    phi_high = min(b[1], TWO_PI)
                    phi_low = max(b[0], t[0])
                elif b[0] <= t[1] and t[0] <= t[1]:
                    phi_high = min(b[1], t[1])
                    phi_low = max(b[0], t[0])
                else:
                    intervals.append([-1, -1])
                    continue
                r_low = self.track.rho_of_phi(phi_low)
                r_high = self.track.rho_of_phi(phi_high)
                intervals.append(sorted([r_low, r_high]))
            else:
                intervals.append([-1, -1])
        end_t = time.time()
        #print 'corr phi took %.2f'%((end_t-start_t)*1000)
        return np.array(intervals, dtype=FTYPE)

    def correlate_r(self, bin_edges, t, pos):
        """get the track parameter rho intervals, which lie inside bins"""
        start_t = time.time()
        intervals = []
        for i in xrange(len(bin_edges) - 1):
            # get interval overlaps
            # from these two intervals:
            b = (bin_edges[i], bin_edges[i+1])
            if b[0] <= t[1] and t[0] <= b[1]:
                val_high = min(b[1], t[1])
                val_low = max(b[0], t[0])
                if (val_low >= val_high and not pos) or (val_low <= val_high and pos):
                    r_low = self.track.rho_of_r_neg(val_low)
                    r_high = self.track.rho_of_r_neg(val_high)
                else:
                    r_low = self.track.rho_of_r_pos(val_low)
                    r_high = self.track.rho_of_r_pos(val_high)
                intervals.append(sorted([r_low, r_high]))
            else:
                intervals.append([-1, -1])
        end_t = time.time()
        #print 'corr r took %.2f'%((end_t-start_t)*1000)
        return np.array(intervals, dtype=FTYPE)

    #@profile
    def compute_matrices(self, hit_dom_coord):
        """Calculate the matrices for a given DOM hit, i.e.,
        DOM 3D position + hit time.

        Parameters
        ----------
        hit_dom_coord : TimeSpaceCoord or convertible thereto
            Hit position in time and space.

        Returns
        -------
        matrix in t, r, theta, phi
        photon_counts = number of photons
        photon_corr_theta = source direction in theta
        photon_corr_phi = delta phi direction of source
        photon_corr_len = correlation of photons

        """
        # Set the origin of the spherical coordinate system to the DOM hit
        # position
        self.track.set_origin(hit_dom_coord)

        # Closest point (r inflection point)
        tb = self.track.tb
        if tb >= self.track.t0 and tb < self.track.t0 + self.track.dt:
            xb, yb, zb = self.track.point(tb)
            r_inflection_point = self.cr(xb, yb, zb)

        # theta inflection point
        ts = self.track.ts
        if ts >= self.track.t0 and ts < self.track.t0 + self.track.dt:
            xs, ys, zs = self.track.point(ts)
            theta_inflection_point = self.ctheta(xs, ys, zs)

        # the big matrix z
        photon_counts = Sparse(shape=self.shape, default=0, dtype=FTYPE)
        photon_corr_theta = Sparse(shape=self.shape, default=0, dtype=FTYPE)
        photon_corr_phi = Sparse(shape=self.shape, default=0, dtype=FTYPE)
        photon_corr_len = Sparse(shape=self.shape, default=0, dtype=FTYPE)

        start_t = time.time()
        t_inner_loop = 0
        corr_loop = 0
        # iterate over time bins
        total_rho = 0
        for k in range(len(self.bin_edges.t) - 1):
            time_bin = [self.bin_edges.t[k], self.bin_edges.t[k+1]]
            # maximum extent:
            t_extent = self.track.extent(*time_bin)

            # only continue if there is anything in that time bins
            if t_extent is not None:
                # Caluculate the maximal values for each coordinate (r, theta
                # and phi) over which the track spans in the given time bins
                extent = [self.track.point(t_extent[0]),
                          self.track.point(t_extent[1])]
                rho_extent = [self.track.rho_of_t(t_extent[0]),
                              self.track.rho_of_t(t_extent[1])]
                total_rho += rho_extent[1] - rho_extent[0]

                # Radius
                track_r_extent = [self.cr(*extent[0]), self.cr(*extent[1])]
                if tb <= t_extent[0]:
                    track_r_extent_neg = sorted(track_r_extent)
                    track_r_extent_pos = [-1, -1]
                elif tb > t_extent[1]:
                    track_r_extent_pos = sorted(track_r_extent)
                    track_r_extent_neg = [-1, -1]
                else:
                    track_r_extent_pos = sorted([track_r_extent[0], r_inflection_point])
                    track_r_extent_neg = sorted([r_inflection_point, track_r_extent[1]])

                # theta
                track_theta_extent = [self.ctheta(*extent[0]), self.ctheta(*extent[1])]
                if ts <= t_extent[0]:
                    track_theta_extent_neg = track_theta_extent
                    track_theta_extent_pos = [-1, -1]
                elif ts >= t_extent[1]:
                    track_theta_extent_pos = track_theta_extent
                    track_theta_extent_neg = [-1, -1]
                else:
                    track_theta_extent_neg = [track_theta_extent[0], theta_inflection_point]
                    track_theta_extent_pos = [theta_inflection_point, track_theta_extent[1]]

                # Phi
                track_phi_extent = sorted([self.cphi(*extent[0]), self.cphi(*extent[1])])
                if np.abs(track_phi_extent[1] - track_phi_extent[0]) > np.pi:
                    track_phi_extent.append(track_phi_extent.pop(0))

                # Clalculate intervals with bin edges:
                start_t3 = time.time()
                theta_inter_neg = self.correlate_theta(self.bin_edges.theta, track_theta_extent_neg, rho_extent)
                theta_inter_pos = self.correlate_theta(self.bin_edges.theta, track_theta_extent_pos, rho_extent)
                phi_inter = self.correlate_phi(self.bin_edges.phi, track_phi_extent, rho_extent)
                r_inter_neg = self.correlate_r(self.bin_edges.r, track_r_extent_neg, False)
                r_inter_pos = self.correlate_r(self.bin_edges.r, track_r_extent_pos, True)
                end_t3 = time.time()
                corr_loop += end_t3 - start_t3

                start_t2 = time.time()
                photon_counts = inner_loop(photon_counts, k, phi_inter,
                                           theta_inter_neg, theta_inter_pos,
                                           r_inter_neg, r_inter_pos,
                                           self.photons_per_meter)
                end_t2 = time.time()
                t_inner_loop += end_t2 - start_t2

        end_t = time.time()

        # add angles:
        for element in photon_counts:
            idx, _ = element
            theta = self.track.params.zenith
            phi = self.bin_centers.phi[idx[-1]]
            #delta_phi = np.abs(self.track.params.azimuth - phi)%(TWO_PI)
            # calculate same way as in photon propapgation (CLsim)
            delta = np.abs(phi - self.track.params.azimuth)
            delta_phi = delta if delta <= np.pi else TWO_PI - delta
            #delta_phi = np.pi - np.abs((np.abs(phi - self.track.params.azimuth)  - np.pi))
            photon_corr_theta[idx] = theta
            photon_corr_phi[idx] = delta_phi
            # set corr. length for tracks to 1.0, i.e. totally directed
            photon_corr_len[idx] = 1.0

        # add cascade as point:
        # get bin at self.track.t0, ...
        t0 = self.track.t0
        x0 = self.track.x0
        y0 = self.track.y0
        z0 = self.track.z0
        r0 = self.cr(x0, y0, z0)
        theta0 = self.ctheta(x0, y0, z0)
        phi0 = self.cphi(x0, y0, z0)
        # find bins
        t_bin = self.get_bin(t0, self.bin_edges.t)
        r_bin = self.get_bin(r0, self.bin_edges.r)
        theta_bin = self.get_bin(theta0, self.bin_edges.theta)
        phi_bin = self.get_bin(phi0, self.bin_edges.phi)
        if None not in (t_bin, r_bin, theta_bin, phi_bin):
            # Weighted average of corr length from track and cascade, while
            # assuming 0.5 for cascade right now
            photon_corr_len[t_bin, r_bin, theta_bin, phi_bin] = np.average(
                [photon_corr_len[t_bin, r_bin, theta_bin, phi_bin], 0.5],
                weights=[photon_counts[t_bin, r_bin, theta_bin, phi_bin], self.cascade_photons]
            )
            photon_counts[t_bin, r_bin, theta_bin, phi_bin] += self.cascade_photons

        self.photon_counts = photon_counts
        self.photon_corr_theta = photon_corr_theta
        self.photon_corr_phi = photon_corr_phi
        self.photon_corr_len = photon_corr_len

    @staticmethod
    def get_bin(val, bin_edges):
        """ find bin for value val in bin_edges
            return None if val is outside binning
        """
        for k in range(len(bin_edges) - 1):
            if bin_edges[k] <= val and val < bin_edges[k+1]:
                return k
        return None
