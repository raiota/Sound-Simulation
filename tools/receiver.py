
"""
The :mod:`receiver` module is intended to cover all modules
related to receivers, microphones, or control points
for arbitrary numerical simulation of sound fields,
including sound field visualization and loudspeaker-based control.
"""

import numpy as np
import pandas as pd


class ReceiverParams(pd.DataFrame):
    """
    Container of receivers/microphones and their parameters of placement.

    Notes
    -----
    This class holds coordinate parameters of the multiple receivers/microphones on cartesian,
    like a grid of sound pressure calculation points, or a microphone array, in :pandas.DataFrame: type.
    There are so many ways of measuring sound pressure, so it's possible to set
    arbitrary description in :meth:`__init__`.
    Each receivers/microphones, it is necessary to call :meth:`add_receivers`.

    Parameters
    ----------
    
    """

    DESCRIPTION = None
    INDEXING = 'xy'
    COORD_DATA = {}

    def __init__(self, x, y, z, is_meshgrid=False, description='control points'):

        self.DESCRIPTION = description
        self.COORD_DATA.update({'x': x, 'y': y, 'z': z})

        if is_meshgrid:
            coords = np.meshgrid(x, y, z, indexing=self.INDEXING)
            x, y, z = (tmp.flatten() for tmp in coords)

        super().__init__(pd.DataFrame(np.vstack((x, y, z)).T, columns=['x', 'y', 'z']))


    def get_coord_data(self):
        return self.COORD_DATA



class ReceiverPlacementHelper(object):

    def __init__(self):
        self.params = {}


    def make(self, seed=64, **kw_discretize):

        if self.coordinate_system == 'cartesian':
            self.params.update({'kw_x': kw_discretize['x']})
            self.params.update({'kw_y': kw_discretize['y']})
            self.params.update({'kw_z': kw_discretize['z']})
            X, Y, Z = self._cartesian(self.params['kw_x'], self.params['kw_y'], self.params['kw_z'], seed)

            try:
                self.set_coordinate(X=X, Y=Y, Z=Z)
            except AttributeError:
                return X, Y, Z

        elif shapes_of_region == 'spherical':
            self.params.update({'kw_r': kw_discretize['r']})
            self.params.update({'kw_azi': kw_discretize['azimuth']})
            self.params.update({'kw_ele': kw_discretize['elevation']})
            R, AZI, ELE = self._spherical(self.params['kw_r'], self.params['kw_azi'], self.params['kw_ele'], seed)

            try:
                self.set_coordinate(R=R, AZI=AZI, ELE=ELE)
            except AttributeError:
                return R, AZI, ELE

        else:
            raise ValueError(f":arg:`shapes_of_region`={shapes_of_region} is invalid, \
                choose from 'cartesian' or 'spherical'.")


    @staticmethod
    def _discretize(scale, Min, Max, num=None, step=None, seed=None):
        np.random.seed(seed)

        if scale == 'equidistant':
            discretized_array = np.linspace(Min, Max, num) if num else np.arange(Min, Max+step, step)
        elif scale == 'log':
            if not num:
                raise ValueError(":arg:`num' is needed for log scale discretization.")
            else:
                discretized_array = np.logspace(np.log10(Min), np.log10(Max), num, endpoint=False)
        elif scale == 'random':
            if not num:
                raise ValueError(":arg:`num` is needed for random discretization.")
            else:
                discretized_array = (Max - Min) * np.random.rand(num) + Min
        else:
            raise ValueError(f":arg:`scale`={scale} is invalid.")

        return discretized_array


    @staticmethod
    def _get_from_kwargs(kw):
        Min = kw.get("Min")
        Max = kw.get("Max")
        step = kw.get("step")
        num = kw.get("num")
        scale = kw.get("scale")
        return Min, Max, step, num, scale


    @staticmethod
    def _cartesian(kw_x, kw_y, kw_z, seed=64):
        if isinstance(kw_x, dict):
            Min, Max, step, num, scale = self._get_from_kwargs(kw_x)
            x = self._discretize(scale, Min, Max, num, step, seed)
        else:
            x = np.array([kw_x])

        if isinstance(kw_y, dict):
            Min, Max, step, num, scale = self._get_from_kwargs(kw_y)
            y = self._discretize(scale, Min, Max, num, step, seed*2)
        else:
            y = np.array([kw_y])

        if isinstance(kw_z, dict):
            Min, Max, step, num, scale = self._get_from_kwargs(kw_z)
            z = self._discretize(scale, Min, Max, num, step, seed*3)
        else:
            z = np.array([kw_z])

        return x, y, z


    @staticmethod
    def _spherical(kw_r, kw_azimuth, kw_elevation, seed=64):
        if isinstance(kw_r, dict):
            Min, Max, step, num, scale = self._get_from_kwargs(kw_r)
            r = self._discretize(scale, Min, Max, num, step, seed)
        else:
            r = np.array([kw_r])

        if isinstance(kw_azimuth, dict):
            Min, Max, step, num, scale = self._get_from_kwargs(kw_azimuth)
            azimuth = self._discretize(scale, Min, Max, num, step, seed*2)
        else:
            azimuth = np.array([kw_azimuth])

        if isinstance(kw_elevation, dict):
            Min, Max, step, num, scale = self._get_from_kwargs(kw_elevation)
            elevation = self._discretize(scale, Min, Max, num, step, seed*3)
        else:
            elevation = np.array([kw_elevation])

        return r, azimuth, elevation 