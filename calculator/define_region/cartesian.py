"""
Modules for the line, plane or solid shaped region.
"""

import numpy as np
from . import _discretize


def line(self, axis, Min, Max, step=None, num=None, scale='equidistant', seed=64):
    """
    Module for the line discretization.

    Parameters
    ----------
    axis : {'x', 'y', 'z', str}
        Axis of the discretizing line. *str* is the name of
        a custom line which is described in a closed form of equation.
    scale : {'equidistant', 'log', 'random'}
        The scale of discretization:
            - 'equidistant' returns evenly spaced values.
            - 'log' returns evenly on a log scale.
            - 'random' returns randomly sampled value.
    Min : float
        Minimum value for the discretizing line.
    Max : float
        Maximum value for the discretizing line.
    step : float, optional
        Spacing between values. Must be non-negative.
        This argument results in :mod:`numpy.arange`.
    num : int, optional
        Number of samples to generate. Must be non-negative.
        This argument results in :mod:`numpy.linspace`,
        :mod:`numpy.logspace' or :mod: `numpy.random`.
    seed : int or array_like, optional
        Seed the generator. This argument results in :mod:`numpy.random.seed`.

    Notes
    -----
    In 'equidistant' scale, given both arguments *step* and *num*,
    *num* has priority to *step*.
    """
    np.random.seed(seed)

    if axis == 'x':
        x = _discretize(scale, Min, Max, num, step)
        y = np.zeros_like(x)
        z = np.zeros_like(x)
    elif axis == 'y':
        y = _discretize(scale, Min, Max, num, step)
        x = np.zeros_like(y)
        z = np.zeros_like(y)
    elif axis == 'z':
        z = _discretize(scale, Min, Max, num, step)
        x = np.zeros_like(z)
        y = np.zeros_like(z)
    else:
        raise ValueError("Sorry, *scale*={} is now building...".format(axis))

    return x, y, z


def plane(self, axis, Min, Max, step=None, num=None, scale='equidistant', seed=None):
    """
    Module for the line discretization.

    Parameters
    ----------
    axis : 2-tuple of {'x', 'y', 'z', str}
        Axis of the discretizing line. *str* is the name of
        a custom line which is described in a closed form of equation.
    scale : {'equidistant', 'log', 'random'}
        The scale of discretization:
            - 'equidistant' returns evenly spaced values.
            - 'log' returns evenly on a log scale.
            - 'random' returns randomly sampled value.
    Min : float
        Minimum value for the discretizing line.
    Max : float
        Maximum value for the discretizing line.
    step : float, optional
        Spacing between values. Must be non-negative.
        This argument results in :mod:`numpy.arange`.
    num : int, optional
        Number of samples to generate. Must be non-negative.
        This argument results in :mod:`numpy.linspace`,
        :mod:`numpy.logspace' or :mod: `numpy.random`.
    seed : int or array_like, optional
        Seed the generator. This argument results in :mod:`numpy.random.seed`.

    Notes
    -----
    In 'equidistant' scale, given both arguments *step* and *num*,
    *num* has priority to *step*.
    """
    np.random.seed(seed)
    plane_str = ''.join(sorted(axis))

    if plane_str == 'xy':
        x = _discretize(scale, Min, Max, num, step)
        y = _discretize()
    elif plane_str == 'yz':
        y = _discretize(scale, Min, Max, num, step)
    elif plane_str == 'xz':
        z = _discretize(scale, Min, Max, num, step)
