
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