#   This file is part of Sound-Simulation.

"""
The :mod:`source` module is intended to cover all modules,
related to sound sources for arbitrary numerical simulation of sound fields, except driving signals,
including sound field visualization and loudspeaker-based control.
"""

import warnings
import numpy as np
from enum import Enum, auto



class SourceCategory(Enum):
    POINT = auto()
    LINE = auto()
    CIRCULAR = auto()



class SourceVariables(object):
    POINT =    ('x', 'y', 'z')
    LINE =     ('x', 'y', 'z', 'length', 'elevation', 'azimuth')
    CIRCULAR = ('x', 'y', 'z', 'diameter', 'elevation', 'azimuth')



class _SourcePlacementHelper(object):
    """
    Tools to help you create a source/loudspeaker placement list,
    which is used in :arg:`source_param_list` in :class:`SourcerParams`.

    Warnings
    --------
    This class is recommended for use in :class:`SpeakerParams`.
    """

    def define_temp_sources(self):
        """
        Define the temporary used source.
        """

        source_param_list = np.zeros(len(self.param_keys))

        try:
            self.add_sources(source_param_list)
        except AttributeError:
            return None

        return self


    def define_sources_on_line(self, source_num, start_point, end_point):
        """
        Define "Line Array".
        Return or define source/loudspeaker parameters in :inst:`SourceParams`. 

        Parameters
        ----------
        source_num : int
            The total number of sources/speakers.
        start_point : numpy.ndarray-(3,)
            The start point of line on cartesian coordinate.
        end_point : numpy.ndarray-(3,)
            The end point of line on cartesian coordinate.
        """

        source_param_list = np.zeros((source_num, len(self.param_keys)))

        source_param_list[:, self.param_keys.index('x')] = np.linspace(start_point[0], end_point[0], source_num)
        source_param_list[:, self.param_keys.index('y')] = np.linspace(start_point[1], end_point[1], source_num)
        source_param_list[:, self.param_keys.index('z')] = np.linspace(start_point[2], end_point[2], source_num)

        if self.shape != SourceCategory.POINT:
            warnings.warn(f"In this method, only the coordinates can be determined, so all the other parameters \
                            of the loudspeaker are set to zero.")

        try:
            self.add_sources(source_param_list.flatten())
        except AttributeError:
            return source_param_list

        return self


    def define_sources_on_spherical(self, source_num, centre, r, azimuthes, elevations):
        """
        Define "Spherical Loudspeaker Array".
        Return or define source/speaker parameters in :inst:`SourceParams`.

        Parameters
        ----------
        source_num : int
            The total number of sources/speakers.
        centre : numpy.ndarray-(3,)
            The central position of sphere on cartesian coordinate.
        r : float
            The radius of sphere.
        azimuth : array_like-(source_num,)
            The azimuthes of sources/speakers.
        elevations : array_like-(source_num,)
            The elevations of sources/speakers.
        """

        if len(azimuthes) == source_num and len(elevations) == source_num:
            source_param_list = np.zeros((source_num, len(self.param_keys)))

            source_param_list[:, self.param_keys.index('x')] = centre[0] + r * np.sin(elevations) * np.cos(azimuthes)
            source_param_list[:, self.param_keys.index('y')] = centre[1] + r * np.sin(elevations) * np.sin(azimuthes)
            source_param_list[:, self.param_keys.index('z')] = centre[2] + r * np.cos(elevations)

        else:
            raise ValueError(f"len(azimuthes), len(elevations) must be ({source_num}, {source_num}) \
                                , not ({len(azimuthes)}, {len(elevations)})")

        if self.shape != SourceCategory.POINT:
            warnings.warn(f"In this method, only the coordinates can be determined, so all the other parameters \
                            of the loudspeaker are set to zero.")

        try:
            self.add_sources(source_param_list.flatten())
        except AttributeError:
            return source_param_list

        return self



class SourceParams(dict, _SourcePlacementHelper):
    """
    Container of sources/speakers and their parameters.

    Notes
    -----
    This class holds parameters of the multiple sources/loudspeakers, like loudspeaker array, in :dict: type.
    Each source/loudspeaker is assigned a key of :int:, which is 0 or greater,
    and its value is stored in a :dict: with the corresponding parameters of the source/loudspeaker.
    :meth:`__init__` method needs to be given the shape of the source/loudspeaker, like point, line or circular,
    which is shared within the instance. That is, it's NOT possible to store two different types
    of sources/loudspeakers in a single instance. The definable source/speaker shapes and the parameters
    required for each are as follows,

        - 'point': IDEAL point source
            'x', 'y', 'z': the position of the source/speaker on cartesian coordinate [m]
        - 'line': IDEAL line source
            'x', 'y', 'z': the position of the centre of line on cartesian coordinate [m]
            'length': the total length of the line source [m]
            'elevation': the angle of line vector from z axis [rad]
            'azimuth': the angle of line vector from x axis [rad]
        - 'circular': IDEAL circular source
            'x', 'y', 'z': the position of the centre of circle on cartesian coordinate [m]
            'diameter': the diameter of the circular source [m]
            'elevation': the angle of normal vector of circle surface from z axis [m]
            'azimuth': the angle of normal vector of circle surface from x axis [m]


    Parameters
    ----------
    shape : enum
        Shapes of sources/speakers from :class:`SourceCategory`.
    source_param_list : 1D-numpy.ndarray, optional.
        The rows are lists of source/loudspeaker parameters and the columns correspond to each source/loudspeaker,
        The parameters along row must be in the order in which they appear in :class:`SourceVariables`,
        for 2D-shape examples of 4 circular sources/loudspeakers;

        =================================================================================================
            source_param_list = np.array([[x1, y1, z1, diameter1, elevation1, azimuth1],
                                            [x2, y2, z2, diameter2, elevation2, azimuth2],
                                            [x3, y3, z3, diameter3, elevation3, azimuth3],
                                            [x4, y4, z4, diameter4, elevation4, azimuth4]]).flatten()
        =================================================================================================

        can be used.


    About This
    ----------
    The final form of this instance is the following dictionary;

        {
            0: {                        # <--- Speaker/Source Number(key)   (1)
                "x": ---,               # ¯|
                "y": ---,               #  |
                "z": ---,               #  | <--- means;
                "diameter": ---,        #  |          "parameter": "value"  (2)
                "elevation": ---,       #  |
                "azimuth": ---,         # _|
            },
            1: {                        # <--- Speaker/Source Number(key)   (1)
                "x": ---,               # ¯|
                "y": ---,               #  |
                "z": ---,               #  | <--- means;
                "diameter": ---,        #  |          "parameter": "value"  (2)
                "elevation": ---,       #  |
                "azimuth": ---,         # _|
            },
            2: {                        # <--- Speaker/Source Number(key)   (1)
                "x": ---,               # ¯|
                "y": ---,               #  |
                "z": ---,               #  | <--- means;
                "diameter": ---,        #  |          "parameter": "value"  (2)
                "elevation": ---,       #  |
                "azimuth": ---,         # _|
            }
        }

    Described above means that this dictionary has 3 circular sources/speakers.
    """

    # list2dict = lambda keys, lists: {key: value for key, value in zip(keys, lists)}

    def __init__(self, shape, source_param_list=None, *args, **kwargs):

        self.shape = shape

        if shape == SourceCategory.POINT:
            self.param_keys = SourceVariables.POINT
        elif shape == SourceCategory.LINE:
            self.param_keys = SourceVariables.LINE
        elif shape == SourceCategory.CIRCULAR:
            self.param_keys = SourceVariables.CIRCULAR
        else:
            raise ValueError(f":arg:`shape`={shape} is invalid.")

        if source_param_list is not None:
            self.add_sources(source_param_list, *args, **kwargs)


    def add_sources(self, source_param_list, fixed_parameters=None, fixed_values=None):
        """ Add more sources/speakers on self, the required :arg:`source_param_list` is the same as the one in :meth:`__init__`.
        """
        if fixed_parameters and fixed_values is not None:
            active_param_num = len(self.param_keys) - len(fixed_parameters)
            is_divisible = len(source_param_list) % active_param_num == 0

            if is_divisible:
                source_num = int(len(source_param_list) / active_param_num)
                source_list = source_param_list.reshape(source_num, active_param_num)

                for name, value in zip(fixed_parameters, fixed_values):
                    source_list = np.insert(source_list, self.param_keys.index(name), value, axis=1)

                self.update({key+len(self): self._param_list2dict(value) for key,value in enumerate(source_list)})

            else:
                raise ValueError(f":arg:`source_param_list` cannot be transformed into the given :arg:`shape`.")

        elif fixed_parameters is None and fixed_values is None:
            is_divisible = len(source_param_list) % len(self.param_keys) == 0

            if is_divisible:
                source_num = int(len(source_param_list) / len(self.param_keys))
                source_list = source_param_list.reshape(source_num, len(self.param_keys)).tolist()

                self.update({key+len(self): self._param_list2dict(value) for key,value in enumerate(source_list)})

            else:
                raise ValueError(f":arg:`source_param_list` cannot be transformed into the given :arg:`shape`.")

        else:
            raise ValueError(f":arg:`source_param_list` cannot be transformed into the given :arg:`shape`.")


    def _param_list2dict(self, param_list):
        """
        Internal method for connecting parameter keyword with the value.

        Parameters
        ----------
        param_list : list
            List that contains parameter values corresponding to their parameter key.

        Returns
        -------
        dict
            Dictionary corresponds to (2) at :document:`About This` in :meth:`__init__`.
        """
        return {key: value for key, value in zip(self.param_keys, param_list)}


    def get_arbit_param(self, keys, source_no=None):
        """
        Outputs a list/value of arbitrary parameters.

        Parameters
        ----------
        keys : string of {'x', 'y', 'z', etc...}
            The parameter which you want to get.
        source_no : tuple of int, optional.
            The source number mentioned (1) at :doc:`About This` in :meth:`__init__`.

        Returns
        -------
        list, float
            You get a list which index coresponds the source/speaker number(1) or :arg:`source_no` and
            the value corresponds the :arg:`keys` parameter.
        """
        if source_no is None:
            return [self[no][keys] for no in range(len(self))]
        else:
            return [self[no][keys] for no in source_no]


    def change_arbit_param(self, keys, value, source_no=None):
        """
        Changes a value of arbitrary parameters.

        Parameters
        ----------
        keys : string of {'x', 'y', 'z', etc...}
            The parameter which you want to change.
        value : array-like
            The values of parameter whose length corresponds to :arg:`source_no`.
        source_no : tuple of int, optional.
            The source number mentioned (1) at :doc:`About This` in :meth:`__init__`.
        """
        if source_no is None:
            for i in range(len(self)):
                self[i][keys] = value[i]
        else:
            for i in source_no:
                self[i][keys] = value[i]