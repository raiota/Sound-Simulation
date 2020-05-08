"""
A module for descretizing sound field.
"""

import numpy as np


class InterestRegion(object):
    """
    Define the interest region.

    Parameters
    ----------
    shapes_of_region : {'cartesian', 'polar', 'spherical'}, optional.
        The way to describe the interest region. The default choice
        results in a 'cartesian' shape.
    seed : int or array_like, optional
        The seed value of random generator, which is used in `numpy.random`.
    
    Other parameters
    ----------------
    **kw_descretize

    """
    def __init__(self, shapes_of_region='cartesian', seed=64, **kw_descretize):
        self.shapes = shapes_of_region

        if shapes_of_region == 'cartesian':
            self.kw_x = kw_descretize["x"]
            self.kw_y = kw_descretize["y"]
            self.kw_z = kw_descretize["z"]
            self.X, self.Y, self.Z = _cartesian(self.kw_x, self.kw_y, self.kw_z, seed)

        elif shapes_of_region == 'polar':
            self.kw_r = kw_descretize["r"]
            self.kw_azimuth = kw_descretize["azimuth"]
            self.R, self.Azimuth = _polar(self.kw_r, self.kw_azimuth, seed)

        elif shapes_of_region == 'spherical':
            self.kw_r = kw_descretize["r"]
            self.kw_azimuth = kw_descretize["azimuth"]
            self.kw_elevation = kw_descretize["elevation"]
            self.R, self.Azimuth, self.Elevation = _spherical(self.kw_r, self.kw_azimuth, self.kw_elevation, seed)


    def cart2meshgrid(self, flatten=False, indexing='ij'):
        """
        Method for creating meshgrid from a cartesian coordinate array.

        Parameters
        ----------
        flatten : bool. Default False, optional.
            Whether or not to return the array collapsed into one dimension.
        indexing : {'xy', 'ij'}. Default 'xy', optional.
            Argument will be used in :mod:`numpy.meshgrid`

        Returns
        -------
        None

        Notes
        -----
        Attribute `XX`, `YY` and `ZZ` will be added.
        """
        self.XX, self.YY, self.ZZ = np.meshgrid(self.X, self.Y, self.Z, indexing='ij')
        if flatten:
            self.XX = self.XX.ravel()
            self.YY = self.YY.ravel()
            self.ZZ = self.ZZ.ravel()


    def polar2cartesian(self, plane, position_vector=np.zeros(3)):
        """
        Method for conversing polar coordinate to cartesian coordinate.

        Parameters
        ----------
        plane : {'xy', 'yz', 'zx'}
            Plane on which the polar coordinate stands.
        position_vector : numpy.ndarray-(3,). Default numpy.zeros(3), optional
            Origin of the ``r`` vector.

        Returns
        -------
        x, y, z
            the resulted cartesian coordinate.
        """
        if hasattr(self, 'R') or hasattr(self, 'Azimuth'):
            if plane == 'xy':
                try:
                    self.X = position_vector[0] + self.R * np.cos(self.Azimuth)
                    self.Y = position_vector[1] + self.R * np.sin(self.Azimuth)
                except ValueError:
                    self.X = position_vector[0] + np.array([self.R[i] * np.cos(self.Azimuth) for i in self.R.size])
                    self.Y = position_vector[1] + np.array([self.R[i] * np.sin(self.Azimuth) for i in self.R.size])
                self.Z = np.array([position_vector[2]])
            elif plane == 'yz':
                self.X = np.array([position_vector[0]])
                try:
                    self.Y = position_vector[1] + self.R * np.cos(self.Azimuth)
                    self.Z = position_vector[2] + self.R * np.sin(self.Azimuth)
                except ValueError:
                    self.Y = position_vector[1] + np.array([self.R[i] * np.cos(self.Azimuth) for i in self.R.size])
                    self.Z = position_vector[2] + np.array([self.R[i] * np.sin(self.Azimuth) for i in self.R.size]) 
            elif plane == 'zx':
                self.Y = np.array([position_vector[1]])
                try:
                    self.Z = position_vector[2] + self.R * np.cos(self.Azimuth)
                    self.X = position_vector[0] + self.R * np.sin(self.Azimuth)
                except ValueError:
                    self.Z = position_vector[2] + np.array([self.R[i] * np.cos(self.Azimuth) for i in self.R.size])
                    self.X = position_vector[0] + np.array([self.R[i] * np.sin(self.Azimuth) for i in self.R.size])
            else:
                raise ValueError(f"argument *plane*={plane} is invalid.")
        else:
            raise AttributeError("Calling :method:`polar2cartesian`, there's needed to be initialized by `polar` shape.")


    def spherical2cartesian(self, plane, position_vector=np.zeros(3)):
        # Now building...
        print("Sorry, this method is now building...")


    def get_x_param(self):
        """ Returns the discretization parameter of x coordinate in :dict: form..
        """
        return self.kw_x

    def get_y_param(self):
        """ Returns the discretization parameter of y coordinate in :dict: form..
        """
        return self.kw_y

    def get_z_param(self):
        """ Returns the discretization parameter of z coordinate in :dict: form..
        """
        return self.kw_z

    def get_r_param(self):
        """ Returns the discretization parameter of r coordinate in :dict: form.
        """
        return self.kw_r

    def get_azimuth_param(self):
        """ Returns the discretization parameter of azimuth coordinate in :dict: form.
        """
        return self.kw_azimuth

    def get_elevation_param(self):
        """ Returns the discretization parameter of elevation coordinate in :dict: form.
        """
        return self.elevation

    def get_data(self):
        """ Returns all discretized data in :dict: form.
        """
        if self.shapes == 'cartesian':
            return {"x": self.X, "y": self.Y, "z": self.Z}
        elif self.shapes == 'polar':
            return {"r": self.R, "azimuth": self.Azimuth}
        elif self.shapes == 'spherical':
            return {"r": self.R, "azimuth": self.Azimuth, "elevation": self.Elevation}



def _discretize(scale, Min, Max, num=None, step=None, seed=None):
    np.random.seed(seed)

    if scale == 'equidistant':
        discretized_array = np.linspace(Min, Max, num) if num else np.arange(Min, Max+step, step)
    elif scale == 'log':
        if not num:
            raise ValueError("argument *num* is needed for log scale discretization.")
        else:
            discretized_array = np.logspace(np.log10(Min), np.log10(Max), num, endpoint=False)
    elif scale == 'random':
        if not num:
            raise ValueError("argument *num* is needed for random discretization.")
        else:
            discretized_array = (Max - Min) * np.random.rand() + Min
    else:
        raise ValueError(f"argument *scale*={scale} is invalid.")

    return discretized_array



def _get_from_kwargs(kw):
    Min = kw.get("Min")
    Max = kw.get("Max")
    step = kw.get("step")
    num = kw.get("num")
    scale = kw.get("scale")
    return Min, Max, step, num, scale



def _cartesian(kw_x, kw_y, kw_z, seed=64):
    if isinstance(kw_x, dict):
        Min, Max, step, num, scale = _get_from_kwargs(kw_x)
        x = _discretize(scale, Min, Max, num, step, seed)
    else:
        x = np.array([kw_x])
    
    if isinstance(kw_y, dict):
        Min, Max, step, num, scale = _get_from_kwargs(kw_y)
        y = _discretize(scale, Min, Max, num, step, seed*2)
    else:
        y = np.array([kw_y])
    
    if isinstance(kw_z, dict):
        Min, Max, step, num, scale = _get_from_kwargs(kw_z)
        z = _discretize(scale, Min, Max, num, step, seed*3)
    else:
        z = np.array([kw_z])

    return x, y, z



def _polar(kw_r, kw_azimuth, seed=64):
    if isinstance(kw_r, dict):
        Min, Max, step, num, scale = _get_from_kwargs(kw_r)
        r = _discretize(scale, Min, Max, num, step, seed)
    else:
        r = np.array([kw_r])
    
    if isinstance(kw_azimuth, dict):
        Min, Max, step, num, scale = _get_from_kwargs(kw_azimuth)
        azimuth = _discretize(scale, Min, Max, num, step, seed*2)
    else:
        azimuth = np.array([kw_azimuth])

    return r, azimuth



def _spherical(kw_r, kw_azimuth, kw_elevation, seed=64):
    if isinstance(kw_r, dict):
        Min, Max, step, num, scale = _get_from_kwargs(kw_r)
        r = _discretize(scale, Min, Max, num, step, seed)
    else:
        r = np.array([kw_r])
    
    if isinstance(kw_azimuth, dict):
        Min, Max, step, num, scale = _get_from_kwargs(kw_azimuth)
        azimuth = _discretize(scale, Min, Max, num, step, seed*2)
    else:
        azimuth = np.array([kw_azimuth])

    if isinstance(kw_elevation, dict):
        Min, Max, step, num, scale = _get_from_kwargs(kw_elevation)
        elevation = _discretize(scale, Min, Max, num, step, seed*3)
    else:
        elevation = np.array([kw_elevation])

    return r, azimuth, elevation 
