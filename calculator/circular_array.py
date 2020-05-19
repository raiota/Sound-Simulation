"""
Calculating sound field from circular source arrays
"""


import numpy as np
from scipy.special import jn

try:
    from __init__ import SpeakerParams
except:
    from .__init__ import SpeakerParams

try:
    from define_region import InterestRegion
except:
    from .define_region import InterestRegion


class CircleFarType(InterestRegion):
    """
    Calculating sound field from circular source arrays, using far-field approximation

    Parameters
    ----------
    frequency : 1D-numpy.ndarray
        List of evaluating frequency [Hz]
    velocity : float, Default 344.0
        Sound velocity [m/s]

    Other Parameters
    ----------------
    **region_param
        This method also takes the keyword arguments for the base :class:`InterestRegion`.
        See more documentation it.
    """

    def __init__(self, frequency=None, velocity=344., **region_param):

        self.velocity = velocity
        self.frequency = frequency
        try:
            self.wavenums = 2 * np.pi * self.frequency / self.velocity
        except:
            pass

        try:
            super().__init__(shapes_of_region=region_param['shapes_of_region'], **region_param['kw_descretize'])
        except KeyError:
            pass


    def set_frequency(self, frequency):
        """
        Setting a frequency to be calculated.

        Parameters
        ----------
        frequency : 1D-numpy.ndarray
            List of evaluating frequency [Hz]
        """
        self.frequency = frequency
        self.wavenums = 2 * np.pi * self.frequency / self.velocity


    def set_field(self, region_object=None, velocity=344., **region_param):
        """
        Setting a field to be calculated.

        Parameters
        ----------
        region_object : type(InterestRegion), optional
            Instance from :class:`InterestRegion`
        
        """
        self.velocity = velocity
        try:
            self.wavenums = 2 * np.pi * self.frequency / self.velocity
        except:
            pass

        if region_object is not None:
            self.shapes = region_object.shapes

            coord_datas = region_object.get_data()
            self.X = coord_datas.get("x")
            self.Y = coord_datas.get("y")
            self.Z = coord_datas.get("z")
            self.R = coord_datas.get("r")
            self.Azimuth = coord_datas.get("azimuth")
            self.Elevation = coord_datas.get("elevation")

        try:
            super().__init__(shapes_of_region=region_param['shapes_of_region'], **region_param['kw_descretize'])
        except KeyError:
            pass


    def set_speakers(self, speaker_param_list, **driving_kwargs):
        """
        Setting a loudspeaker array parameters

        Parameters
        ----------
        speaker_param_list : 1D-numpy.ndarray
            1-D Array that is combined parameters of all loudspeakers into one dimension.
            ex.) np.array([drive1, diameter1, x1, y1, z1, alpha1, beta1,
                            drive2, diameter2, x2, y2, z2, alpha2, beta2,
                            ..., driveM, diameterM, xM, yM, zM, alphaM, betaM])

        Notes
        -----
        For more details on the argument 'speaker_param_list', see base :class:`SpeakerParams`.

        Other Parameters
        ----------------
        **driving_kwargs

        """
        self.speakers = SpeakerParams(shape='circular', speaker_param_list=speaker_param_list)


    def set_driving_function(self, driving_function_list, speaker_no=0):
        """
        Set frequency-dependent driving functions.
        Argument *driving_function_list* is M*N-numpy.ndarray which is consisted of 
        the number M of all speakers and the drive function vector of magnitude N corresponding to each frequency.

        Parameters
        ----------
        speaker_no : int, default 0, optional
            Initial speaker number to set the driving function
        driving_function_list : M*N-numpy.ndarray
        """
        for i in range(driving_function_list.shape[0]):
            self.speakers.set_driving_function(speaker_no+i, frequency=self.frequency,
                                                value=driving_function_list[i])


    def _generate_transfer_matrix(self, mesh=True):

        speaker_diameters = np.array(self.speakers.get_arbit_param("diameter"))
        speaker_xs = np.array(self.speakers.get_arbit_param("x"))
        speaker_ys = np.array(self.speakers.get_arbit_param("y"))
        speaker_zs = np.array(self.speakers.get_arbit_param("z"))
        speaker_as = np.array(self.speakers.get_arbit_param("alpha"))
        speaker_bs = np.array(self.speakers.get_arbit_param("beta"))

        # GREEN FUNC.
        if mesh:
            self.cart2meshgrid(flatten=True)
            tmp1, tmp2 = np.meshgrid(self.XX, speaker_xs, indexing='ij')
            distance_x = tmp1 - tmp2
            tmp1, tmp2 = np.meshgrid(self.YY, speaker_ys, indexing='ij')
            distance_y = tmp1 - tmp2
            tmp1, tmp2 = np.meshgrid(self.ZZ, speaker_zs, indexing='ij')
            distance_z = tmp1 - tmp2
        else:
            tmp1, tmp2 = np.meshgrid(self.X, speaker_xs, indexing='ij')
            distance_x = tmp1 - tmp2
            tmp1, tmp2 = np.meshgrid(self.Y, speaker_ys, indexing='ij')
            distance_y = tmp1 - tmp2
            tmp1, tmp2 = np.meshgrid(self.Z, speaker_zs, indexing='ij')
            distance_z = tmp1 - tmp2
        distance_matrix = np.sqrt(distance_x**2 + distance_y**2 + distance_z**2)
        greens_tensor = np.exp(1j * np.einsum('i,jk->ijk', self.wavenums, distance_matrix)) / distance_matrix

        # DIRECTIVITY FUNC.
        normal_vecs = np.array([np.sin(speaker_as)*np.cos(speaker_bs), np.sin(speaker_as)*np.sin(speaker_bs), np.cos(speaker_as)])
        normal_vecs_norm = np.linalg.norm(normal_vecs, axis=0)
        if mesh:
            inner_matrix = np.einsum('i,j->ij', self.XX, normal_vecs[0]) + np.einsum('i,j->ij', self.YY, normal_vecs[1]) \
                            + np.einsum('i,j->ij', self.ZZ, normal_vecs[2])
        else:
            inner_matrix = np.einsum('i,j->ij', self.X, normal_vecs[0]) + np.einsum('i,j->ij', self.Y, normal_vecs[1]) \
                            + np.einsum('i,j->ij', self.Z, normal_vecs[2])
        theta_matrix = np.arccos(inner_matrix / (distance_matrix * normal_vecs_norm))
        inside_vessel = np.einsum('i,jk->ijk', self.wavenums, np.sin(theta_matrix)) * speaker_diameters * 0.5
        directivity_tensor = jn(1, inside_vessel) / inside_vessel
        np.nan_to_num(directivity_tensor, nan=0.5, copy=False)

        self.transfer_matrix = directivity_tensor * greens_tensor


    def getPressure(self, is_driving_func=False, mesh=True):
        """
        Compute the sound pressure
        """
        self._generate_transfer_matrix(mesh=mesh)
        if is_driving_func:
            driving_matrix = np.array([self.speakers.driving_function2array(i) for i in range(len(self.speakers))])
        else:
            driving_matrix = np.ones((len(self.speakers), self.frequency.size))

        self.pressure = np.einsum('ijk,ki->ji', self.transfer_matrix, driving_matrix)
        return self.pressure, self.get_data()


    def getSPL(self, p0=10**(-5)*2.):
        """
        Compute the Sound Pressure Level
        """
        try:
            spl = 10*np.log10(np.abs(self.pressure)**2) - 20*np.log10(p0)
        except AttributeError:
            raise AttributeError("Before using this method, it is necessary to calculate the sound pressure using :method:`getPressure` first.")

        return spl, self.get_data()


    def getAmplitude(self):
        """
        Compute the amplitude of sound pressure
        """
        try:
            amplitude = np.abs(self.pressure)
        except AttributeError:
            raise AttributeError("Before using this method, it is necessary to calculate the sound pressure using :method:`getPressure` first.")

        return amplitude, self.get_data()


    def getPhase(self):
        """
        Compute the amplitude of sound pressure
        """
        try:
            phase = np.angle(self.pressure)
        except AttributeError:
            raise AttributeError("Before using this method, it is necessary to calculate the sound pressure using :method:`getPressure` first.")

        return phase, self.get_data()


    def getDirectivity(self, plane, central_position_vector=np.zeros(3), datatype='SPL'):
        """
        Method for conversing polar coordinate to cartesian coordinate.

        Parameters
        ----------
        plane : {'xy', 'yz', 'zx'}
            Plane on which the polar coordinate stands.
        central_position_vector : numpy.ndarray-(3,). Default numpy.zeros(3), optional
            Origin of the ``r`` vector.
        datatype : {'Pressure', 'SPL', 'Amplitude', 'Phase'}. Default 'spl', optional.
            Set the data type to be returned

        Returns
        -------
        x, y, z
            the resulted cartesian coordinate.
        """
        try:
            self.polar2cartesian(plane, central_position_vector)
            if (self.R.size > 1):
                raise ValueError("Computing directivity by using this method, 'R' must have the shape of 1.")
        except AttributeError:
            raise ValueError("Before using this method, it is necessary to set polar coordinate using :method:`set_field` first.")

        self.getPressure(mesh=False)

        if datatype == 'Pressure':
            return self.pressure, self.get_data()
        else:
            return eval("self.get" + datatype)()


    def calc_driving_function_using_least_square_method(self, p_des, regularization=0.01):
        conj_tp_transfer_matrix = np.conj(np.einsum('ijk->ikj', self.transfer_matrix))
        inside_inverse = np.einsum('ikj,ijl->ikl', conj_tp_transfer_matrix, self.transfer_matrix) - regularization * np.eye(len(self.speakers))
        return np.einsum('ikk,ikj,ji->ki', np.linalg.inv(inside_inverse), conj_tp_transfer_matrix, p_des)






if __name__ == '__main__':

    REGION_PARAM = {
        "x": {
            "Min": 0.,
            "Max": 5.0,
            "step": 0.02,
            "scale": 'equidistant'
        },
        "y": 0.,
        "z": {
            "Min": 1.0,
            "Max": 10.0,
            "step": 0.02,
            "scale": 'equidistant'
        }
    }

    FREQUENCY = np.array([250, 500, 1000, 2000, 4000, 8000])

    SPEAKER_LIST = np.array([1., 0.08,  0.45, 0., 0., 0., 0.,
                            1., 0.08,  0.35, 0., 0., 0., 0.,
                            1., 0.08,  0.25, 0., 0., 0., 0.,
                            1., 0.08,  0.15, 0., 0., 0., 0.,
                            1., 0.08,    0., 0., 0., 0., 0.,
                            1., 0.08, -0.15, 0., 0., 0., 0.,
                            1., 0.08, -0.25, 0., 0., 0., 0.,
                            1., 0.08, -0.35, 0., 0., 0., 0.,
                            1., 0.08, -0.45, 0., 0., 0., 0.])

    calculator = CircleFarType(frequency=FREQUENCY, shapes_of_region='cartesian', kw_descretize=REGION_PARAM)

    calculator.set_speakers(SPEAKER_LIST)

    pressure, region = calculator.getPressure()
    level, _ = calculator.getSPL()
    phase, _ = calculator.getPhase()

    # VISUALIZATION
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    fig.suptitle("Sound Pressure Level Plot", size=16)

    for i, ax in enumerate(axes.flat):
        im = ax.pcolorfast(region["z"], region["x"], level[:,i].reshape(region["x"].size, region["z"].size), vmin=30, vmax=120, cmap='magma')
        ax.set_title(f'{FREQUENCY[i]} Hz')
    fig.subplots_adjust(hspace=0.3)
    fig.colorbar(im, ax=axes.flat)
    plt.show()


    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    fig.suptitle("Phase Plot", size=16)

    for i, ax in enumerate(axes.flat):
        im = ax.pcolorfast(region["z"], region["x"], phase[:,i].reshape(region["x"].size, region["z"].size), cmap='viridis')
        ax.set_title(f'{FREQUENCY[i]} Hz')
    fig.subplots_adjust(hspace=0.3)
    fig.colorbar(im, ax=axes.flat)
    plt.show()


    # -*-*-*- DIRECTIVITY -*-*-*-
    POLAR_PARAM = {
        "r": 10.0,
        "azimuth": {
            "Min": -np.pi*.5,
            "Max": np.pi*.5,
            "num": 200,
            "scale": "equidistant",
        }
    }

    calculator.set_field(shapes_of_region='polar', kw_descretize=POLAR_PARAM)
    value, region = calculator.getDirectivity(plane='zx')

    # Visualization
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    fig.suptitle("Directivity Plot", size=16)

    for i, ax in enumerate(axes.flat):
        ax.plot(region["azimuth"], value[:, i])
        ax.set_title(f'{FREQUENCY[i]} Hz')
        ax.grid(which='both', color='lightgrey')
    fig.subplots_adjust(hspace=0.3)
    plt.show()
