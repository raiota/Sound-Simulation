"""
Calculating sound field from circular source arrays
"""


import numpy as np
from scipy.special import jn

from __init__ import SpeakerParams
from define_region import InterestRegion


class CircleFarType(InterestRegion):
    """
    Calculating sound field from circular source arrays, using far-field approximation

    Parameters
    ----------
    frequency : array_like
        List of evaluating frequency [Hz]
    velocity : float
        Sound velocity, default 344.0 [m/s]

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
            self.cart2meshgrid(flatten=True)
        except KeyError:
            pass


    def set_frequency(self, frequency):
        """
        Setting a frequency to be calculated.

        Parameters
        ----------
        frequency : array_like
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
            coord_datas = region_object.get_data()
            self.X = coord_datas.get("x")
            self.Y = coord_datas.get("y")
            self.Z = coord_datas.get("z")
            self.R = coord_datas.get("r")
            self.Azimuth = coord_datas.get("azimuth")
            self.Elevation = coord_datas.get("elevation")
            self.cart2meshgrid(flatten=True)

        try:
            super().__init__(shapes_of_region=region_param['shapes_of_region'],
                                kw_descretize=region_param['region_param'])
            self.cart2meshgrid(flatten=True)
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
        For more details on the argument 'speaker_list', see base :class:`SpeakerParams`.

        Other Parameters
        ----------------
        **driving_kwargs

        """
        self.speakers = SpeakerParams(shape='circular', speaker_param_list=speaker_param_list)


    def _generate_transfer_matrix(self):

        speaker_diameters = np.array(self.speakers.get_arbit_param("diameter"))
        speaker_xs = np.array(self.speakers.get_arbit_param("x"))
        speaker_ys = np.array(self.speakers.get_arbit_param("y"))
        speaker_zs = np.array(self.speakers.get_arbit_param("z"))
        speaker_as = np.array(self.speakers.get_arbit_param("alpha"))
        speaker_bs = np.array(self.speakers.get_arbit_param("beta"))

        # GREEN FUNC.
        tmp1, tmp2 = np.meshgrid(self.XX, speaker_xs, indexing='ij')
        distance_x = tmp1 - tmp2
        tmp1, tmp2 = np.meshgrid(self.YY, speaker_ys, indexing='ij')
        distance_y = tmp1 - tmp2
        tmp1, tmp2 = np.meshgrid(self.ZZ, speaker_zs, indexing='ij')
        distance_z = tmp1 - tmp2
        distance_matrix = np.sqrt(distance_x**2 + distance_y**2 + distance_z**2)
        greens_tensor = np.exp(1j * np.einsum('i,jk->ijk', self.wavenums, distance_matrix)) / distance_matrix

        # DIRECTIVITY FUNC.
        normal_vecs = np.array([np.sin(speaker_as)*np.cos(speaker_bs), np.sin(speaker_as)*np.sin(speaker_bs), np.cos(speaker_as)])
        normal_vecs_norm = np.linalg.norm(normal_vecs, axis=0)
        inner_matrix = np.einsum('i,j->ij', self.XX, normal_vecs[0]) + np.einsum('i,j->ij', self.YY, normal_vecs[1]) \
                        + np.einsum('i,j->ij', self.ZZ, normal_vecs[2])
        theta_matrix = np.arccos(inner_matrix / distance_matrix * normal_vecs_norm)
        inside_vessel = np.einsum('i,jk->ijk', self.wavenums, np.sin(theta_matrix)) * speaker_diameters * 0.5
        directivity_tensor = jn(1, inside_vessel) / inside_vessel
        np.nan_to_num(directivity_tensor, nan=0.5, copy=False)

        self.transfer_matrix = directivity_tensor * greens_tensor


    def getPressure(self, is_driving_func=False):
        """
        Compute the sound pressure
        """
        self._generate_transfer_matrix()
        if is_driving_func:
            driving_matrix = np.array([self.speakers.driving_function2array(i) for i in range(len(self.speakers))])
        else:
            driving_matrix = np.ones((len(self.speakers), self.frequency.size))

        self.pressure = np.einsum('ijk,ki->ji', self.transfer_matrix, driving_matrix)
        return self.pressure


    def getSPL(self, p0=10**(-5)*2.):
        """
        Compute the Sound Pressure Level
        """
        try:
            spl = 10*np.log10(np.abs(self.pressure)**2) - 20*np.log10(p0)
        except AttributeError:
            raise AttributeError("Before using this method, it is necessary to calculate the sound pressure using :method:`getPressure` first.")

        return spl



if __name__ == '__main__':

    REGION = 'cartesian'
    REGION_PARAM = {
        "x": {
            "Min": -5.0,
            "Max": 5.0,
            "step": 0.01,
            "scale": 'equidistant'
        },
        "y": 0.,
        "z": {
            "Min": 1.0,
            "Max": 11.0,
            "step": 0.01,
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

    calculator = CircleFarType(frequency=FREQUENCY, shapes_of_region=REGION, kw_descretize=REGION_PARAM)

    calculator.set_speakers(SPEAKER_LIST)
    pressure = calculator.getPressure()
    print(pressure)