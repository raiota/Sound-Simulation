"""
Calculating sound field from ideal line source
"""

from tqdm import tqdm
import numpy as np
import quadpy

try:
    from __init__ import SpeakerParams
except:
    from .__init__ import SpeakerParams

try:
    from define_region import InterestRegion
except:
    from .define_region import InterestRegion


class IdealLine(InterestRegion):
    """
    Calculating sound field from ideal line source

    Parmaeters
    ----------
    frequency : array_like
        List of evaluating frequency [Hz]
    velocity : float, Default 344.0
        Sound velocity [m/s]

    Other Parameters
    ----------------
    **region_param
        This method also takes the keyword arguments for the base :class:`InterestRegion`.
        See more documentation it.
    """

    def __init__(self, frequency=None, velocity=344., scheme=None, **region_param):

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

        if scheme is None:
            self.scheme = quadpy.line_segment.gauss_patterson(5)
        else:
            self.scheme = scheme


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


    def set_line(self, speaker_param_list):
        """
        Setting a line speaker parameters

        Parameters
        ----------
        speaker_param_list : 1D-numpy.ndarray
            1-D Array that is combined parameters of all loudspeakers into one dimension.

        Notes
        -----
        For more details on the argument 'speaker_param_list', see base :class:`SpeakerParams`.

        Other Parameters
        ----------------
        **driving_kwargs
        """
        self.line = SpeakerParams(shape='line', speaker_param_list=speaker_param_list)


    def getPressure(self, is_driving_func=False, correction_term=1.0, mesh=True):
        """
        Compute the sound pressure
        """
        line_drive = np.array(driving_function2array(0)) if is_driving_func \
            else np.array([correction_term] * self.frequency.size)
        line_length = np.array(self.line.get_arbit_param("length"))
        line_x = self.line.get_arbit_param("x")[0]
        line_y = self.line.get_arbit_param("y")[0]
        line_z = self.line.get_arbit_param("z")[0]
        line_a = self.line.get_arbit_param("alpha")[0]
        line_b = self.line.get_arbit_param("beta")[0]

        term_on_x = np.sin(line_a) * np.cos(line_b)
        term_on_y = np.sin(line_a) * np.sin(line_b)
        term_on_z = np.cos(line_a)

        def green(l):
            distance = np.sqrt((line_x + l*term_on_x - x)**2 + (line_y + l*term_on_y - y)**2 + (line_z + l*term_on_z - z)**2)
            return np.exp(1j * k * distance) / distance

        if mesh:
            self.cart2meshgrid(flatten=True)
            self.pressure = np.zeros((self.XX.size, self.frequency.size), dtype='complex128')

            for i, x, y, z in tqdm(zip(range(self.XX.size), self.XX, self.YY, self.ZZ), total=self.XX.size, desc="[calc.]"):
                for j, k in enumerate(self.wavenums):
                    self.pressure[i,j] = line_drive[j] * self.scheme.integrate(green, [-line_length*.5, line_length*.5])

        else:
            self.pressure = np.zeros((self.X.size, self.frequency.size), dtype='complex128')

            for i, x, y, z in tqdm(zip(range(self.X.size), self.X, self.Y, self.Z), total=self.X.size, desc="[calc.]"):
                for j, k in enumerate(self.wavenums):
                    self.pressure[i,j] = line_drive[j] * self.scheme.integrate(green, [-line_length*.5, line_length*.5])

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




if __name__ == '__main__':

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

    LINE_LIST = np.array([1., 8., 0., 0., 0., np.pi*.5, 0.])

    calculator = IdealLine(frequency=FREQUENCY, shapes_of_region='cartesian', kw_descretize=REGION_PARAM)

    calculator.set_line(LINE_LIST)

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