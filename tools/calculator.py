
"""
The :mod:`calculator` module is to
"""

import numpy as np
from scipy.special import jn
from enum import Enum, auto

try:
    from . import source
    from . import receiver
except ImportError:
    import source
    import receiver



class _ConditionSettingTools(object):

    def __init__(self, sources, receivers, frequency, velocity=344.):

        if type(sources) == source.SourceParams and sources.shape == self.SOURCE_SHAPE:
            self.sources = sources
        else:
            raise ValueError(f":arg:`sources` is needed to be an object from :class:`source.SourceParams, \
                                and must have an :attr:`shape`={self.SOURCE_SHAPE}.")

        if type(receivers) == receiver.ReceiverParams:
            self.receivers = receivers
        else:
            raise ValueError(f":arg:`receivers` is needed to be an object from :class:`receiver.ReceiverParams.")

        if type(frequency) == np.ndarray:
            self.frequency = frequency
        else:
            raise ValueError(f":arg:`frequency` must be numpy array.")

        self.velocity = velocity
        self.wavenums = 2 * np.pi * self.frequency / self.velocity
        self._generate_transfer_matrix()


    def set_sources(self, sources):
        """ Method for update the sources for calculation.
        """

        if type(sources) == source.SourceParams and sources.shape == self.SOURCE_SHAPE:
            self.sources = sources
        else:
            raise ValueError(f":arg:`sources` is needed to be an object from :class:`source.SourceParams, \
                                and must have an :attr:`shape`={self.SOURCE_SHAPE}.")

        self._generate_transfer_matrix()


    def set_receivers(self, receivers):
        """ Method for update the receivers for calculation.
        """
        if type(receivers) == receiver.ReceiverParams:
            self.receivers = receivers
        else:
            raise ValueError(f":arg:`receivers` is needed to be an object from :class:`receiver.ReceiverParams.")

        self._generate_transfer_matrix()


    def set_frequency(self, frequency):
        """ Method for update the receivers for calculation.
        """
        if type(frequency) == np.ndarray:
            self.frequency = frequency
        else:
            raise ValueError(f":arg:`frequency` must be numpy array.")

        self.wavenums = 2 * np.pi * self.frequency / self.velocity
        self._generate_transfer_matrix()



class _CalculationTools(object):

    def get_pressure(self, driving_signals=None, is_driven=False):
        """
        Method for calculating sound pressure based on a linear equation using transfer matrix and driving functions on sources.

        Parameters
        ----------
        driving_signals : numpy.ndarray-(F, L), optional.
            The signals weightend on sources, and next :arg:`is_driven` must be changed to :bool:`True` to use this signals.
        is_driven : bool
            Whether the sources to be driven.

        Returns
        -------
        numpy.ndarray-(F, M)
            Sound pressure distribution
        """

        if is_driven:
            self.pressure = np.einsum('ijk,ik->ij', self.transfer_matrix, driving_signals)
        else:
            pseudo_driving_signals = np.ones((self.frequency.size, len(self.sources)))
            self.pressure = np.einsum('ijk,ik->ij', self.transfer_matrix, pseudo_driving_signals)

        return self.pressure


    def get_driving_signals_by_pressure_matching(self, p_des, regularize=True, regularization_param=0.01):
        """
        Method for calculating driving signals based on multipoint sound pressure control.

        Parameters
        ----------
        p_des : numpy.ndarray-(F, M_mic)
            The desired sound pressures.
        regularize : bool, default True.
            Whether to regularize.
        regularization_param : float, optional, defaut 0.01.
            The regularization parameter.
        """
        inv_transfer_matrix = np.array([np.linalg.pinv(g_omega) for g_omega in self.transfer_matrix])

        if regularize == True:
            return np.array([np.linalg.inv(inv_g_omega @ g_omega + regularization_param * np.eye(len(self.sources))) @ inv_g_omega @ p_des_omega
                            for g_omega, inv_g_omega, p_des_omega in zip(self.transfer_matrix, inv_transfer_matrix, p_des)])
        else:
            return np.array([inv_g_omega @ p_des_omega for inv_g_omega, p_des_omega in zip(inv_transfer_matrix, p_des)])


    def get_SPL(self, p0=10**(-5)*2.):
        """
        Compute the Sound Pressure Level(SPL) [dB]

        Parameters
        ----------
        p0 : float, optional, default 10**(-5)*2.

        Returns
        -------
        numpy.ndarray-(F, M)
            SPL distribution

        Raises
        ------
        AttributeError
            Before using this method, it is necessary to calculate the sound pressure using :meth:`get_pressure` first.
        """
        try:
            spl = 10*np.log10(np.abs(self.pressure)**2) - 20*np.log10(p0)
        except AttributeError:
            raise AttributeError("Before using this method, it is necessary to calculate the sound pressure using :meth:`get_pressure` first.")

        return spl


    def get_amplitude(self):
        """
        Compute the amplitude of sound pressure

        Returns
        -------
        numpy.ndarray-(F, M)
            Amplitude distribution

        Raises
        ------
        AttributeError
            Before using this method, it is necessary to calculate the sound pressure using :meth:`get_pressure` first.
        """
        try:
            amplitude = np.abs(self.pressure)
        except AttributeError:
            raise AttributeError("Before using this method, it is necessary to calculate the sound pressure using :meth:`get_pressure` first.")

        return amplitude


    def get_phase(self):
        """
        Compute the amplitude of sound pressure

        Returns
        -------
        numpy.ndarray-(F, M)
            Phase distribution

        Raises
        ------
        AttributeError
            Before using this method, it is necessary to calculate the sound pressure using :meth:`get_pressure` first.
        """
        try:
            phase = np.angle(self.pressure)
        except AttributeError:
            raise AttributeError("Before using this method, it is necessary to calculate the sound pressure using :meth:`get_pressure` first.")

        return phase



class PointSourceFreeFieldCalculator(_CalculationTools, _ConditionSettingTools):
    """
    Calculating sound field from multiple point sources.

    Parameters
    ----------
    sources : source.SourceParams
        Object generated from the :class:`source.SourceParams` which represents the placement data of source array
    receivers : receiver.ReceiverParams
        Object generated from the :class:`receiver.ReceiverParams` which represents the placement data of control points.
    frequency : numpy.ndarray
        Frequencies to be calculated, which must be :type:'numpy.ndarray'.

    Notes
    -----
    Once the instance is created based on the class, the transfer function matrix will be calculated.
    Thereafter, any parameter changed by any :meth: in :class:`_ConditionSettingTools` will update the transfer function matrix accordingly.

    About :attr:`transfer_matrix`
    -----------------------------
    The transfer functions matrix is represented as a 3-dimensional tensor of number of frequency(F), receivers(M), sources(L),
    which is stored in :type:`numpy.ndarray`.

    ===============================================
        self.transfer_matrix.shape => (F, M, L)
    ===============================================
    """

    SOURCE_SHAPE = source.SourceCategory.POINT

    def __init__(self, sources, receivers, frequency, velocity=344.):

        super().__init__(sources=sources, receivers=receivers, frequency=frequency, velocity=velocity)


    def _generate_transfer_matrix(self):

        source_xs = np.array(self.sources.get_arbit_param('x'))
        source_ys = np.array(self.sources.get_arbit_param('y'))
        source_zs = np.array(self.sources.get_arbit_param('z'))

        receiver_xs = np.array(self.receivers['x'].values)
        receiver_ys = np.array(self.receivers['y'].values)
        receiver_zs = np.array(self.receivers['z'].values)

        distance_x = source_xs - receiver_xs[:, np.newaxis]
        distance_y = source_ys - receiver_ys[:, np.newaxis]
        distance_z = source_zs - receiver_zs[:, np.newaxis]

        distance_matrix = np.sqrt(distance_x**2 + distance_y**2 + distance_z**2)
        self.transfer_matrix = np.exp(1j * np.einsum('i,jk->ijk', self.wavenums, distance_matrix)) / distance_matrix[np.newaxis, :]



class CircularSourceFreeFieldCalculator(_CalculationTools, _ConditionSettingTools):
    """
    Calculating sound field from multiple circular sources.

    Parameters
    ----------
    sources : source.SourceParams
        Object generated from the :class:`source.SourceParams` which represents the placement data of source array
    receivers : receiver.ReceiverParams
        Object generated from the :class:`receiver.ReceiverParams` which represents the placement data of control points.
    frequency : numpy.ndarray
        Frequencies to be calculated, which must be :type:'numpy.ndarray'.

    Notes
    -----
    Once the instance is created based on the class, the transfer function matrix will be calculated.
    Thereafter, any parameter changed by any :meth: in :class:`_ConditionSettingTools` will update the transfer function matrix accordingly.

    About :attr:`transfer_matrix`
    -----------------------------
    The transfer functions matrix is represented as a 3-dimensional tensor of number of frequency(F), receivers(M), sources(L),
    which is stored in :type:`numpy.ndarray`.

    ===============================================
        self.transfer_matrix.shape => (F, M, L)
    ===============================================
    """

    SOURCE_SHAPE = source.SourceCategory.CIRCULAR

    def __init__(self, sources, receivers, frequency, velocity=344.):

        super().__init__(sources=sources, receivers=receivers, frequency=frequency, velocity=velocity)


    def _generate_transfer_matrix(self):

        source_diameters = np.array(self.sources.get_arbit_param('Diameter'))
        source_xs = np.array(self.sources.get_arbit_param('x'))
        source_ys = np.array(self.sources.get_arbit_param('y'))
        source_zs = np.array(self.sources.get_arbit_param('z'))
        source_as = np.array(self.sources.get_arbit_param('Elevation'))
        source_bs = np.array(self.sources.get_arbit_param('Azimuth'))

        receiver_xs = np.array(self.receivers['x'].values)
        receiver_ys = np.array(self.receivers['y'].values)
        receiver_zs = np.array(self.receivers['z'].values)

        # GREEN FUNC.
        distance_x = source_xs - receiver_xs[:, np.newaxis]
        distance_y = source_ys - receiver_ys[:, np.newaxis]
        distance_z = source_zs - receiver_zs[:, np.newaxis]

        distance_matrix = np.sqrt(distance_x**2 + distance_y**2 + distance_z**2)
        greens_tensor = np.exp(1j * np.einsum('i,jk->ijk', self.wavenums, distance_matrix)) / distance_matrix[np.newaxis, :]

        # DIRECTIVITY FUNC.
        normal_vecs = np.array([np.sin(source_as) * np.cos(source_bs),
                                np.sin(source_as) * np.sin(source_bs),
                                np.cos(source_as)])
        normal_vecs_norm = np.linalg.norm(normal_vecs, axis=0)
        inner_matrix = normal_vecs[0] * receiver_xs[:, np.newaxis] \
                        + normal_vecs[1] * receiver_ys[:, np.newaxis] \
                        + normal_vecs[2] * receiver_zs[:, np.newaxis]
        theta_matrix = np.arccos(inner_matrix / (normal_vecs_norm[np.newaxis, :] * distance_matrix))
        inside_vessel = np.einsum('i,jk->ijk', self.wavenums, np.sin(theta_matrix)) * source_diameters[np.newaxis, np.newaxis, :] * 0.5
        directivity_tensor = jn(1, inside_vessel) / inside_vessel
        np.nan_to_num(directivity_tensor, nan=0.5, copy=False)

        self.transfer_matrix = directivity_tensor * greens_tensor



class FieldCategory(Enum):
    SOURCE = auto()
    PLANE = auto()



class ReproductionApproach(Enum):
    PRESSURE_MATCHING = auto()
    HOA = auto()



class ReproductionTool(object):

    def __init__(self, primary_field_type, primary_source, fields, control_points, secondary_sources, frequency, velocity=344.):

        self.primary_field_type = primary_field_type
        self.primary_source = primary_source
        self.fields = fields
        self.control_points = control_points
        self.secondary_sources = secondary_sources

        self.frequency = frequency
        self.velocity = velocity

        if primary_field_type == FieldCategory.SOURCE:
            self.primary_field_generator = self._define_calculator(sources=primary_source, receivers=fields)
            self.primary_field_collector = self._define_calculator(sources=primary_source, receivers=control_points)
        else:
            raise ValueError(f"Sorry, method for the primary field of plane is now building...")

        self.system_representer = self._define_calculator(sources=secondary_sources, receivers=control_points)
        self.secondary_field_generator = self._define_calculator(sources=secondary_sources, receivers=fields)

        self.p_des_on_fields = self.primary_field_generator.get_pressure()
        self.p_des_at_mics = self.primary_field_collector.get_pressure()


    def _define_calculator(self, sources, receivers):

        if sources.shape == source.SourceCategory.POINT:
            return PointSourceFreeFieldCalculator(sources=sources, receivers=receivers, frequency=self.frequency, velocity=self.velocity)
        elif sources.shape == source.SourceCategory.CIRCULAR:
            return CirucularSourceFreeFieldCalculator(sources=sources, receivers=receivers, frequency=self.frequency, velocity=self.velocity)
        else:
            raise ValueError(f"The source tyep of {source.shape} is not supported.")


    def set_control_points(self, control_points):

        self.control_points = control_points

        self.primary_field_collector.set_receivers(control_points)
        self.system_representer.set_receivers(control_points)

        self.p_des_at_mics = self.primary_field_collector.get_pressure()


    def set_fields(self, fields):

        self.fields = fields

        self.primary_field_generator.set_receivers(fields)
        self.secondary_field_generator.set_receivers(fields)

        self.p_des_on_fields = self.primary_field_generator.get_pressure()


    def set_secondary_sources(self, secondary_sources):

        self.secondary_sources = secondary_sources

        self.system_representer.set_sources(secondary_sources)
        self.secondary_field_generator.set_sources(secondary_sources)


    def get_driving_signals(self, approach=ReproductionApproach.PRESSURE_MATCHING, *args, **kwargs):

        if approach == ReproductionApproach.PRESSURE_MATCHING:
            driving_signals = self.system_representer.get_driving_signals_by_pressure_matching(self.p_des_at_mics, *args, **kwargs)
        else:
            raise ValueError(f"The {approach} for sound field reproduction is not supported now.")

        return driving_signals


    def get_secondary_pressure(self, *args, **kwargs):

        driving_signals = self.get_driving_signals(*args, **kwargs)
        return self.secondary_field_generator.get_pressure(is_driven=True, driving_signals=driving_signals)


    def get_normalized_error(self, take_average=False, *args, **kwargs):

        driving_signals = self.get_driving_signals(*args, **kwargs)
        p_rep_on_fields = self.secondary_field_generator.get_pressure(is_driven=True, driving_signals=driving_signals)
        normalized_error = 10 * np.log10(np.abs(self.p_des_on_fields - p_rep_on_fields)**2 / np.abs(self.p_des_on_fields)**2)

        if take_average:
            return np.average(normalized_error, axis=1)
        else:
            return normalized_error