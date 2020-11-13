
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



class ConditionBase(object):

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



class CalculationBase(object):

    def get_pressure(self, driving_signals=None, is_driven=False, activation_vector=None):
        """ a method for calculating sound pressure based on a linear equation
        using transfer matrix and driving functions on sources.

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

        if activation_vector:
            if is_driven:
                self.pressure = np.einsum('ijk,kk,ik->ij', self.transfer_matrix, np.diag(activation_vector), driving_signals)
            else:
                pseudo_driving_signals = np.ones((self.frequency.size, len(self.sources)))
                self.pressure = np.einsum('ijk,kk,ik->ij', self.transfer_matrix, np.diag(activation_vector), pseudo_driving_signals)
        else:
            if is_driven:
                self.pressure = np.einsum('ijk,ik->ij', self.transfer_matrix, driving_signals)
            else:
                pseudo_driving_signals = np.ones((self.frequency.size, len(self.sources)))
                self.pressure = np.einsum('ijk,ik->ij', self.transfer_matrix, pseudo_driving_signals)

        return self.pressure


    def get_SPL(self, p0=10**(-5)*2., **kwargs):
        """ a method for calculating the Sound Pressure Level(SPL) [dB]

        Parameters
        ----------
        p0 : float, optional, default 10**(-5)*2.
            Reference sound pressure.
        **kwargs : :meth:`get_pressure()` properties.

        Returns
        -------
        numpy.ndarray-(F, M)
            SPL distribution
        """
        try:
            return 10*np.log10(np.abs(self.pressure)**2) - 20*np.log10(p0)
        except AttributeError:
            return 10*np.log10(np.abs(self.get_pressure(**kwargs)**2) - 10*np.log10(p0))


    def get_amplitude(self, **kwargs):
        """ a method for calculating the amplitude of sound pressure

        Returns
        -------
        numpy.ndarray-(F, M)
            Amplitude distribution
        """
        try:
            return np.abs(self.pressure)
        except AttributeError:
            return np.abs(self.get_pressure(**kwargs))


    def get_phase(self, **kwargs):
        """ a method for calculating the amplitude of sound pressure

        Returns
        -------
        numpy.ndarray-(F, M)
            Phase distribution
        """
        try:
            return np.angle(self.pressure)
        except AttributeError:
            return np.angle(self.get_pressure(**kwargs))



class PointSourceFreeFieldCalculator(CalculationBase, ConditionBase):
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



class CircularSourceFreeFieldCalculator(CalculationBase, ConditionBase):
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