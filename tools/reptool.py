
""" The :mod:`tools.reptool` provides the modules
"""

from .calculator import *
from .source import *
from .receiver import *


class FieldCategory(Enum):
    SPHERICAL = auto()
    PLANE = auto()


class Estimator(Enum):
    PM_LS = auto()
    PM_Lasso = auto()
    HOA = auto()


class ReproductionTool(object):

    def __init__(self, primary_field_type,
                        primary_source,
                        fields,
                        control_points,
                        secondary_sources,
                        frequency,
                        estimator,
                        velocity=344.):

        self.primary_field_type = primary_field_type
        self.primary_source = primary_source
        self.fields = fields
        self.control_points = control_points
        self.secondary_sources = secondary_sources

        self.frequency = frequency
        self.velocity = velocity

        if primary_field_type == FieldCategory.SPHERICAL:
            self.primary_field_generator = self._define_calculator(sources=primary_source,
                                                                    receivers=fields)
            self.primary_field_collector = self._define_calculator(sources=primary_source,
                                                                    receivers=control_points)
        else:
            raise NotImplementedError("Method for the plane type of primary field is now building...")

        self.system_representer = self._define_calculator(sources=secondary_sources,
                                                            receivers=control_points)
        self.secondary_field_generator = self._define_calculator(sources=secondary_sources,
                                                            receivers=fields)

        self.p_des_on_fields = self.primary_field_generator.get_pressure()
        self.p_des_at_mics = self.primary_field_collector.get_pressure()

        self.transfer_matrix = self.system_representer.transfer_matrix

        if estimator == Estimator.PM_LS:
            self.get_driving_signals = self.get_driving_signals_by_PM_LS
        elif estimator == Estimator.PM_Lasso:
            self.get_driving_signals = self.get_driving_signals_by_PM_Lasso
        elif estimator == Estimator.HOA:
            self.get_driving_signals = self.get_driving_signals_by_HOA


    def _define_calculator(self, sources, receivers):

        if sources.shape == source.SourceCategory.POINT:
            return PointSourceFreeFieldCalculator(sources=sources, receivers=receivers,
                                                frequency=self.frequency, velocity=self.velocity)
        elif sources.shape == source.SourceCategory.CIRCULAR:
            return CirucularSourceFreeFieldCalculator(sources=sources, receivers=receivers,
                                                frequency=self.frequency, velocity=self.velocity)
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


    def get_secondary_pressure(self, activation_vector=None, *args, **kwargs):

        driving_signals = self.get_driving_signals(activation_vector=activation_vector, *args, **kwargs)
        return self.secondary_field_generator.get_pressure(is_driven=True, driving_signals=driving_signals)


    def get_normalized_error(self, take_average=False, *args, **kwargs):

        p_rep_on_fields = self.get_secondary_pressure(*args, **kwargs)
        normalized_error = 10 * np.log10(np.abs(self.p_des_on_fields - p_rep_on_fields)**2 / np.abs(self.p_des_on_fields)**2)

        if take_average:
            return np.average(normalized_error, axis=1)
        else:
            return normalized_error


    def get_driving_signals_by_PM_LS(self, activation_vector=None, regularize=False, reg_param=0.01):
        """ a method for calculating driving signals by
        pressure matching based on least square.

        Parameters
        ----------
        regularize : bool, default True.
            Whether to regularize.
        reg_param : float, optional, defaut 0.01.
            The regularization parameter.
        """
        if activation_vector:
            transfer_matrix = np.einsum('ijk,kk->ijk', self.transfer_matrix, np.diag(activation_vector))
        else:
            transfer_matrix = self.transfer_matrix

        inv_transfer_matrix = np.array([np.linalg.pinv(g_omega) for g_omega in transfer_matrix])

        if regularize:
            return np.array([np.linalg.inv(inv_g_omega @ g_omega
                                + reg_param * np.eye(len(self.secondary_sources))) @ inv_g_omega @ p_des_omega
                            for g_omega, inv_g_omega, p_des_omega
                            in zip(transfer_matrix, inv_transfer_matrix, self.p_des_at_mics)])
        else:
            return np.array([inv_g_omega @ p_des_omega for inv_g_omega, p_des_omega
                            in zip(inv_transfer_matrix, self.p_des_at_mics)])


    def get_driving_signals_by_PM_Lasso(self, reg_param):
        """
        """
        return NotImplementedError


    def get_driving_signals_by_HOA(self):
        """
        """
        return NotImplemetedError
