"""
A Module for calculating sound fields.
"""


class SpeakerParams(dict):
    """
    Container of speakers and their parameters.

    Notes
    -----
    This class holds parameters of the speaker array in :dict: type.
    Each loudspeaker is assigned a key of :int:, which is 0 or greater,
    and its value is stored in a :dict: with the corresponding parameters of the loudspeaker.
    `__init__` method needs to be given the shape of the speaker, like point, line or circular,
    which is shared within an instance. That is, it's not possible to store two different types
    of speakers in a single instance. The definable speaker shapes and the parameters
    required for each are as follows,

        - 'point' : ideal point source
            'x', 'y', 'z' : the position of the speaker on cartesian coordinate
        - 'line' : ideal line source
            'length' : the total length of the line source
            'x', 'y', 'z' : the position of the centre of line on cartesian coordinate
            'alpha' : the angle of line vector from z axis [rad]
            'beta' : the angle of line vector from x axis [rad]
        - 'circular' : ideal circular source
            'diameter' : the diameter of the circular source
            'x', 'y', 'z' : the position of the centre of circle on cartesian coordinate
            'alpha' : the angle of normal vector of circle plane from z axis [rad]
            'beta' : the angle of normal vector of circle plane from x axis [rad]

    Another necessary key(parameter) that is common to all is 'driving function',
    of which value can be set as dictionary corresponding to frequencies.
    If there's no need to set driving functions, it is recommended to set the value 1.0.
    This class also provides the method to set driving functions.


    Parameters
    ----------
    shape : {'point', 'line', 'circular'}
        Shapes of speakers.
    speaker_param_list : numpy.ndarray
        1-D Array that is combined parameters of all loudspeakers into one dimension.
        All parameters must be in the order in which they appear in :attribute:`param_keys`,
        see more documentation of attributes below.


    Attributes
    ----------
    * shape
    * param_keys
        - 'point' source : ('driving function', 'x', 'y', 'z')
        - 'line' source : ('driving function', 'length', 'x', 'y', 'z', 'alpha', 'beta')
        - 'circular' source : ('driving function', 'diameter', 'x', 'y', 'z', 'alpha', 'beta')
    """

    def __init__(self, shape, speaker_param_list=None):

        self.shape = shape
        if shape == 'point':
            self.param_keys = ('driving function', 'x', 'y', 'z')
        elif shape == 'line':
            self.param_keys = ('driving function', 'length', 'x', 'y', 'z', 'alpha', 'beta')
        elif shape == 'circular':
            self.param_keys = ('driving function', 'diameter', 'x', 'y', 'z', 'alpha', 'beta')
        else:
            raise ValueError(f'argument *shape*={shape} is invalid.')

        if speaker_param_list is not None:
            is_divisible = len(speaker_param_list) % len(self.param_keys) == 0

            if is_divisible:
                speaker_num = int(len(speaker_param_list) / len(self.param_keys))
                speaker_lists = speaker_param_list.reshape(speaker_num, len(self.param_keys)).tolist()

                self.update({key: self._param_list2dict(value) for key, value in enumerate(speaker_lists)})

            else:
                raise ValueError('argument *speaker_param_list* cannot be transformed into the given *shape*.')


    def _param_list2dict(self, param_list):
        return {key: value for key, value in zip(self.param_keys, param_list)}


    def add_speaker(self, speaker_param_list):
        is_divisible = len(speaker_param_list) % len(self.param_keys) == 0

        if is_divisible:
            speaker_num = int(len(speaker_param_list) / len(self.param_keys))
            speaker_lists = speaker_param_list.reshape(speaker_num, len(self.param_keys)).tolist()

            self.update({key+len(self): self._param_list2dict(value) for key, value in enumerate(speaker_lists)})

        else:
            raise ValueError('argument *speaker_param_list* cannot be transformed into the given *shape*.')


    def get_arbit_param(self, keys, speaker_no=None):
        if speaker_no is None:
            return [self[no][keys] for no in range(len(self))]


    def set_driving_function(self, speaker_no, frequency=None, value=1.0):
        self[speaker_no]['driving function'] = value if not hasattr(value, "__iter__") \
                                                else {f: val for f, val in zip(frequency, value)}


    def driving_function2array(self, speaker_no):
        if isinstance(self[speaker_no]['driving function'], dict):
            return [val for val in self[speaker_no]['driving function'].values()]