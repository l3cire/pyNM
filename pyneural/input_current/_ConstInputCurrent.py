import numpy as np
from ._InputCurrent import InputCurrent

class ConstInputCurrent(InputCurrent):
    """
    Constant current stimulation.
    """

    _start_time: float
    _end_time: float
    _I: float

    def __init__(self, N_neurons: int = 1, start_time: float = 0.0, end_time: float = np.inf, I: float = 0.0):
        """
        :param start_time: start time of the external stimulation in ms.
        :param end_time: end time of the external stimulation in ms.
        :param I: value of the stimulation in mA.
        """
        super().__init__(N_neurons)
        self._start_time = start_time
        self._end_time = end_time
        self._I = I

    def get_current(self, t: float) -> np.ndarray:
        if(t >= self._start_time and t <= self._end_time):
            return np.zeros(self.N) + self._I
        return np.zeros(self.N)

