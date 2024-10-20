from typing import Optional
import numpy as np
from ._InputCurrent import InputCurrent

class ConstInputCurrent(InputCurrent):
    """
    Constant current stimulation. Each neuron gets the constant current specified by the input.
    """

    _start_time: float
    _end_time: float
    _I: np.ndarray

    def __init__(self, N_neurons: int = 1, start_time: float = 0.0, end_time: float = np.inf, I: Optional[np.ndarray] = None):
        """
        :param N_neurons: number of neurons in a simulation.
        :param start_time: start time of the external stimulation in ms.
        :param end_time: end time of the external stimulation in ms.
        :param I: numpy array containing values of external stimulation for each neuron. 
        """
        super().__init__(N_neurons)
        self._start_time = start_time
        self._end_time = end_time
        if I is None:
            self._I = np.zeros(N_neurons)
        else:
            self._I = I

    def get_current(self, t: float) -> np.ndarray:
        if(t >= self._start_time and t <= self._end_time):
            return self._I.copy()
        return np.zeros(self.N_neurons)

