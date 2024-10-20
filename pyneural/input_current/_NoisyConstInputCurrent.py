import numpy as np
from typing import Optional
from ._ConstInputCurrent import ConstInputCurrent


class NoisyConstInputCurrent(ConstInputCurrent):
    """
    Input current with gaussian noise added. Often suitable for neurons with large number of incoming synnapses, where the current can be approximated by a gaussian noise.
    """

    _std: float = 0

    def __init__(self, N_neurons: int = 1, start_time: float = 0, end_time: float = np.inf, I: Optional[np.ndarray] = None, std: float = 0):
        """
        Initialize a new noisy input current object. Apart from its superclass parameters, takes one additional argument:

        :param std: standard deviation of the noisy current. Often should be normalized by a time interval between updates of a system (e.g. std^2 should be proportional to tau/dt).
        """
        super().__init__(N_neurons, start_time, end_time, I)
        self._std = std


    def get_current(self, t: float) -> np.ndarray:
        if(t >= self._start_time and t <= self._end_time):
            return self._I + np.random.normal(0.0, self._std, self.N_neurons)

        return np.zeros(self.N_neurons)
        
