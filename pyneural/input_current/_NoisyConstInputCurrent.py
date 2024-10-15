import numpy as np
from ._ConstInputCurrent import ConstInputCurrent


class NoisyConstInputCurrent(ConstInputCurrent):

    _std: float = 0

    def __init__(self, start_time: float = 0, end_time: float = np.inf, I: float = 0, std: float = 0):
        super().__init__(start_time, end_time, I)
        self._std = std


    def get_current(self, t: float) -> float:
        if(t >= self._start_time and t <= self._end_time):
            return np.random.normal(self._I, self._std)

        return 0
        
