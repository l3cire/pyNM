from abc import ABC, abstractmethod
import numpy as np

class InputCurrent(ABC):
    """
    A base class for different modes of input current stimulation.
    """

    def __init__(self, N: int):
        self.N = N

    @abstractmethod
    def get_current(self, t: float) -> np.ndarray:
        """
        Get current stimulation at a particular moment in time.

        :param t: time in ms
        """
        pass
