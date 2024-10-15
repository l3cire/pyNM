from abc import ABC, abstractmethod

class InputCurrent(ABC):
    """
    A base class for different modes of input current stimulation.
    """

    @abstractmethod
    def get_current(self, t: float) -> float:
        """
        Get current stimulation at a particular moment in time.

        :param t: time in ms
        """
        pass
