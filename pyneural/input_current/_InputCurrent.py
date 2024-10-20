from abc import ABC, abstractmethod
import numpy as np

class InputCurrent(ABC):
    """
    A base class for different modes of input current stimulation.
    """

    def __init__(self, N_neurons: int):
        """
        Initialize an InputCurrent object.

        :param N_neurons: number of neurons for the simulation (some models can simulate different input stimulation for dofferent neurons within a single simulation)
        """
        self.N_neurons = N_neurons

    @abstractmethod
    def get_current(self, t: float) -> np.ndarray:
        """
        Get current stimulation at a particular moment in time. Returns a numpy array of stimulation values for each neuron in a simulaiton.

        :param t: time in ms
        """
        pass
