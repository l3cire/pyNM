from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class IonChannel(ABC):
    """
    Ion channel base class. Ion channels are a key instrument by which neuron cells regulate their membrane potential. Different ion channels can vary their conductance based on the circumstances, which allows different ions to flow through the membrane.

    Attributes:
        g: conductace of a channel for each neuron.
    """

    def __init__(self, N_neurons: int = 1):
        """
        Initialize a new ion channel object.

        :param N_neurons: number of neurons in a simulation.
        """
        self.N_neurons = N_neurons
        self.g = np.zeros(N_neurons)
    
    @abstractmethod
    def update_g(self, V: np.ndarray, t: float, dt: float) -> np.ndarray:
        """
        Updates the conductance of the channel for each neuron based on current time and membrane potential for each neuron. Returns new conductance for each neuron.

        :param V: numpy array containing current membrane potentials (relative to the resting potential) for each neuron in mV.
        :param t: current time in ms.
        :param dt: the time interval between two consecutive updates in ms.
        """
        pass

    @abstractmethod
    def reset(self, V_init: Optional[np.ndarray] = None):
        """
        Resets the conductance of each neuron to the stable level, given that the membrane potential is constant.

        :param V_init: numpy array containing membrane potentials (relative to the resting potential) for each neuron in mV.
        """
        pass
