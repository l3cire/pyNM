from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class IonChannel(ABC):
    """
    Ion channel base class. Ion channels are a key instrument by which neuron cells regulate their membrane potential. Different ion channels can vary their conductance based on the circumstances, which allows different ions to flow through the membrane.

    Attributes:
        g: conductace of a channel.
    """

    def __init__(self, N_neurons: int = 1):
        self.N_neurons = N_neurons
        self.g = np.zeros(N_neurons)
    
    @abstractmethod
    def update_g(self, V: np.ndarray, t: float, dt: float) -> np.ndarray:
        """
        Updates the conductance of the channel based on current time and membrane potential. Returns new conductance.

        :param v: current membrane potential relative to the resting potential in mV.
        :param t: current time in ms.
        :param dt: the time interval between two consecutive updates in ms.
        """
        pass

    @abstractmethod
    def reset(self, V_init: Optional[np.ndarray] = None):
        """
        Resets the conductance to the stable level, given that the membrane potential is constant.

        :param v_init: the membrane potential (relative to the resting potential) in mV.
        """
        pass
