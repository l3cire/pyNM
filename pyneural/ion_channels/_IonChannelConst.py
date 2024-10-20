from typing import Optional
from ._IonChannel import IonChannel
import numpy as np


class IonChannelConst(IonChannel):
    """
    Ion channel with constant conductance.

    Attributes:
        g: ion channel conductance for each neuron.
    """

    def __init__(self, N_neurons: int, g: float):
        """
        Initialize a new ion channel with constant conductance.

        :param N_neurons: number of neurons in a simulation.
        :param g: conductance of this channel (same for each neuron).
        """
        super().__init__(N_neurons)
        self.g += g

    def update_g(self, V: np.ndarray, t: float, dt: float) -> np.ndarray: 
        return self.g

    def reset(self, V_init: Optional[np.ndarray] = None):
        return

