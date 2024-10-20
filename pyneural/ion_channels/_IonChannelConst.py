from typing import Optional
from ._IonChannel import IonChannel
import numpy as np


class IonChannelConst(IonChannel):
    """
    Ion channel with constant conductance.

    Attributes:
        g: ion channel conductance.
    """

    def __init__(self, N_neurons: int, g: float):
        """
        Initialize a new ion channel with constant conductance.

        :param g: conductance of this channel.
        """
        super().__init__(N_neurons)
        self.g += g

    def update_g(self, V: np.ndarray, t: float, dt: float) -> np.ndarray: 
        return self.g

    def reset(self, V_init: Optional[np.ndarray] = None):
        return

