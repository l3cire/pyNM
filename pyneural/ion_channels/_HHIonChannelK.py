from ._IonChannel import IonChannel
from ._MarkovIonGate import MarkovIonGate
import numpy as np


class HHIonChannelK(IonChannel):
    """
    Potassium ion channel for the Hodgkin-Huxley model. Models the channel as 4 consequtive Markov ion gates that all have to be open for the channel to let potassium ions through.

    Attributes:
        g: ion channel conductance.
    """
    g: float = 0.0

    def __init__(self, gK: float, v_init: float = 0):
        """
        Initialize a new potassium ion channel.

        :param gK: conductance of potassium channels when all ion gates are open (maximum conductance).
        :param v_init: initial membrane potential (relative to resting potential) at stability in mV
        """
        self._n_gate = MarkovIonGate(alpha=lambda v: ((10 - v) / 100) / (np.exp(0.1 * (10 - v)) - 1),
                                    beta=lambda v: 0.125 * np.exp(-v / 80),
                                    v_init=v_init)

        self._g_max: float = gK
        self.g = self._g_max * np.power(self._n_gate.state, 4)

    def update_g(self, v, t, dt) -> float:
        self._n_gate.update(v, dt)
        self.g = self._g_max * np.power(self._n_gate.state, 4)
        return self.g
    
    def reset(self, v_init: float = 0):
        self._n_gate.set_inf_state(v_init)
        self.g = self._g_max * np.power(self._n_gate.state, 4)

