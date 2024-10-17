from ._IonChannel import IonChannel
from ._MarkovIonGate import MarkovIonGate
import numpy as np


class HHIonChannelNa(IonChannel):
    """
    Sodium ion channel for the Hodgkin-Huxley model. Models the channel as 4 consequtive Markov ion gates (3 m gates and 1 h gate) that all have to be open for the channel to let potassium ions through.

    Attributes:
        g: ion channel conductance.
    """

    g = 0.0

    def __init__(self, gNa: float, v_init: float=0):
        """
        Initialize a new sodium ion channel.

        :param gNa: conductance of sodium channels when all ion gates are open (maximum conductance).
        :param v_init: initial membrane potential (relative to resting potential) at stability in mV
        """

        self._m_gate = MarkovIonGate(alpha=lambda v: ((25 - v) / 10) / (np.exp(0.1 * (25 - v)) - 1),
                                    beta=lambda v: 4 * np.exp(-v / 18),
                                    v_init=v_init)

        self._h_gate = MarkovIonGate(alpha=lambda v: 0.07 * np.exp(-v / 20),
                                    beta=lambda v: 1 / (np.exp((30 - v) / 10) + 1),
                                    v_init=v_init)

        self._g_max: float = gNa
        self.g = self._g_max * np.power(self._m_gate.state, 3) * self._h_gate.state

    def update_g(self, v, t, dt):
        self._m_gate.update(v, dt)
        self._h_gate.update(v, dt)
        self.g = self._g_max * np.power(self._m_gate.state, 3) * self._h_gate.state
        return self.g

    def reset(self, v_init: float = 0):
        self._m_gate.set_inf_state(v_init)
        self._h_gate.set_inf_state(v_init)
        self.g = self._g_max * np.power(self._m_gate.state, 3) * self._h_gate.state

