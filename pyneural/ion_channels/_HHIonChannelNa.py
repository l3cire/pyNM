from typing import Optional
from ._IonChannel import IonChannel
from ._MarkovIonGate import MarkovIonGate
import numpy as np


class HHIonChannelNa(IonChannel):
    """
    Sodium ion channel for the Hodgkin-Huxley model. Models the channel as 4 consequtive Markov ion gates (3 m gates and 1 h gate) that all have to be open for the channel to let potassium ions through.

    Attributes:
        g: ion channel conductance for each neuron.
    """

    def __init__(self, N_neurons: int, gNa: float, V_init: Optional[np.ndarray]=None):
        """
        Initialize a new sodium ion channel for a set of neurons.

        :param N_neurons: number of neurons in a simulation.
        :param gNa: conductance of sodium channels in a single neuron when all ion gates are open (maximum conductance).
        :param V_init: numpy array containing initial membrane potentials (relative to resting potential) for each neuron at stability in mV (zero by default).
        """

        self._m_gate = MarkovIonGate(N_neurons,
                                    alpha=lambda V: ((25 - V) / 10) / (np.exp(0.1 * (25 - V)) - 1),
                                    beta=lambda V: 4 * np.exp(-V / 18),
                                    V_init=V_init)

        self._h_gate = MarkovIonGate(N_neurons,
                                    alpha=lambda V: 0.07 * np.exp(-V / 20),
                                    beta=lambda V: 1 / (np.exp((30 - V) / 10) + 1),
                                    V_init=V_init)

        self._g_max: float = gNa
        self.g: np.ndarray = self._g_max * np.power(self._m_gate.state, 3) * self._h_gate.state

    def update_g(self, V: np.ndarray, t, dt) -> np.ndarray:
        self._m_gate.update(V, dt)
        self._h_gate.update(V, dt)
        self.g = self._g_max * np.power(self._m_gate.state, 3) * self._h_gate.state
        return self.g

    def reset(self, V_init: Optional[np.ndarray] = None):
        self._m_gate.set_inf_state(V_init)
        self._h_gate.set_inf_state(V_init)
        self.g = self._g_max * np.power(self._m_gate.state, 3) * self._h_gate.state

