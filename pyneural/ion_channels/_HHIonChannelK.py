from typing import Optional
from ._IonChannel import IonChannel
from ._MarkovIonGate import MarkovIonGate
import numpy as np


class HHIonChannelK(IonChannel):
    """
    Potassium ion channel for the Hodgkin-Huxley model. Models the channel as 4 consequtive Markov ion gates that all have to be open for the channel to let potassium ions through.

    Attributes:
        g: ion channel conductance for each neuron.
    """

    def __init__(self, N_neurons: int, gK: float, V_init: Optional[np.ndarray] = None):
        """
        Initialize a new potassium ion channel for a set of neurons.

        :param N_neurons: number of neurons in a simulation.
        :param gK: conductance of potassium channels of a single neuron when all ion gates are open (maximum conductance).
        :param V_init: numpy array containing initial membrane potentials (relative to resting potential) for each neuron at stability in mV (zero by default).
        """
        super().__init__(N_neurons)


        self._n_gate = MarkovIonGate(N_neurons,
                                    alpha = lambda V: ((10 - V) / 100) / (np.exp(0.1 * (10 - V)) - 1),
                                    beta = lambda V: 0.125 * np.exp(-V / 80),
                                    V_init = V_init)

        self._g_max: float = gK
        self.g: np.ndarray = self._g_max * np.power(self._n_gate.state, 4)

    def update_g(self, V: np.ndarray, t: float, dt: float) -> np.ndarray:
        self._n_gate.update(V, dt)
        self.g = self._g_max * np.power(self._n_gate.state, 4)
        return self.g
    
    def reset(self, V_init: Optional[np.ndarray] = None):
        self._n_gate.set_inf_state(V_init)
        self.g = self._g_max * np.power(self._n_gate.state, 4)

