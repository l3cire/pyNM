from typing import Callable, Optional
import numpy as np


class MarkovIonGate:
    """
    Markov ion gate models the probability of a particular gate within an ion channel being open as a Markov process. At each particular point in time, it computes the probability of a closed gate transitioning to open and open gate transitioning to closed, and uses that to update the overall probability of a particular gate being open at this point in time. Used in a Hodgkin-Huxley model.

    Attributes:
        state: probability of the gate being open (alternatively, a fraction of open gates of this type).
    """
    state: np.ndarray = np.array([])

    def __init__(self, N_neurons: int, alpha: Callable[[np.ndarray], np.ndarray], beta: Callable[[np.ndarray], np.ndarray], V_init: Optional[np.ndarray] = None):
        """
        Initialize a new Markov ion gate.

        :param alpha: a function that maps current membrane potential in mV to the probability of the gate transitioning from closed to open.
        :param beta: a function that maps current mambrane potantial in mV to the probability of the gate transitioning from open to closed. 
        :param v_init: initial membrane potantial at stability.
        """
        self.N_neurons = N_neurons
        self._alpha = alpha
        self._beta = beta
        
        zero = np.array([0])
        self.rest_val: float = alpha(zero)[0]/(beta(zero)[0] + alpha(zero)[0]) 

        self.set_inf_state(V_init)

    def set_inf_state(self, V: Optional[np.ndarray] = None):
        """
        Set the state to the stable value at a given membrane potential.

        :param v: membrane potential in mV.
        """
        if V == None:
            self.state = np.zeros(self.N_neurons) + self.rest_val
        else:
            self.state = np.divide(self._alpha(V),  self._alpha(V) + self._beta(V))

    def update(self, V: np.ndarray, dt: float):
        """
        Update the state given current membrane potential.

        :param v: membrane potential in mV.
        :param dt: time interval between two consecutive updates in mV.
        """
        self.state += (self._alpha(V) * (1 - self.state) - self._beta(V) * self.state) * dt
