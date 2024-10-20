from typing import Callable, Optional
import numpy as np


class MarkovIonGate:
    """
    Markov ion gate models the probability of a particular gate within an ion channel being open as a Markov process. At each particular point in time, it computes the probability of a closed gate transitioning to open and open gate transitioning to closed, and uses that to update the overall probability of a particular gate being open at this point in time. Used in a Hodgkin-Huxley model.

    Attributes:
        state: numpy array containing probabilities of the gate being open for each neuron (alternatively, a fraction of open gates of this type in each neuron).
    """
    state: np.ndarray = np.array([])

    def __init__(self, N_neurons: int, alpha: Callable[[np.ndarray], np.ndarray], beta: Callable[[np.ndarray], np.ndarray], V_init: Optional[np.ndarray] = None):
        """
        Initialize a new Markov ion gate.

        :param N_neurons: number of neurons in a simulation.
        :param alpha: a function that maps current membrane potentials of all neurons in mV to the probabilities of the gate transitioning from closed to open in each neuron.
        :param beta: a function that maps current mambrane potantial of all neurons in mV to the probabilities of the gate transitioning from open to closed in each neuron. 
        :param V_init: initial membrane potantial (relative to the resting potential) at stability for each neuron (zero by default).
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

        :param V: numpy array containing membrane potentials for each neuron in mV.
        """
        if V is None:
            self.state = np.zeros(self.N_neurons) + self.rest_val
        else:
            self.state = np.divide(self._alpha(V),  self._alpha(V) + self._beta(V))

    def update(self, V: np.ndarray, dt: float):
        """
        Update the state given current membrane potential.

        :param V: numpy array containing membrane potentials for each neuron in mV.
        :param dt: time interval between two consecutive updates in ms.
        """
        self.state += (self._alpha(V) * (1 - self.state) - self._beta(V) * self.state) * dt

