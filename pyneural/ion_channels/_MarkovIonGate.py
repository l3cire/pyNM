from typing import Callable


class MarkovIonGate:
    """
    Markov ion gate models the probability of a particular gate within an ion channel being open as a Markov process. At each particular point in time, it computes the probability of a closed gate transitioning to open and open gate transitioning to closed, and uses that to update the overall probability of a particular gate being open at this point in time. Used in a Hodgkin-Huxley model.

    Attributes:
        state: probability of the gate being open (alternatively, a fraction of open gates of this type).
    """
    state: float = 0.0

    def __init__(self, alpha: Callable[[float], float], beta: Callable[[float], float], v_init: float = 0):
        """
        Initialize a new Markov ion gate.

        :param alpha: a function that maps current membrane potential in mV to the probability of the gate transitioning from closed to open.
        :param beta: a function that maps current mambrane potantial in mV to the probability of the gate transitioning from open to closed. 
        :param v_init: initial membrane potantial at stability.
        """
        self._alpha = alpha
        self._beta = beta
        self.set_inf_state(v_init)

    def set_inf_state(self, v: float):
        """
        Set the state to the stable value at a given membrane potential.

        :param v: membrane potential in mV.
        """
        self.state = self._alpha(v) / (self._alpha(v) + self._beta(v))

    def update(self, v: float, dt: float):
        """
        Update the state given current membrane potential.

        :param v: membrane potential in mV.
        :param dt: time interval between two consecutive updates in mV.
        """
        self.state += (self._alpha(v) * (1 - self.state) - self._beta(v) * self.state) * dt
