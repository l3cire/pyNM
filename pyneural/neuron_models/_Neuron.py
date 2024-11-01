from typing import Optional
from abc import ABC, abstractmethod
import numpy as np
from ..statistics import NeuronStepStatistics


class NeuronGroup(ABC):
    """
    A base class for models of a group of independent neurons, which specifies the main parameters of any neuron cell. 

    """
   
    def __init__(self, N_neurons: int = 1, params: dict = {}):#V_start=-70, V_rest=-70, C_m=1, E_L=-59.4, E_K=-82, E_Na=45, gL=0.3, gK=36.0, gNa=120.0):
        """
        Initialize a new neuron.

        :param N_neurons: number of neurons in a group.
        :param params['V_rest']: resting potential in mV (-70.0 by default).
        :param params['V']: starting membrane potential for each cell in mV (by default, all initialized to be equal to the resting potential).
        :param params['V_threshold']: threshold voltage in mV (0.0 by default). This is the value of membrane potential that certainly generates a spike. Needed for spike detection.
        Note that the conductances of ion channels are not specidied in the base class constructor since they differ in different models.
        """
        self.N_neurons: int = N_neurons
        self._V_rest: float = params.get('V_rest', -70.0)
        self._V: np.ndarray = params.get('V_start', np.zeros(self.N_neurons) + self._V_rest)
        self._V_threshold = params.get('V_threshold', 0.0)
        self._max_spike_frequency = params.get("max_spike_f", 0.5)

       
    def reset(self, V: Optional[np.ndarray] = None):
        """
        Reset the neuron to the stable state. All ion channels are reset with respect to the membrane potential.

        :param V: numpy array containing new membrane potentials for each neuron in mV. If not specified, neurons are set to the resting potential.
        """

        if not V:
            self._V = np.zeros(self.N_neurons) + self._V_rest 
        else:
            self._V = V.copy()

    @abstractmethod
    def step(self, I_ext: np.ndarray, t: float, dt: float) -> NeuronStepStatistics:
        """
        Perform one step of a simulation. Returns a `pyneural.statistics.NeuronStepStatistics` object.

        :param I_ext: external stimulation for each neuron cell.
        :param t: current time in ms.
        :param dt: time between two consecutive simulation steps in ms.
        """
        pass

