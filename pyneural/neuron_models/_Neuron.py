from typing import Optional
from abc import ABC, abstractmethod
from ..statistics import NeuronStepStatistics


class Neuron(ABC):
    """
    A base class for models of a single neuron, which specifies the main parameters of any neuron cell. 

    """
    I_ext: float = 0.0
    """External current stimulation in mA. Typically between 0 and 50. The variation in current is usually caused by other neurons connected to the current one by synnapses."""

   
    def __init__(self, params: dict = {}):#V_start=-70, V_rest=-70, C_m=1, E_L=-59.4, E_K=-82, E_Na=45, gL=0.3, gK=36.0, gNa=120.0):
        """
        Initialize a new neuron.

        :param params['V_rest']: resting potential in mV (-70.0 by default).
        :param params['V']: starting membrane potential in mV (-70.0 by default).
        :param params['V_threshold']: threshold voltage in mV (0.0 by default). This is the value of membrane potential that certainly generates a spike. Needed for spike detection.
        Note that the conductances of ion channels are not specidied in the base class constructor since they differ in different models.
        """
        self._V_rest = params.get('V_rest', -70.0)
        self._V = params.get('V_start', -70.0)
        self._V_threshold = params.get('V_threshold', 0.0)

       
    def reset(self, V: Optional[float] = None):
        """
        Reset the neuron to the stable state. All ion channels are reset with respect to the membrane potential.

        :param V: new membrane potential in mV. If not specified, is set to the resting potential.
        """

        if not V:
            self._V = self._V_rest
        else:
            self._V = V

    @abstractmethod
    def step(self, t: float, dt: float) -> NeuronStepStatistics:
        """
        Perform one step of a simulation. Returns a `pyneural.statistics.NeuronStepStatistics` object.

        :param t: current time in ms.
        :param dt: time between two consecutive simulation steps in ms.
        """
        pass

