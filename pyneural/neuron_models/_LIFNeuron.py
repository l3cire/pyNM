from typing import Optional
from ..statistics import NeuronStepStatistics
from ._ConstCondNeuron import ConstCondNeuron

class LIFNeuron(ConstCondNeuron):
    """
    Implementation of the Leaky Integrate and Fire model of a neuron.

    """

    I_ext: float = 0.0

    
    def __init__(self, params: dict = {}):
        """
        Initialize a new neuron.

        This model inherits from `pyneural.neuron_models.ConstCondNeuron`, since it also models conductances as constant. However, apart from parameters required for the base class, it need two additional parameters:
        :param params['V_reset']: the potential to reset to after a spike in mV (-80.099 by default).
        :param params['V_spike']: spike potential in mV (35.685 by default).
        :param params['V_threshold']: threshold potential in mV (-50.0 by default).
        """
        super().__init__()
        self._V_reset = params.get('V_reset', -75.0)
        self._V_spike = params.get('V_spike', 35.0)
        self._V_threshold = params.get('V_threshold', -50.0)


    def step(self, t: float, dt: float) -> NeuronStepStatistics:
        stats = super().step(t, dt)
        if(self._V > self._V_threshold):
            self._V = self._V_reset
            stats.Vm = self._V_spike
        return stats

    def reset(self, V: Optional[float] = None):
        return super().reset(V)

