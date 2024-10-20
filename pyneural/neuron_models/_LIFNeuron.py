from typing import Optional
import numpy as np
from ..statistics import NeuronStepStatistics
from ._ConstCondNeuron import ConstCondNeuronGroup

class LIFNeuronGroup(ConstCondNeuronGroup):
    """
    Implementation of the Leaky Integrate and Fire model of a neuron.

    """

    def __init__(self, N_neurons: int, params: dict = {}):
        """
        Initialize a new group of Leaky Integrate and Fire neuron.

        This model inherits from `pyneural.neuron_models.ConstCondNeuron`, since it also models conductances as constant. However, apart from parameters required for the base class, it need two additional parameters:
        :param params['V_reset']: the potential to reset to after a spike in mV (-80.099 by default).
        :param params['V_spike']: spike potential in mV (35.685 by default).
        :param params['V_threshold']: threshold potential in mV (-50.0 by default).
        """
        super().__init__(N_neurons, params)
        self._V_reset = params.get('V_reset', -75.0)
        self._V_spike = params.get('V_spike', 35.0)
        self._V_threshold = params.get('V_threshold', -50.0)


    def step(self, I_ext: np.ndarray, t: float, dt: float) -> NeuronStepStatistics:
        stats = super().step(I_ext, t, dt)
        stats.Vm[self._V > self._V_threshold] = self._V_spike
        self._V[self._V > self._V_threshold] = self._V_reset
        return stats

    def reset(self, V: Optional[np.ndarray] = None):
        return super().reset(V)

