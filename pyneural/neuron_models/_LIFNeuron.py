from typing import Optional
import numpy as np

from ..input_current import CONST_ZERO_INPUT, InputCurrent
from ..statistics import NeuronStatistics, NeuronStepStatistics
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
        """
        super().__init__()
        self.V_reset = params.get('V_reset', -80.099)
        self.V_spike = params.get('V_spike', 35.685)

    def step(self, t: float, dt: float) -> NeuronStepStatistics:
        stats = super().step(t, dt)
        if(self.V > self.V_threshold):
            self.V = self.V_reset
            stats.Vm = self.V_spike
        return stats

    def reset(self, V: Optional[float] = None):
        return super().reset(V)

    def simulate(self, N: int, dt: float, I_input: InputCurrent = CONST_ZERO_INPUT) -> NeuronStatistics:
        return super().simulate(N, dt, I_input)


