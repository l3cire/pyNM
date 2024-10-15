from typing import Optional
import numpy as np
from ..input_current import InputCurrent, ConstInputCurrent
from ..statistics import NeuronStatistics, NeuronStepStatistics
from ._Neuron import Neuron
from ..ion_channels import IonChannelConst

class ConstCondNeuron(Neuron):
    """
    A neuron with constant conductance for each ion channel type.
    """

    I_ext: float = 0.0

    def __init__(self, params: dict = {}):
        """
        Initialize a new neuron.

        Apart from parameters specified in the base class `pyneural.neuron_models.Neuron`, this class requires additional parameters to specify conductances:
        :param params['gL']: leak ion channels conductance (0.3 by default).
        :param params['gK']: potassium ion channels conductance (0.366 by default).
        :param params['gNa']: sodium ion channels conductance (0.0106 by default).
        """
        super().__init__(params)
        self.g_L = IonChannelConst(params.get('gL', 0.3))
        self.g_K = IonChannelConst(params.get('gK', 0.366))
        self.g_Na = IonChannelConst(params.get('gNa', 0.0106))
        self.V_rest = (self.g_L.g * self.E_L + self.g_K.g * self.E_K + self.g_Na.g * self.E_Na) / (self.g_L.g + self.g_K.g + self.g_Na.g)

    def step(self, t: float, dt: float) -> NeuronStepStatistics:
        return super().step(t, dt)

    def reset(self, V: Optional[float] = None):
        return super().reset(V)

    def simulate(self, N: int, dt: float, I_input:InputCurrent = ConstInputCurrent()) -> NeuronStatistics:
        return super().simulate(N, dt, I_input)

