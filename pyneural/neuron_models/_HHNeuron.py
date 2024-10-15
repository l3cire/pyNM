from typing import Optional
import numpy as np
from ..statistics import NeuronStepStatistics, NeuronStatistics
from ..ion_channels import HHIonChannelNa, HHIonChannelK, IonChannelConst
from ._Neuron import Neuron

class HHNeuron(Neuron):
    """
    Implementation of the Hodgkin-Huxley model of a neuron.
    """

    I_ext: float = 0.0

    def __init__(self, params: dict = {}):
        """
        Initialize a new neuron.

        Apart from parameters specified in the base class `pyneural.neuron_models.Neuron`, this class requires additional parameters to specify conductances:
        :param params['gL']: leak ion channels conductance (0.3 by default).
        :param params['gK']: potassium ion channels conductance when all potassium channels are open (36.0 by default).
        :param params['gNa']: sodium ion channels conductance when all sodium channels are open (120.0 by default).
 
        """
        super().__init__(params)
        self.g_L = IonChannelConst(params.get('gL', 0.3))
        self.g_K = HHIonChannelK(params.get('gK', 36.0), self.V - self.V_rest)
        self.g_Na = HHIonChannelNa(params.get('gNa', 120.0), self.V - self.V_rest)

    def step(self, t: float, dt: float) -> NeuronStepStatistics:
        assert isinstance(self.g_K, HHIonChannelK) and isinstance(self.g_Na, HHIonChannelNa)
        stats = super().step(t, dt)
        stats.gate_n = self.g_K.n_gate.state
        stats.gate_m = self.g_Na.m_gate.state
        stats.gate_h = self.g_Na.h_gate.state
        return stats

    def reset(self, V: Optional[float] = None):
        return super().reset(V)

    def simulate(self, N: int, dt: float, I_input=np.array([])) -> NeuronStatistics:
        return super().simulate(N, dt, I_input)

