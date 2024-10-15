from ..ion_channels import HHIonChannelNa
from ..ion_channels import IonChannelConst
from ._Neuron import Neuron
from ..ion_channels import HHIonChannelK

class HHNeuron(Neuron):

    def __init__(self, params: dict = {}):
        super().__init__(params)
        self.g_L = IonChannelConst(params.get('gL', 0.3))
        self.g_K = HHIonChannelK(params.get('gK', 36.0), self.V - self.V_rest)
        self.g_Na = HHIonChannelNa(params.get('gNa', 120.0), self.V - self.V_rest)

    def step(self, t, dt):
        assert isinstance(self.g_K, HHIonChannelK) and isinstance(self.g_Na, HHIonChannelNa)
        stats = super().step(t, dt)
        stats.gate_n = self.g_K.n_gate.state
        stats.gate_m = self.g_Na.m_gate.state
        stats.gate_h = self.g_Na.h_gate.state
        return stats
