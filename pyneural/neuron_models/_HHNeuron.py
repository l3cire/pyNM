from typing import Optional
from ..input_current import CONST_ZERO_INPUT, InputCurrent
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
        :param params['C_m']: membrane capacitance in μF/cm2 (1.0 by default).
        :param params['E_L']: leak ion channels reversal potantial in mV (-59.4 by default).
        :param params['E_K']: potassium ion channels reversal potential in mV (-82.0 by default).
        :param params['E_Na']: sodium ion channels reversal potential in mV (45.0 by default).
        """

        super().__init__(params)
        
        self._E_L = params.get('E_L', -59.4)
        self._E_K = params.get('E_K', -82)
        self._E_Na = params.get('E_Na', 45)

        self._C_m = params.get('C_m', 1.0)

        self._g_L = IonChannelConst(params.get('gL', 0.3))
        self._g_K = HHIonChannelK(params.get('gK', 36.0), self._V - self._V_rest)
        self._g_Na = HHIonChannelNa(params.get('gNa', 120.0), self._V - self._V_rest)

    def step(self, t: float, dt: float) -> NeuronStepStatistics:
        stats = NeuronStepStatistics()
        stats.T = t

        stats.g_leak = self._g_L.update_g(self._V - self._V_rest, t, dt)
        stats.g_K = self._g_K.update_g(self._V - self._V_rest, t, dt)
        stats.g_Na = self._g_Na.update_g(self._V - self._V_rest, t, dt)
        stats.g_m = self._g_L.g + self._g_K.g + self._g_Na.g

        I_leak = -self._g_L.g * (self._V - self._E_L)
        I_K = -self._g_K.g * (self._V - self._E_K)
        I_Na = -self._g_Na.g * (self._V - self._E_Na)
        stats.I_leak, stats.I_K, stats.I_Na, stats.I_ext, stats.I_total = I_leak, I_K, I_Na, self.I_ext, (I_leak + I_K + I_Na + self.I_ext)

        self._V += stats.I_total * dt / self._C_m  # since dV/dt = CI
        stats.Vm = self._V

        stats.gate_n = self._g_K.n_gate.state
        stats.gate_m = self._g_Na.m_gate.state
        stats.gate_h = self._g_Na.h_gate.state
        return stats

    def reset(self, V: Optional[float] = None):
        super().reset(V)
        self._g_L.reset(self._V - self._V_rest)
        self._g_K.reset(self._V - self._V_rest)
        self._g_Na.reset(self._V - self._V_rest)

