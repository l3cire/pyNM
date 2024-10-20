from typing import Optional
import numpy as np
from ..statistics import NeuronStepStatistics
from ..ion_channels import HHIonChannelNa, HHIonChannelK, IonChannelConst
from ._Neuron import NeuronGroup

class HHNeuronGroup(NeuronGroup):
    """
    Implementation of the Hodgkin-Huxley model of a neuron.
    """

    def __init__(self, N_neurons: int, params: dict = {}):
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

        super().__init__(N_neurons, params)
        
        self._E_L = params.get('E_L', -59.4)
        self._E_K = params.get('E_K', -82)
        self._E_Na = params.get('E_Na', 45)

        self._C_m = params.get('C_m', 1.0)

        self._g_L = IonChannelConst(self.N_neurons, params.get('gL', 0.3))
        self._g_K = HHIonChannelK(self.N_neurons, params.get('gK', 36.0), self._V - self._V_rest)
        self._g_Na = HHIonChannelNa(self.N_neurons, params.get('gNa', 120.0), self._V - self._V_rest)

    def step(self, I_ext: np.ndarray, t: float, dt: float) -> NeuronStepStatistics:
        stats = NeuronStepStatistics()
        stats.T = t

        stats.g_leak = self._g_L.update_g(self._V - self._V_rest, t, dt).copy()
        stats.g_K = self._g_K.update_g(self._V - self._V_rest, t, dt).copy()
        stats.g_Na = self._g_Na.update_g(self._V - self._V_rest, t, dt).copy()

        I_leak: np.ndarray = -self._g_L.g * (self._V - self._E_L)
        I_K: np.ndarray = -self._g_K.g * (self._V - self._E_K)
        I_Na: np.ndarray = -self._g_Na.g * (self._V - self._E_Na)
        stats.I_leak, stats.I_K, stats.I_Na, stats.I_ext, stats.I_total = I_leak, I_K, I_Na, I_ext, (I_leak + I_K + I_Na + I_ext)

        self._V += stats.I_total * dt / self._C_m  # since dV/dt = CI
        stats.Vm = self._V.copy()

        stats.gate_n = self._g_K._n_gate.state.copy()
        stats.gate_m = self._g_Na._m_gate.state.copy()
        stats.gate_h = self._g_Na._h_gate.state.copy()
        return stats

    def reset(self, V: Optional[np.ndarray] = None):
        super().reset(V)
        self._g_L.reset(self._V - self._V_rest)
        self._g_K.reset(self._V - self._V_rest)
        self._g_Na.reset(self._V - self._V_rest)

