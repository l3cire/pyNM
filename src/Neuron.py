from src.ion_channels.HHIonChannelK import HHIonChannelK
from src.ion_channels.IonChannel import IonChannel
from src.ion_channels.IonChannelConst import IonChannelConst
from src.ion_channels.HHIonChannelNa import HHIonChannelNa
from src.statistics.NeuronStatistics import NeuronStatistics
from src.statistics.NeuronStepStatistics import NeuronStepStatistics
import numpy as np


class Neuron:
    I_ext = 0.0

    g_L: IonChannel
    g_K: IonChannel
    g_Na: IonChannel

    def __init__(self, model='hh', params: dict = {}):#V_start=-70, V_rest=-70, C_m=1, E_L=-59.4, E_K=-82, E_Na=45, gL=0.3, gK=36.0, gNa=120.0):
        self.V_rest = params.get('V_rest', -70.0)
        self.V = params.get('V_start', -70.0)

        self.C_m = params.get('C_m', 1.0)

        self.E_L = params.get('E_L', -59.4)
        self.E_K = params.get('E_K', -82)
        self.E_Na = params.get('E_Na', 45)

        self.model = model
        if model == 'hh':
            self.g_L = IonChannelConst(params.get('gL', 0.3))
            self.g_K = HHIonChannelK(params.get('gK', 36.0), self.V - self.V_rest)
            self.g_Na = HHIonChannelNa(params.get('gNa', 120.0), self.V - self.V_rest)
        elif model == 'const_g' or model == 'lif':
            self.g_L = IonChannelConst(params.get('gL', 0.3))
            self.g_K = IonChannelConst(params.get('gK', 0.366))
            self.g_Na = IonChannelConst(params.get('gNa', 0.0106))
            self.C_m = params.get('C_m', 2.0)
            self.V_rest = (self.g_L.g * self.E_L + self.g_K.g * self.E_K + self.g_Na.g * self.E_Na) / (self.g_L.g + self.g_K.g + self.g_Na.g)
            if model == 'lif':
                self.V_threshold = params.get('V_threshold', -56.0)
                self.V_reset = params.get('V_reset', -80.099)
                self.V_spike = params.get('V_spike', 35.685)
        else:
            pass

    def step(self, t, dt):
        stats = NeuronStepStatistics()
        stats.T = t

        stats.g_leak = self.g_L.update_g(self.V - self.V_rest, t, dt)
        stats.g_K = self.g_K.update_g(self.V - self.V_rest, t, dt)
        stats.g_Na = self.g_Na.update_g(self.V - self.V_rest, t, dt)

        if isinstance(self.g_K, HHIonChannelK):
            stats.gate_n = self.g_K.n_gate.state
        if isinstance(self.g_Na, HHIonChannelNa):
            stats.gate_m = self.g_Na.m_gate.state
            stats.gate_h = self.g_Na.h_gate.state

        stats.I_leak = -self.g_L.g * (self.V - self.E_L)
        stats.I_K = -self.g_K.g * (self.V - self.E_K)
        stats.I_Na = -self.g_Na.g * (self.V - self.E_Na)
        stats.I_ext = self.I_ext
        stats.I_total = stats.I_leak + stats.I_K + stats.I_Na + stats.I_ext

        self.V += stats.I_total * dt / self.C_m  # since dV/dt = CI
        stats.Vm = self.V

        if self.model == 'lif':
            if(self.V > self.V_threshold):
                self.V = self.V_reset
                stats.Vm = self.V_spike

        return stats

    def simulate(self, N, dt, I_input=np.array([])):
        if len(I_input) == 0:
            I_input = np.zeros(N)

        stats = NeuronStatistics(N, dt)
        for i in range(N):
            t = i * dt
            self.I_ext = I_input[i]

            stats.data.append(self.step(t, dt))
        return stats
