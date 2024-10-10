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

    def __init__(self, model='hh', V_start=-70, V_rest=-70, C_m=1, E_L=-59.4, E_K=-82, E_Na=45, gL=0.3, gK=36.0, gNa=120.0):
        self.V_rest = V_rest
        self.V = V_start

        self.C_m = C_m

        self.E_L = E_L
        self.E_K = E_K
        self.E_Na = E_Na

        self.model = model
        if model == 'hh':
            self.g_L = IonChannelConst(gL)
            self.g_K = HHIonChannelK(gK, V_start - V_rest)
            self.g_Na = HHIonChannelNa(gNa, V_start - V_rest)
        elif model == 'const_g':
            self.g_L = IonChannelConst(gL)
            self.g_K = IonChannelConst(gK)
            self.g_Na = IonChannelConst(gNa)
            self.V_rest = (gL * E_L + gK * E_K + gNa * E_Na) / (gL + gK + gNa)
            self.V = self.V_rest
        elif model == 'l-if':
            pass
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

        return stats

    def simulate(self, N, dt, I_input=np.array([])):
        if len(I_input) == 0:
            I_input = np.zeros(N)

        stats = NeuronStatistics(N, dt)
        for i in range(N):
            t = i * dt
            self.I_ext = I_input[i]

            stats.data[i] = self.step(t, dt)
        return stats
