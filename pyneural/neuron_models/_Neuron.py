from typing import Optional
from ..ion_channels import IonChannel
from ..statistics import NeuronStatistics
from ..statistics import NeuronStepStatistics
import numpy as np


class Neuron:
    I_ext = 0.0

    g_L: IonChannel
    g_K: IonChannel
    g_Na: IonChannel

    def __init__(self, params: dict = {}):#V_start=-70, V_rest=-70, C_m=1, E_L=-59.4, E_K=-82, E_Na=45, gL=0.3, gK=36.0, gNa=120.0):
        self.V_rest = params.get('V_rest', -70.0)
        self.V = params.get('V_start', -70.0)
        self.V_threshold = params.get('V_threshold', -56.0)


        self.C_m = params.get('C_m', 1.0)

        self.E_L = params.get('E_L', -59.4)
        self.E_K = params.get('E_K', -82)
        self.E_Na = params.get('E_Na', 45)

    def reset(self, V: Optional[float] = None):
        if not V:
            self.V = self.V_rest
        else:
            self.V = V

        self.g_L.reset(self.V - self.V_rest)
        self.g_K.reset(self.V - self.V_rest)
        self.g_Na.reset(self.V - self.V_rest)

    def step(self, t, dt):
        stats = NeuronStepStatistics()
        stats.T = t

        stats.g_leak = self.g_L.update_g(self.V - self.V_rest, t, dt)
        stats.g_K = self.g_K.update_g(self.V - self.V_rest, t, dt)
        stats.g_Na = self.g_Na.update_g(self.V - self.V_rest, t, dt)

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

        self.reset()
        stats = NeuronStatistics(N, dt)
        for i in range(N):
            t = i * dt
            self.I_ext = I_input[i]

            stats.step_data.append(self.step(t, dt))

        for i in range(1, N-1):
            if(stats.step_data[i].Vm > self.V_threshold and stats.step_data[i].Vm > stats.step_data[i-1].Vm 
                   and stats.step_data[i].Vm > stats.step_data[i+1].Vm):
                stats.step_data[i].spiked = True
                stats.spikes.append(i)

        for i in range(1, len(stats.spikes)):
            stats.spike_intervals.append((stats.spikes[i] - stats.spikes[i-1])*dt)
        stats.mean_interspike_int = np.mean(stats.spike_intervals)

        return stats
