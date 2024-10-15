from typing import Optional

from ..input_current import InputCurrent, ConstInputCurrent
from ..ion_channels import IonChannel
from ..statistics import NeuronStatistics
from ..statistics import NeuronStepStatistics
import numpy as np


class Neuron:
    """
    A base class for models of a single neuron, which specifies the main parameters of any neuron cell. 

    """
    I_ext: float = 0.0
    """External current stimulation in mA. Typically between 0 and 50. The variation in current is usually caused by other neurons connected to the current one by synnapses."""

    g_L: IonChannel
    """@private Leak ion channel."""
    g_K: IonChannel
    """@private Potassium ion channel."""
    g_Na: IonChannel
    """@private Sodium ion channel."""

    def __init__(self, params: dict = {}):#V_start=-70, V_rest=-70, C_m=1, E_L=-59.4, E_K=-82, E_Na=45, gL=0.3, gK=36.0, gNa=120.0):
        """
        Initialize a new neuron.

        :param params['V_rest']: resting potential in mV (-70.0 by default).
        :param params['V']: starting membrane potential in mV (-70.0 by default).
        :param params['V_threshold']: threshold voltage in mV (-56.0 by default). This is the value of membrane potential that certainly generates a spike. Needed for spike detection.
        :param params['C_m']: membrane capacitance in μF/cm2 (1.0 by default).
        :param params['E_L']: leak ion channels reversal potantial in mV (-59.4 by default).
        :param params['E_K']: potassium ion channels reversal potential in mV (-82.0 by default).
        :param params['E_Na']: sodium ion channels reversal potential in mV (45.0 by default).

        Note that the conductances of ion channels are not specidied in the base class constructor since they differ in different models.
        """
        self.V_rest = params.get('V_rest', -70.0)
        """@private"""
        self.V = params.get('V_start', -70.0)
        """@private"""
        self.V_threshold = params.get('V_threshold', -56.0)
        """@private"""

        self.C_m = params.get('C_m', 1.0)
        """@private"""

        self.E_L = params.get('E_L', -59.4)
        """@private"""
        self.E_K = params.get('E_K', -82)
        """@private"""
        self.E_Na = params.get('E_Na', 45)
        """@private"""

    def reset(self, V: Optional[float] = None):
        """
        Reset the neuron to the stable state. All ion channels are reset with respect to the membrane potential.

        :param V: new membrane potential in mV. If not specified, is set to the resting potential.
        """

        if not V:
            self.V = self.V_rest
        else:
            self.V = V

        self.g_L.reset(self.V - self.V_rest)
        self.g_K.reset(self.V - self.V_rest)
        self.g_Na.reset(self.V - self.V_rest)

    def step(self, t: float, dt: float) -> NeuronStepStatistics:
        """
        Perform one step of a simulation. Returns a `pyneural.statistics.NeuronStepStatistics` object.

        :param t: current time in ms.
        :param dt: time between two consecutive simulation steps in ms.
        """
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

    def simulate(self, N: int, dt: float, I_input: InputCurrent = ConstInputCurrent())-> NeuronStatistics:
        """
        Simulate `N` steps given the external current stimulation. Returns a `pyneural.statistics.NeuronStatistics` object.

        :param N: number of steps in a simulation
        :param dt: time interval between two consecutive steps in ms.
        :param I_input: `pyneural.input_current.InputCurrent` object specifying the current stimulation.
        """

        self.reset()
        stats = NeuronStatistics(N, dt)
        for i in range(N):
            t = i * dt
            self.I_ext = I_input.get_current(t)

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
