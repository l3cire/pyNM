import numpy as np
from .input_current import InputCurrent, NoisyConstInputCurrent, CONST_ZERO_INPUT
from .neuron_models import Neuron
from .statistics import NeuronStatistics


class NeuralModel:
    """
    This is the class for modeling the activity of neurons.
    """
        
    def simulate_neuron(self, neuron: Neuron, N: int, dt: float, I_input: InputCurrent = CONST_ZERO_INPUT) -> NeuronStatistics:
        """
        Simulate `N` steps given the external current stimulation for a single neuron. Returns a `pyneural.statistics.NeuronStatistics` object.

        :param neuron: neuron to simulate.
        :param N: number of steps in a simulation.
        :param dt: time interval between two consecutive steps in ms.
        :param I_input: `pyneural.input_current.InputCurrent` object specifying the current stimulation.
        """

        neuron.reset()
        stats = NeuronStatistics(N, dt)
        for i in range(N):
            t = i * dt
            neuron.I_ext = I_input.get_current(t)
            stats.step_data.append(neuron.step(t, dt))

        for i in range(1, N-1):
            if(stats.step_data[i].Vm > neuron._V_threshold and stats.step_data[i].Vm > stats.step_data[i-1].Vm 
                   and stats.step_data[i].Vm > stats.step_data[i+1].Vm):
                stats.step_data[i].spiked = True
                stats.spikes.append(i)

        for i in range(1, len(stats.spikes)):
            stats.spike_intervals.append((stats.spikes[i] - stats.spikes[i-1])*dt)
        if(len(stats.spike_intervals) > 0):
            stats.mean_interspike_int = np.mean(stats.spike_intervals)
        else:
            stats.mean_interspike_int = 0
        return stats
        
    def get_fi_curve(self, neuron: Neuron, I_ext, N_iter = 100000, dt = 1):
        """
        This function computes the f-I (spiking frequency vs. current stimulation) curve for a given neuron.

        :param neuron: a neuron object for which the curve is computed.
        :param I_ext: a list of different current stimulations for which the spiking frequency should be computed.
        :param N_iter: a number of iterations per current in a simulation.
        :param dt: an interval between two consequtive iterations in a simulation.
        """

        firing_rates = []
        for I in I_ext:
            stats = self.simulate_neuron(neuron, N_iter, dt, NoisyConstInputCurrent(I = I, std=15))
            firing_rates.append(1/stats.mean_interspike_int if stats.mean_interspike_int != 0 else 0)
        return firing_rates

