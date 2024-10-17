import numpy as np
from .input_current import InputCurrent, ConstInputCurrent, CONST_ZERO_INPUT
from .neuron_models import Neuron
from .statistics import NeuronStatistics


class NeuralModel:
    """
    This is the class for modeling the activity of neurons.
    """

    def get_fi_curve(self, neuron: Neuron, I_ext, N_iter = 10000, dt = 0.01):
        """
        Compute the f-I curve of a given neuron.

        This functions computes the f-I (spiking frequency vs. current stimulation) curve for a given neuron.

        Parameters
        ----------
        neuron
            a neuron object for which the curve is computed.
        I_ext : list[float]
            a list of different current stimulations for which the spiking frequency should be computed.
        N_iter : int
            a number of iterations per current in a simulation.
        dt : float
            an interval between two consequtive iterations in a simulation.
        """
        firing_rates = []
        for I in I_ext:
            stats = self.simulate_neuron(neuron, N_iter, dt, ConstInputCurrent(I = I))
            firing_rates.append(1/stats.mean_interspike_int)
        return firing_rates
    
    def simulate_neuron(self, neuron: Neuron, N: int, dt: float, I_input: InputCurrent = CONST_ZERO_INPUT) -> NeuronStatistics:
        """
        Simulate `N` steps given the external current stimulation for a single neuron. Returns a `pyneural.statistics.NeuronStatistics` object.

        :param neuron: neuron to simpulate.
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
        stats.mean_interspike_int = np.mean(stats.spike_intervals)

        return stats
