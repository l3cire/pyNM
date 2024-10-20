import numpy as np
from scipy.signal import find_peaks
from .input_current import InputCurrent, NoisyConstInputCurrent, CONST_ZERO_INPUT
from .neuron_models import NeuronGroup
from .statistics import NeuronStatistics


class NeuralModel:
    """
    This is the class for modeling the activity of neurons.
    """
        
    def simulate_neurons(self, neurons: NeuronGroup, N_steps: int, dt: float, I_input: InputCurrent = CONST_ZERO_INPUT) -> NeuronStatistics:
        """
        Simulate `N` steps given the external current stimulation for a single neuron. Returns a `pyneural.statistics.NeuronStatistics` object.

        :param neurons: neurons to simulate.
        :param N: number of steps in a simulation.
        :param dt: time interval between two consecutive steps in ms.
        :param I_input: `pyneural.input_current.InputCurrent` object specifying the current stimulation.
        """

        neurons.reset()
        stats = NeuronStatistics(N_steps, dt)
        potentials = np.zeros((N_steps, neurons.N_neurons))
        for i in range(N_steps):
            t = i * dt
            step = neurons.step(I_input.get_current(t), t, dt)
            potentials[i] = step.Vm
            stats.step_data.append(step)

        for i in range(neurons.N_neurons):
            spike_ind, _ = find_peaks(potentials[:,i], height=neurons._V_threshold)
            stats.spikes.append(spike_ind)
            stats.spike_intervals.append(np.diff(spike_ind)*dt)
            stats.mean_interspike_int.append(np.mean(stats.spike_intervals[-1]))

        return stats
        
    def get_fi_curves(self, neuron_params: dict, I_ext: np.ndarray, N_iter = 100000, dt = 1):
        """
        This function computes the f-I (spiking frequency vs. current stimulation) curve for a given neuron.

        :param neuron: a neuron object for which the curve is computed.
        :param I_ext: a list of different current stimulations for which the spiking frequency should be computed.
        :param N_iter: a number of iterations per current in a simulation.
        :param dt: an interval between two consequtive iterations in a simulation.
        """

        firing_rates = []
        #for I in I_ext:
       #     stats = self.simulate_neurons(neurons, N_iter, dt, NoisyConstInputCurrent(I = I, std=15))
       #     firing_rates.append(1/stats.mean_interspike_int if stats.mean_interspike_int != 0 else 0)
        return firing_rates

