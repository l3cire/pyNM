import numpy as np
from scipy.signal import find_peaks
from .neuron_models import NeuronGroup, ConstCondNeuronGroup, LIFNeuronGroup, HHNeuronGroup
from .input_current import InputCurrent, ConstInputCurrent, NoisyConstInputCurrent, CONST_ZERO_INPUT
from .statistics import NeuronStatistics


class NeuralModel:
    """
    This is the class for modeling the activity of neurons.
    """

    _MODEL_TYPE_TO_CLASS_MAP: dict[str, type[NeuronGroup]] = {
        'const': ConstCondNeuronGroup,
        'lif': LIFNeuronGroup,
        'hh': HHNeuronGroup
    }

    def __init__(self, model: str):
        if model not in NeuralModel._MODEL_TYPE_TO_CLASS_MAP:
            raise ValueError(f'Bad model type: {model}')
        self.model_class: type[NeuronGroup] = NeuralModel._MODEL_TYPE_TO_CLASS_MAP[model]

    def create_model(self, N_neurons: int, params: dict = {}) -> NeuronGroup:
        return self.model_class(N_neurons, params)
        
    def simulate_neurons(self, neurons: NeuronGroup, N_steps: int, dt: float, I_input: InputCurrent = CONST_ZERO_INPUT) -> NeuronStatistics:
        """
        Simulate `N_steps` steps given the external current stimulation for a group of neurons. Returns a `pyneural.statistics.NeuronStatistics` object.

        :param neurons: neurons to simulate.
        :param N_steps: number of steps in a simulation.
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
            spike_ind, _ = find_peaks(potentials[:,i], height=neurons._V_threshold, distance=1/(neurons._max_spike_frequency*dt))
            stats.spikes.append(spike_ind)
            stats.spike_intervals.append(np.diff(spike_ind)*dt)

            mean_interspike_int = np.mean(stats.spike_intervals[-1])
            stats.mean_interspike_int.append(mean_interspike_int)
            stats.spiking_frequency.append(np.floating(0) if mean_interspike_int == 0 else 1/mean_interspike_int)

        return stats
        
    def get_fi_curve(self, I_ext: np.ndarray, std: float = 0, params: dict = {}, N_iter = 100000, dt: float = 1) -> np.ndarray:
        """
        This function computes the f-I (spiking frequency vs. current stimulation) curve for a given neuron.

        :param neuron: a neuron object for which the curve is computed.
        :param I_ext: a list of different current stimulations for which the spiking frequency should be computed.
        :param N_iter: a number of iterations per current in a simulation.
        :param dt: an interval between two consequtive iterations in a simulation.
        """
        
        neurons = self.model_class(N_neurons=I_ext.size, params=params)
        current = NoisyConstInputCurrent(N_neurons=I_ext.size, I=I_ext, std=std)
        stats = self.simulate_neurons(neurons, N_iter, dt, current)

        return np.array(stats.spiking_frequency)

