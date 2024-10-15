from .input_current import ConstInputCurrent
from .neuron_models import Neuron


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
            stats = neuron.simulate(N_iter, dt, ConstInputCurrent(I = I))
            firing_rates.append(1/stats.mean_interspike_int)
        return firing_rates

