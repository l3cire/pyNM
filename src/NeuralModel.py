from src.Neuron import Neuron
import numpy as np


class NeuralModel:

    def get_fi_curve(self, neuron: Neuron, I_ext, N_iter = 10000, dt = 0.01):
        firing_rates = []
        for I in I_ext:
            stats = neuron.simulate(N_iter, dt, (np.zeros(N_iter) + I))
            firing_rates.append(1/stats.mean_interspike_int)
        return firing_rates

