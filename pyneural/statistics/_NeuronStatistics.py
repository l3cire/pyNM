import numpy as np
from ._NeuronStepStatistics import NeuronStepStatistics

class NeuronStatistics:
    """
    This class contains the information about the whole simulation for a group of neurons.
    """

    def __init__(self, N_steps: int, dt: float):
        """
        Initialize a new statistics object

        :param N_steps: number of simulation steps.
        :param dt: time interval between consecutive steps. 
        """
        self.N_steps = N_steps
        """The number of simulation steps."""
        self.dt = dt
        """The time interval between two consecutive simulation steps in ms."""
        self.step_data: list[NeuronStepStatistics] = []
        """The list of `pyneural.statistics.NeuronStepStatistics` objects containing the statistics per each step."""
        self.spikes: list[np.ndarray] = []
        """The list containding the numpy arrays of steps where spikes occured for each neuron."""
        self.spike_intervals: list[np.ndarray] = []
        """The list containting the numpy arrays of interspike intervals for each neuron in ms."""
        self.mean_interspike_int: list[float] = []
        """The list containing mean interspike intervals for each neuron (0 if none)."""
        self.spiking_frequency: list[float] = []
        """The list containing the spiking frequencies foe each neuron (0 of no spikes)."""

