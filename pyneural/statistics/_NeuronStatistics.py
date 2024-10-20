import numpy as np
from ._NeuronStepStatistics import NeuronStepStatistics

class NeuronStatistics:
    """
    This class contains the information about the whole simulation for a single neuron.
    """

    def __init__(self, N_steps: int, dt: float):
        """
        Initialize a new statistics object

        :param N: number of simulation steps.
        :param dt: time interval between consecutive steps. 
        """
        self.N_steps = N_steps
        """The number of simulation steps."""
        self.dt = dt
        """The time interval between two consecutive simulation steps in ms."""
        self.step_data: list[NeuronStepStatistics] = []
        """The list of `pyneural.statistics.NeuronStepStatistics` objects containing the statistics per each step."""
        self.spikes: list[np.ndarray] = []
        """The list containding the indices of steps where spikes occured."""
        self.spike_intervals: list[np.ndarray] = []
        """The list containting the interspike intervals in ms."""
        self.mean_interspike_int: list[float] = []
        """The mean interspike interval (0 if none)."""

