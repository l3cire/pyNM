from ._NeuronStepStatistics import NeuronStepStatistics

class NeuronStatistics:
    """
    This class contains the information about the whole simulation for a single neuron.
    """

    N: int
    """The number of simulation steps."""
    dt: float
    """The time interval between two consecutive simulation steps in ms."""
    step_data: list[NeuronStepStatistics] = []
    """The list of `pyneural.statistics.NeuronStepStatistics` objects containing the statistics per each step."""
    spikes: list[int] = []
    """The list containding the indices of steps where spikes occured."""
    spike_intervals: list[float] = []
    """The list containting the interspike intervals in ms."""
    mean_interspike_int: float = 0
    """The mean interspike interval (0 if none)."""
    
    def __init__(self, N: int, dt: float):
        """
        Initialize a new statistics object

        :param N: number of simulation steps.
        :param dt: time interval between consecutive steps. 
        """
        self.N = N
        self.dt = dt

