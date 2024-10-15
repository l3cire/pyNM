from ._NeuronStepStatistics import NeuronStepStatistics

class NeuronStatistics:
    def __init__(self, N, dt):
        self.N = N
        self.dt = dt
        self.step_data: list[NeuronStepStatistics] = [] 
        self.spikes: list[int] = []
        self.spike_intervals: list[float] = []
        self.mean_interspike_int: float = 0

