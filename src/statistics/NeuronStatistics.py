from src.statistics.NeuronStepStatistics import NeuronStepStatistics

class NeuronStatistics:
    def __init__(self, N, dt):
        self.N = N
        self.dt = dt
        self.data: list[NeuronStepStatistics] = [] 

