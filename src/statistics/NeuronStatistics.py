from src.statistics.NeuronStepStatistics import NeuronStepStatistics
from typing import Optional

class NeuronStatistics:
    def __init__(self, N, dt):
        self.N = N
        self.dt = dt
        self.data: list[Optional[NeuronStepStatistics]] = [None] * N
