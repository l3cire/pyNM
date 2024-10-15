from ._ConstCondNeuron import ConstCondNeuron

class LIFNeuron(ConstCondNeuron):
    def __init__(self, params: dict = {}):
        super().__init__()
        self.V_reset = params.get('V_reset', -80.099)
        self.V_spike = params.get('V_spike', 35.685)

    def step(self, t, dt):
        stats = super().step(t, dt)
        if(self.V > self.V_threshold):
            self.V = self.V_reset
            stats.Vm = self.V_spike
        return stats


