from src.neuron_models.Neuron import Neuron
from src.ion_channels.IonChannelConst import IonChannelConst

class ConstCondNeuron(Neuron):
    def __init__(self, params: dict = {}):
        super().__init__(params)
        self.g_L = IonChannelConst(params.get('gL', 0.3))
        self.g_K = IonChannelConst(params.get('gK', 0.366))
        self.g_Na = IonChannelConst(params.get('gNa', 0.0106))
        self.V_rest = (self.g_L.g * self.E_L + self.g_K.g * self.E_K + self.g_Na.g * self.E_Na) / (self.g_L.g + self.g_K.g + self.g_Na.g)

