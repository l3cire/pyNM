from src.ion_channels.IonChannel import IonChannel
from src.ion_channels.MarkovIonGate import MarkovIonGate
import numpy as np


class HHIonChannelK(IonChannel):
    g = 0.0

    def __init__(self, gK: float, v_init: float = 0):
        self.n_gate = MarkovIonGate(alpha=lambda v: ((10 - v) / 100) / (np.exp(0.1 * (10 - v)) - 1),
                                    beta=lambda v: 0.125 * np.exp(-v / 80),
                                    v_init=v_init)

        self.g_max: float = gK
        self.g = self.g_max * np.power(self.n_gate.state, 4)

    def update_g(self, v, t, dt) -> float:
        self.n_gate.update(v, dt)
        self.g = self.g_max * np.power(self.n_gate.state, 4)
        return self.g
