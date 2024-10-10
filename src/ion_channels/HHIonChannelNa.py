from src.ion_channels.IonChannel import IonChannel
from src.ion_channels.MarkovIonGate import MarkovIonGate
import numpy as np


class HHIonChannelNa(IonChannel):
    g = 0.0

    def __init__(self, gNa: float, v_init: float=0):
        self.m_gate = MarkovIonGate(alpha=lambda v: ((25 - v) / 10) / (np.exp(0.1 * (25 - v)) - 1),
                                    beta=lambda v: 4 * np.exp(-v / 18),
                                    v_init=v_init)

        self.h_gate = MarkovIonGate(alpha=lambda v: 0.07 * np.exp(-v / 20),
                                    beta=lambda v: 1 / (np.exp((30 - v) / 10) + 1),
                                    v_init=v_init)

        self.g_max: float = gNa
        self.g = self.g_max * np.power(self.m_gate.state, 3) * self.h_gate.state

    def update_g(self, v, t, dt):
        self.m_gate.update(v, dt)
        self.h_gate.update(v, dt)
        self.g = self.g_max * np.power(self.m_gate.state, 3) * self.h_gate.state
        return self.g
