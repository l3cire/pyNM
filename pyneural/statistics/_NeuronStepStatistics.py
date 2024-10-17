from dataclasses import dataclass
from typing import Optional

@dataclass
class NeuronStepStatistics:
    Vm = 0.0
    T = 0.0

    I_total = 0.0
    I_ext = 0.0
    # currents through specific channels are defined only for the Hodgkin-Huxley model
    I_leak: Optional[float] = None
    I_K: Optional[float] = None
    I_Na: Optional[float] = None

    g_m = 0.0
    # specific channel conductances are defined only for the Hodgkin-Huxley model
    g_leak: Optional[float] = None
    g_K: Optional[float] = None
    g_Na: Optional[float] = None

    spiked: bool = False

    # gate values are defined only for the Hodgkin-Huxley model
    gate_n: Optional[float] = None
    gate_m: Optional[float] = None
    gate_h: Optional[float] = None

