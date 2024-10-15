from dataclasses import dataclass
from typing import Optional

@dataclass
class NeuronStepStatistics:
    Vm = 0.0
    T = 0.0

    I_total = 0.0
    I_leak = 0.0
    I_K = 0.0
    I_Na = 0.0
    I_ext = 0.0

    g_leak = 0.0
    g_K = 0.0
    g_Na = 0.0

    gate_n: Optional[float] = None
    gate_m: Optional[float] = None
    gate_h: Optional[float] = None

    spiked: bool = False

