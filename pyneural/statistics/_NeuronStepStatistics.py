from dataclasses import dataclass
from typing import Optional

@dataclass
class NeuronStepStatistics:
    """
    This is the data class for storing an information about a single step of a simulation of a neuron.
    """

    Vm: float = 0.0
    """Membrane potential in mV."""
    T: float = 0.0
    """Time at this step in ms."""

    I_total: float = 0.0
    """Total Current through a membrane in nA."""
    I_ext: float = 0.0
    """External current stimulation in nA."""
    # currents through specific channels are defined only for the Hodgkin-Huxley model
    I_leak: Optional[float] = None
    """Leak current in nA. Only available for the Hodgkin-Huxley (HH) model."""
    I_K: Optional[float] = None
    """Potassium current in nA. Only available for the Hodgkin-Huxley (HH) model."""
    I_Na: Optional[float] = None
    """Sodium current in nA. Only available for the Hodgkin-Huxley (HH) model."""

    g_m: float = 0.0
    """Total membrane conductance."""
    # specific channel conductances are defined only for the Hodgkin-Huxley model
    g_leak: Optional[float] = None
    """Leak conductance. Only available for the Hodgkin-Huxley (HH) model."""
    g_K: Optional[float] = None
    """Potassium conductance. Only available for the Hodgkin-Huxley (HH) model."""
    g_Na: Optional[float] = None
    """Sodium conductance. Only available for the Hodgkin-Huxley (HH) model."""

    spiked: bool = False
    """Whether or not there was a spike at this step."""

    # gate values are defined only for the Hodgkin-Huxley model
    gate_n: Optional[float] = None
    """State of the n gates of the potassium channel. Only available for the Hodgkin-Huxley (HH) model."""
    gate_m: Optional[float] = None
    """State of the m gates of the sodium channel. Only available for the Hodgkin-Huxley (HH) model."""
    gate_h: Optional[float] = None
    """State of the h gates of the sodium hannel. Only available for the Hodgkin-Huxley (HH) model."""

