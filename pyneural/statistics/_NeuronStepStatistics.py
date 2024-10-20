from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class NeuronStepStatistics:
    """
    This is the data class for storing an information about a single step of a simulation of a neuron.
    """

    Vm: np.ndarray = np.array([])
    """Membrane potential in mV."""
    T: float = 0.0
    """Time at this step in ms."""

    I_total: np.ndarray = np.array([])
    """Total Current through a membrane in nA."""
    I_ext: np.ndarray = np.array([])
    """External current stimulation in nA."""
    # currents through specific channels are defined only for the Hodgkin-Huxley model
    I_leak: Optional[np.ndarray] = None
    """Leak current in nA. Only available for the Hodgkin-Huxley (HH) model."""
    I_K: Optional[np.ndarray] = None
    """Potassium current in nA. Only available for the Hodgkin-Huxley (HH) model."""
    I_Na: Optional[np.ndarray] = None
    """Sodium current in nA. Only available for the Hodgkin-Huxley (HH) model."""

    # specific channel conductances are defined only for the Hodgkin-Huxley model
    g_leak: Optional[np.ndarray] = None
    """Leak conductance. Only available for the Hodgkin-Huxley (HH) model."""
    g_K: Optional[np.ndarray] = None
    """Potassium conductance. Only available for the Hodgkin-Huxley (HH) model."""
    g_Na: Optional[np.ndarray] = None
    """Sodium conductance. Only available for the Hodgkin-Huxley (HH) model."""

    # gate values are defined only for the Hodgkin-Huxley model
    gate_n: Optional[np.ndarray] = None
    """State of the n gates of the potassium channel. Only available for the Hodgkin-Huxley (HH) model."""
    gate_m: Optional[np.ndarray] = None
    """State of the m gates of the sodium channel. Only available for the Hodgkin-Huxley (HH) model."""
    gate_h: Optional[np.ndarray] = None
    """State of the h gates of the sodium hannel. Only available for the Hodgkin-Huxley (HH) model."""

