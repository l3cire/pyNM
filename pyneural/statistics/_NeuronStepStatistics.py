from typing import Optional
import numpy as np

class NeuronStepStatistics:
    """
    This is the data class for storing an information about a single step of a simulation of group of neurons.
    """

    Vm: np.ndarray = np.array([])
    """Membrane potential for each neuron in mV."""
    T: float = 0.0
    """Time at this step in ms."""

    I_total: np.ndarray = np.array([])
    """Total current through a membrane for each neuron in nA."""
    I_ext: np.ndarray = np.array([])
    """External current stimulation for each neuron in nA."""
    # currents through specific channels are defined only for the Hodgkin-Huxley model
    I_leak: Optional[np.ndarray] = None
    """Leak current for each neuron in nA. Only available for the Hodgkin-Huxley (HH) model."""
    I_K: Optional[np.ndarray] = None
    """Potassium current for each neuron in nA. Only available for the Hodgkin-Huxley (HH) model."""
    I_Na: Optional[np.ndarray] = None
    """Sodium current for each neuron in nA. Only available for the Hodgkin-Huxley (HH) model."""

    # specific channel conductances are defined only for the Hodgkin-Huxley model
    g_leak: Optional[np.ndarray] = None
    """Leak conductance for each neuron. Only available for the Hodgkin-Huxley (HH) model."""
    g_K: Optional[np.ndarray] = None
    """Potassium conductance for each neuron. Only available for the Hodgkin-Huxley (HH) model."""
    g_Na: Optional[np.ndarray] = None
    """Sodium conductance for each neuron. Only available for the Hodgkin-Huxley (HH) model."""

    # gate values are defined only for the Hodgkin-Huxley model
    gate_n: Optional[np.ndarray] = None
    """State of the n gates of the potassium channel for each neuron. Only available for the Hodgkin-Huxley (HH) model."""
    gate_m: Optional[np.ndarray] = None
    """State of the m gates of the sodium channel for each neuron. Only available for the Hodgkin-Huxley (HH) model."""
    gate_h: Optional[np.ndarray] = None
    """State of the h gates of the sodium channel for each neuron. Only available for the Hodgkin-Huxley (HH) model."""

