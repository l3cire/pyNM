"""
This module contains different models for external current stimulation of a neuron. All of these models are inherited from an abstract class `pyneural.input_current.InputCurrent`.
"""

from ._InputCurrent import InputCurrent
from ._ConstInputCurrent import ConstInputCurrent
from ._NoisyConstInputCurrent import NoisyConstInputCurrent

CONST_ZERO_INPUT = ConstInputCurrent()

__all__ = [
    'InputCurrent',
    'ConstInputCurrent',
    'NoisyConstInputCurrent',
    'CONST_ZERO_INPUT'
]
