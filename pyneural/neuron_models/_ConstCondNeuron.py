from typing import Optional
from ..statistics import NeuronStepStatistics
from ._Neuron import Neuron
from ..ion_channels import IonChannelConst

class ConstCondNeuron(Neuron):
    """
    A neuron with constant conductance for each ion channel type.
    """

    I_ext: float = 0.0

    def __init__(self, params: dict = {}):
        """
        Initialize a new neuron.

        Apart from parameters specified in the base class `pyneural.neuron_models.Neuron`, this class requires additional parameters to specify conductances. There are two possible ways to specify a neuron. First one (preferred):
        :param params['g_m']: total membrane conductance (1.0 by default).
        :param params['V_rest']: resting potential in mV (-70.0 by default).
        :param params['tau']: time constant (10.0 by default)
        
        However, it is also possible to specify conductances and reversal potentials for each type of ion channels separately:
        :param params['gL']: leak ion channels conductance (0.3 by default).
        :param params['gK']: potassium ion channels conductance (0.366 by default).
        :param params['gNa']: sodium ion channels conductance (0.0106 by default).
        :param params['E_L']: leak ion channels reversal potantial in mV (-59.4 by default).
        :param params['E_K']: potassium ion channels reversal potential in mV (-82.0 by default).
        :param params['E_Na']: sodium ion channels reversal potential in mV (45.0 by default).
        :param params['C_m']: membrane capacitance in μF/cm2 (1.0 by default).

        The first option is preferred; however, if any of the parameters from the second option are specified, it is chosen instead.
        """
        super().__init__(params)

        if any(param in params for param in ['gL', 'gK', 'gNa', 'E_L', 'E_K', 'E_Na', 'C_m']):
            g_L = params.get('gL', 0.3)
            g_K = params.get('gK', 0.366)
            g_Na = params.get('gNa', 0.0106)
            E_L = params.get('E_L', -59.4)
            E_K = params.get('E_K', -82.0)
            E_Na = params.get('E_Na', 45.0)
            C_m = params.get('C_m', 1.0)
            
            self._g_m = IonChannelConst(g_L + g_K + g_Na)
            self._V_rest = (g_L*E_L + g_K*E_K + g_Na*E_Na)/(g_L + g_K + g_Na)
            self._tau = C_m/(g_L + g_K + g_Na)
        else:
            self._g_m = IonChannelConst(params.get('g_m', 1.0))
            self._V_rest = params.get('V_rest', -70.0)
            self._tau = params.get('tau', 10.0)


    def step(self, t: float, dt: float) -> NeuronStepStatistics:
        stats = NeuronStepStatistics()
        stats.T = t
        stats.g_m = self._g_m.update_g(self._V, t, dt)
        stats.I_ext = self.I_ext
        stats.I_total = self.I_ext - self._g_m.g*(self._V - self._V_rest)

        self._V += (-(self._V - self._V_rest) + self.I_ext/self._g_m.g)*dt/self._tau
        stats.Vm = self._V

        return stats

    def reset(self, V: Optional[float] = None):
        return super().reset(V)

