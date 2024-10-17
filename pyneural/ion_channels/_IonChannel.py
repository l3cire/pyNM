from abc import ABC, abstractmethod


class IonChannel(ABC):
    """
    Ion channel base class. Ion channels are a key instrument by which neuron cells regulate their membrane potential. Different ion channels can vary their conductance based on the circumstances, which allows different ions to flow through the membrane.

    Attributes:
        g: conductace of a channel.
    """
    g: float = 0
    
    @abstractmethod
    def update_g(self, v: float, t: float, dt: float) -> float:
        """
        Updates the conductance of the channel based on current time and membrane potential. Returns new conductance.

        :param v: current membrane potential relative to the resting potential in mV.
        :param t: current time in ms.
        :param dt: the time interval between two consecutive updates in ms.
        """
        pass

    @abstractmethod
    def reset(self, v_init: float):
        """
        Resets the conductance to the stable level, given that the membrane potential is constant.

        :param v_init: the membrane potential (relative to the resting potential) in mV.
        """
        pass
