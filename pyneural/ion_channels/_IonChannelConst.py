from ._IonChannel import IonChannel


class IonChannelConst(IonChannel):
    """
    Ion channel with constant conductance.

    Attributes:
        g: ion channel conductance.
    """
    g: float = 0.0

    def __init__(self, g: float):
        """
        Initialize a new ion channel with constant conductance.

        :param g: conductance of this channel.
        """
        self.g = g

    def update_g(self, v, t, dt) -> float: 
        return self.g

    def reset(self, v_init: float = 0):
        return

