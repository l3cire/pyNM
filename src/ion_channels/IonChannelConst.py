from src.ion_channels.IonChannel import IonChannel


class IonChannelConst(IonChannel):
    g = 0.0

    def __init__(self, g):
        self.g = g

    def update_g(self, v, t, dt) -> float: 
        return self.g

