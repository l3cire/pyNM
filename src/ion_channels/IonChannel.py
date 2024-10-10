from abc import ABC, abstractmethod


class IonChannel(ABC):
    g: float = 0
    
    @abstractmethod
    def update_g(self, v, t, dt) -> float:
        pass
