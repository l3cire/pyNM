class MarkovIonGate:
    state: float = 0.0

    def __init__(self, alpha, beta, v_init: float = 0):
        self.alpha = alpha
        self.beta = beta
        self.set_inf_state(v_init)

    def set_inf_state(self, v):
        self.state = self.alpha(v) / (self.alpha(v) + self.beta(v))

    def update(self, v, dt):
        self.state += (self.alpha(v) * (1 - self.state) - self.beta(v) * self.state) * dt
