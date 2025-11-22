import numpy as np

class Moonet:
    def __init__(self, n_inputs, n_neurons: int):
        self.weights = np.random.normal(0,1,(n_inputs, n_neurons))
        self.bias = np.zeros(n_neurons)

    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x,self.weights.T) + self.bias


