import numpy as np

class MooNet:
    def __init__(self, n_inputs, n_neurons: int):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward_pass(self, inputs: np.ndarray) -> np.ndarray:
        self.output = np.dot(inputs,self.weights) + self.biases


class Relu:
    def forward_pass(self, inputs: np.ndarray) -> np.ndarray:
        self.output = np.maximum(0,inputs)

class SoftMax:
    def forward_pass(self, inputs: np.ndarray):
        exp_norm = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_norm / np.sum(exp_norm, axis=1, keepdims=True)
        self.output = probabilities