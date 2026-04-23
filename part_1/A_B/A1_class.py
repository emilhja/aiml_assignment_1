#!/usr/bin/env python3
# neuron.py

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, inputs):
        x = 0.0
        for i in range(len(inputs)):
            x += inputs[i] * self.weights[i]
        x += self.bias

        return x if x > 0 else 0.01 * x


A = [2.0, -3.0, 4.5]  # indata
W = [0.5, 1.2, -2.0]  # Model weights for this neuron
B = 1.2

neuron = Neuron(W, B)
y = neuron.forward(A)

print(f"{y}")
