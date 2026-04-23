#!/usr/bin/env python3
# neuron.py
# Copyright Â© Spiking Neurons AB

A = [2.0, -3.0, 4.5]  # indata
W = [0.5, 1.2, -2.0]  # Model weights for this neuron
B = 1.2

x = 0.0
for i in range(len(A)):
    x += A[i] * W[i]
x += B

# activation function
y = x if x > 0 else 0.01 * x

print(f"{y}")