from MLP.micrograd import Value
import numpy as np
import math
class Neuron:
    def __init__(self , nin):
        self.w = [Value(np.random.randn()) for _ in range(nin)]
        self.b = Value(np.random.randn())

    def __call__(self,x):
        act = sum((wi*xi for wi,xi in zip(self.w,x)) , self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self , nin , nouts):
        neurons = [Neuron(nin) for _ in range(nouts)]
        self.neurons = neurons

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    

    def __call__(self , x):
        outs = [n(x) for n in self.neurons]
        return outs
    
class MLP:
    def __init__(self , nin , nouts):
        sz = [nin]+nouts
        self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(nouts))]

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __call__(self , x):
        for layer in self.layers:
            x = layer(x)
        return x
    

