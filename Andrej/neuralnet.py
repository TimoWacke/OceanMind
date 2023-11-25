from __future__ import annotations
import random
from value import Value 

class Neuron:
    def __init__(self, nin: int):
        self.w = [Value(random.uniform(-1,1), label=f'w{i}') for i in range(nin)]
        self.b = Value(random.uniform(-1,1), label='b')

    def __call__(self, x: list[Value]):
        # w * x + b
        act = self.w[0] * x[0]
        for wi, xi in zip(self.w[1:], x[1:]):
            act = act + wi * xi
        act = act + self.b
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin: int, nout: int):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
        
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
class MLP:
    def __init__(self, nin: int, nouts: list[int]):
        layersizes = [nin] + nouts
        # self.layers = [Layer(nin, nout) for nin, nout in zip(layersizes[:-1], layersizes[1:])]
        self.layers = [Layer(layersizes[i],layersizes[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            # apply each layer step by step to x
            x = layer(x)
        # return the last layer as result
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]