from __future__ import annotations
import math

class Value:
    def __init__(self, data: float, children: list[Value] = [], _op: str = '', label: str = ''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = children
        self._op = _op
        self.label = label

    def __repr__(self):
        return f'Value({self.data})'
    
    def __add__(self, other) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, [self, other], '+')
    
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

     
    def __radd__(self, other) -> Value: # other + self
        return self + other

    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other) -> Value:
        other = other if isinstance(other, Value) else Value(other)	
        out = Value(self.data * other.data, [self, other], '*')
    
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other): # other * self
        return self * other

    def __pow__(self, other: float) -> Value:
        assert isinstance(other, (int, float)), 'only supporting float powers for now'
        out = Value(self.data ** other, [self], f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def tanh(self) -> Value:
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, [self], 'tanh')

        def _backward() -> None:
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward

        return out

    def backward(self) -> None:
        self.grad = 1.0
        visited = set()
        topo = []

        # we have to make sure that when we calculate the gradient of a node,
        # we have already calculated the gradient of it's out

        # we also want to have a neat list of nodes to go through

        def build_topo(v): # topological sort
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v) 
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
