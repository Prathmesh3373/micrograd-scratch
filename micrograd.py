
import numpy as np
import math 


class Value:

    def __init__(self,data,_children = (),_op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op
        self.grad = 0.0
        self.label = label

    def __repr__(self):
        return(f'Value(data={self.data})')
 
    def __add__(self, other):
        other = other if isinstance(other , Value) else Value(other)
        out = Value(self.data + other.data , (self , other) , '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out   
        
    def __mul__(self,other):
        other = other if isinstance(other , Value) else Value(other)
        out  = Value(self.data * other.data , (self,other) , '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            # return [self.grad , other.grad]
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
        
    def __pow__(self, other):
        assert isinstance(other , (int ,float))
        out = Value(self.data**other , (self,), "**")
        
        def _backward():
            self.grad += other *(self.data ** (other-1))*out.grad
            # return [self.grad]
        out._backward = _backward
        
        return out
    
    def tanh(self):
        n = self.data
        t = (math.exp(2*n) - 1) / (math.exp(2*n) + 1)
        out = Value(t,(self,),'tanh')

        def _backward():
             # there was a problem of overwriting the gradients in the backward pass, we need to add to the gradients instead of overwriting them.
            self.grad += (1.0 - t**2) * out.grad
        out._backward = _backward
        return out
    
    def __div__(self, other):
        other  = (other if isinstance(other , Value) else Value(other)) if other != 0 else Value(1e-10)
        out = Value(self.data / other.data  , (self,other) , '/')

        def _backward():
            self.grad += (1/other.data) * out.grad
            other.grad += (-self.data / (other.data**2)) * out.grad
        out._backward = _backward

        return out
    
    def __sub__(self, other):
        other = other if isinstance(other , Value) else Value(other)
        return self + (-other)
    
    def relu(self):
        out = max(0,Value(self.data , (self,) , 'ReLU'))
        
        def _backward():
            self.grad += out.grad if out.data > 0 else 0
        out._backward = _backward
        return out
    
    def softmax(self, data : np.ndarray):
        probs = []
        for i in data:
            exp = math.exp(i) / sum(math.exp(j) for j in data)
            probs.append(exp)
        
        def _backward():
            for i in probs:
                for j in data:
                    if i == j:
                        self.grad += (i * (1 - i)) * self.grad
                    else:
                        self.grad += (-i * j) * self.grad
        out = Value(probs , (self,) , 'Softmax')
        out._backward = _backward
            

        return probs
        


    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t , (self,) , 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()