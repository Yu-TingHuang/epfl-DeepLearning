import torch
from torch import Tensor
class Module(object):
    def forward(self, input):
        raise NotImplementedError
    def backward(self, upstream_derivative):
        raise NotImplementedError
    def param(self):
        return []
    
class Linear(Module):
    def __init__(self, input_dim, output_dim, bias = True):
        self.w = Tensor(input_dim, output_dim).normal_()
        self.b = Tensor(1, output_dim).normal_()

        self.gradW = Tensor(input_dim, output_dim).zero_()
        self.gradB = Tensor(1, output_dim).zero_()

        self.Bias = bias
        if(not bias):
            self.b.fill_(0)
    def forward(self, input):
        # input size : batch_size * input_dim
        self.input = input
        # output size: batch_size * output_dim
        self.output = input.mm(self.w) + self.b
        return self.output
    def backward(self, upstream_derivative):
        # upstream_derivatie: batch_size * output_dim    
        
        # gradient of weight
        self.gradW = (self.input.t()).mm(upstream_derivative)
        # gradient of bias
        if(self.Bias):
            self.gradB = upstream_derivative.sum(dim = 0)          
        return upstream_derivative.mm(self.w.t())
    def update(self, optimizer):
        self.w = optimizer.step(self.w, self.gradW)
        if(self.Bias):
            self.b = optimizer.step(self.b, self.gradB)
    def param(self):
        return []
    
class ReLU(Module):
    def forward(self, input):
        self.input = input
        self.positive = (input > 0).float()
        self.output = input.mul(self.positive)
        return self.output
    def backward(self, upstream_derivative):
        return self.positive.mul(upstream_derivative)
    def update(self, optimizer):
        pass
    def param(self):
        return []
    
class Tanh(Module):
    def forward(self, input):
        self.input = input
        self.output = (input.exp() - (-input).exp())/(input.exp() + (-input).exp())
        return self.output
    def backward(self, upstream_derivative):
        return upstream_derivative * (torch.ones_like(self.input) - self.output**2)
    def update(self, optimizer):
        pass
    def param(self):
        return []
    
class sigmoid(Module):
    def forward(self, input):
        self.input = input
        self.output = 1 / (1 + (-input).exp())
        return self.output
    def backward(self, upstream_derivative):
        return upstream_derivative * (self.output - self.output**2)
    def update(self, optimizer):
        pass
    def param(self):
        return []

class BN(Module):
    def forward(self, input):
        epsilon = 1e-6
        self.input = input
        sum = input.mean(dim = 0)
        std = input.std(dim = 0)
        denominator = torch.sqrt(std**2 + epsilon)
        self.output = (input - sum) / denominator
        return self.output
    def backward(self, upstream_derivative):
        return upstream_derivative
    def update(self, optimizer):
        pass
    def param(self):
        return []

class MLP(object):
    def __init__(self, *args):
        self.sequential_modules = []
        for module in args:
            self.sequential_modules.append(module)
    def forward(self, input):
        self.input = input
        output = input
        for module in self.sequential_modules:
            output = module.forward(output)
        self.output = output
        return self.output
    def backward(self, upstream_derivative):
        for module in reversed(self.sequential_modules):
            upstream_derivative = module.backward(upstream_derivative)
        self.grad = upstream_derivative
        return self.grad
    def update(self, optimizer):
        for module in (self.sequential_modules):
            module.update(optimizer)