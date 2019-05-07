import torch
from torch import Tensor

class Loss_fnc(object):
    def loss(self, output, target):
        raise NotImplementedError

    def backward(self, MLP):
        raise NotImplementedError

class MSE(Loss_fnc):
    def loss(self, output, target):
        self.output = output
        self.target = target
        diff = output - target
        length = diff.size()[0]
        return (diff * diff).sum()/ length
    
    def backward(self, model):
        # gradient of loss
        # diff size: batch_size * output_dim
        diff = self.output - self.target
        n = diff.size()[0]
        # back propagation of the model
        return model.backward(2 * diff/n)
    
class CrossEntropy(Loss_fnc):
    # binary cross entropy
    def loss(self, output, target):
        # output_size : batch_size * output_dim
        self.output = output
        self.target = target
        #print(self.target)
        y = 1/(1 + (self.output[:, 0] - self.output[:, 1]).exp())
        #print(y.size())
        return -(target[:, 0] * (y.log()) + (target[:, 1]) * ((1 - y).log())).mean()
    
    def backward(self, model):
        y = 1/(1 + (self.output[:, 0] - self.output[:, 1]).exp())
        n = self.output.size()[0]
        grad = -(self.target[:, 0] / y - (self.target[:, 1]) / (1 - y))
        
        return model.backward(grad/n)