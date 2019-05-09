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
        softmax = torch.softmax(output, dim = 1)
        loss = (-target * (softmax.log())).sum(dim = 1)
        return loss.mean()
        
    def backward(self, model):
        n = self.output.size()[0]
        softmax = torch.softmax(self.output, dim = 1)
        y = -self.target[:, 0] * softmax[:, 1] + self.target[:, 1] * softmax[:, 0]
        y = y.view(-1, 1)
        grad = torch.cat([y, -y], dim = 1)
        return model.backward(grad/n)