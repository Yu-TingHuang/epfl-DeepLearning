import torch
from torch import Tensor

class Optimizer(object):
    def step(self):
        raise NotImplementedError                   
    def one_pass(self, input, target, loss_fnc, model):
        # one forward and backward pass
        output = model.forward(input)
        error = loss_fnc.loss(output, target)
        # loss backward -> MLP backward -> module backward
        loss_fnc.backward(model)
        model.update(self)
        return error

class GD(Optimizer):
    def __init__(self, gamma):
        self.gamma = gamma
    def step(self, w, grad):
        return w - self.gamma * grad
    def train(self, input, target, loss_fnc, model, nb_epoch):
        for i in range(nb_epoch):
            error = self.one_pass(input, target, loss_fnc, model)
            print(error)
            
class SGD(Optimizer):
    def __init__(self, gamma, batch_size):
        self.gamma = gamma
        self.batch_size = batch_size
    def step(self, w, grad):
        return w - self.gamma * grad  
    def train(self, input, target, loss_fnc, model, nb_epoch):  
        for i in range(nb_epoch):
            idx = torch.randperm(input.size()[0])
            batch_num = input.size()[0] / self.batch_size
            for i in range(batch_num):
                batch_idx = idx[i * self.batch_size: (i+1)* self.batch_size - 1]
                train_input = input[batch_index, :]
                train_target = target[batch_index, :]
                error = self.one_pass(train_input, train_target, loss_fnc, model)