import torch
import torch.nn as nn


class TransformNet(nn.Module):
    def __init__(self, size):
        super(TransformNet, self).__init__()
        self.size = size
        self.net = nn.Sequential(nn.Linear(self.size, self.size))

    def forward(self, input):
        out = self.net(input)
        return out / torch.sqrt(torch.sum(out ** 2, dim=1, keepdim=True))
    
class Mapping(nn.Module):
    def __init__(self, size):
        super(Mapping, self).__init__()
        self.size = size
        self.net = nn.Sequential(nn.Linear(self.size, self.size), nn.ReLU())
    
    def forward(self, inputs):
        outputs =self.net(inputs)
        return torch.cat((inputs, outputs), dim=-1)#/torch.sqrt(torch.sum(out**2,dim=1,keepdim=True))
