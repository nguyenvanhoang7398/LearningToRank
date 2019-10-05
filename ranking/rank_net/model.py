import torch
from torch import nn
import torch.nn.functional as F


class RankNet(nn.Module):
    def __init__(self, net_structures):
        super(RankNet, self).__init__()
        self.fc_layers = len(net_structures)
        for i in range(len(net_structures) - 1):
            layer = nn.Linear(net_structures[i], net_structures[i + 1])
            setattr(self, 'fc' + str(i + 1), layer)

        last_layer = nn.Linear(net_structures[-1], 1)
        setattr(self, 'fc' + str(len(net_structures)), last_layer)
        self.activation = nn.ReLU6()

    def forward(self, input1):
        for i in range(1, self.fc_layers):
            fc = getattr(self, 'fc' + str(i))
            input1 = F.relu(fc(input1))

        fc = getattr(self, 'fc' + str(self.fc_layers))
        return self.activation(fc(input1))


class RankNetPairs(RankNet):
    def __init__(self, net_structures):
        super(RankNetPairs, self).__init__(net_structures)

    def forward(self, input1, input2):
        # from 1 to N - 1 layer, use ReLU as activation function
        for i in range(1, self.fc_layers):
            fc = getattr(self, 'fc' + str(i))
            input1 = F.relu(fc(input1))
            input2 = F.relu(fc(input2))

        # last layer use ReLU6 Activation Function
        fc = getattr(self, 'fc' + str(self.fc_layers))
        input1 = self.activation(fc(input1))
        input2 = self.activation(fc(input2))

        # normalize input1 - input2 as a probability that doc1 should rank higher than doc2
        return torch.sigmoid(input1 - input2)
