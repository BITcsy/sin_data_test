import torch
import torch.nn as nn

class SinClassifyNet(nn.Module):
    def __init__(self, class_num, hidden_size):
        super(SinClassifyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, class_num)
        )

    def forward(self, input):
        output = self.fc(input)
        return output

    # def train():

    # def test():

class SinRegressionNet(nn.Module):
    def __init__(self, hidden_size):
        super(SinRegressionNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input):
        output = self.fc(input)
        return output

class Sin2DMLP(nn.Module):
    def __init__(self, area_size, hidden_size):
        super(Sin2DMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(area_size * area_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input):
        output = self.fc(input)
        return output

