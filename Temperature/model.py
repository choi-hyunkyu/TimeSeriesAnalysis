import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 25)
        self.fc2 = nn.Linear(25, 50)
        self.fc3 = nn.Linear(50, 25)
        self.fc4 = nn.Linear(25, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x
