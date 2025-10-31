from torch import nn
from torch.nn import functional as f

class QNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = f.relu(self.linear2(x))
        x = f.relu(self.linear3(x))
        x = f.relu(self.linear4(x))
        x = self.linear5(x)
        return x