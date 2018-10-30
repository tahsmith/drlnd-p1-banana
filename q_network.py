from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.qa = nn.Sequential(
          nn.Linear(in_features=state_size, out_features=64),
          nn.ELU(),
          nn.Linear(in_features=64, out_features=64),
          nn.ELU(),
          nn.Linear(in_features=64, out_features=action_size),
        )

    def forward(self, state):
        return self.qa(state)
