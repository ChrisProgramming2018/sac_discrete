import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from  torch.distributions.categorical import Categorical


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, fc_1, fc_2):
        super(QNetwork, self).__init__()
        self.leak = 0.01
        self.state_size = state_size
        self.action_size = action_size
        self.layer1 = nn.Linear(state_size, fc_1)
        self.layer2 = nn.Linear(fc_1, fc_2)
        self.layer3 = nn.Linear(fc_2, action_size)
        self.layer4 = nn.Linear(state_size, fc_1)
        self.layer5 = nn.Linear(fc_1, fc_2)
        self.layer6 = nn.Linear(fc_2, action_size)
        self.reset_parameters()
    
    def forward(self, state):
        x1 = F.relu(self.layer1(state))
        x2 = F.relu(self.layer2(x1))
        x3 = self.layer3(x2)

        x4 = F.relu(self.layer4(state))
        x5 = F.relu(self.layer5(x4))
        x6 = self.layer6(x5)
        return x3, x6
    
    def Q1(self, state):
        x1 = F.relu(self.layer1(state))
        x2 = F.relu(self.layer2(x1))
        x3 = self.layer3(x2)
        return x3

    def reset_parameters(self):
        """ Initilaize the weights using He et al (2015) weights """
        torch.nn.init.kaiming_normal_(self.layer1.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layer2.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.layer3.weight.data, -3e-4, 3e-4)
        torch.nn.init.kaiming_normal_(self.layer4.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layer5.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.layer6.weight.data, -3e-4, 3e-4)




class SACActor(nn.Module):
    def __init__(self, state_size, action_size, fc_1=64, fc_2=64, action_space=None):
        super(SACActor, self).__init__()
        self.leak = 0.01
        self.state_size = state_size
        self.action_size = action_size
        self.layer1 = nn.Linear(state_size, fc_1)
        self.layer2 = nn.Linear(fc_1, fc_2)
        self.layer3 = nn.Linear(fc_2, action_size)
        self.softmax = F.softmax
        self.reset_parameters()

    def reset_parameters(self):
        """ Initilaize the weights using He et al (2015) weights """
        torch.nn.init.kaiming_normal_(self.layer1.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layer2.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.layer3.weight.data, -3e-4, 3e-4)


    def forward(self, state):
        x1 = F.relu(self.layer1(state))
        x2 = F.relu(self.layer2(x1))
        x3 = self.layer3(x2)
        action_prob = self.softmax(x3, dim=1)
        log_prob = action_prob + torch.finfo(torch.float32).eps
        log_prob = torch.log(log_prob)
        return action_prob, log_prob


    def sample(self, state):
        action_prob, log_prob = self.forward(state)
        m = Categorical(action_prob)
        action = m.sample()
        return action
