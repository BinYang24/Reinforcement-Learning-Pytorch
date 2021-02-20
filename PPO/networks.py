import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

class nature_cnn(nn.Module):
    def __init__(self,action_space,state_space):
        """
        CNN from Nature paper.
        """
        super(nature_cnn, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=state_space[-1],out_channels=32,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,kernel_size=4,stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64,kernel_size=3,stride=1)

        self.fc = nn.Linear(3136,512)
        self.logits = nn.Linear(512,action_space)
        self.value = nn.Linear(512,1)

        #code level skill 3

        nn.init.xavier_normal_(self.conv1.weight.data,gain=np.sqrt(2.))
        nn.init.xavier_normal_(self.conv2.weight.data,gain=np.sqrt(2.))
        nn.init.xavier_normal_(self.conv3.weight.data,gain=np.sqrt(2.))
        nn.init.constant_(self.conv1.bias.data,0.0)
        nn.init.constant_(self.conv2.bias.data,0.0)
        nn.init.constant_(self.conv3.bias.data,0.0)

        nn.init.xavier_normal_(self.fc.weight.data,gain=np.sqrt(2.))
        nn.init.constant_(self.fc.bias.data,0.0)
        nn.init.xavier_normal_(self.logits.weight.data,gain=np.sqrt(2.))
        nn.init.constant_(self.logits.bias.data,0.0)
        nn.init.xavier_normal_(self.value.weight.data)
        nn.init.constant_(self.value.bias.data,0.0)

    def forward(self,unscaled_images):
        s = unscaled_images / 255. # scale
        s = torch.transpose(s,1,3) # NHWC -> NCHW

        s = nn.functional.relu(self.conv1(s))
        s = nn.functional.relu(self.conv2(s))
        s = nn.functional.relu(self.conv3(s))

        s = s.view(-1,self.fc.in_features)

        s = nn.functional.relu(self.fc(s))
        logits = self.logits(s)

        p = torch.nn.Softmax(dim=-1)(logits) + 1e-8
        # p = torch.nn.Softmax(dim=-1)(logits)
        policy_head = Categorical(probs=p)

        # print('logits:',logits)
        # policy_head = Categorical(logits=logits)
        # policy_head.probs# will change random number????
        # print('policy sample:',policy_head.probs.shape)

        value = self.value(s)

        return policy_head,value
