import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet

import sys

class FTOPS_Mlp(BaseNet):

    def __init__(self, **kwargs):
        super().__init__()

        # Copy kwargs
        kwargs = dict(kwargs)

        self.num_features = kwargs['num_features'] 
        self.rep_dim = kwargs['training']['rep_dim']
        
        self.fc_params = kwargs['fc_params']
        if self.fc_params is not None:
            fcs = []
            in_dim = self.num_features
            for out_dim, drop_rate in self.fc_params:
                fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim, bias=False), nn.ReLU(), nn.Dropout(drop_rate)))
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, self.rep_dim, bias=False))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None
 
    def forward(self, x):
        x = self.fc(x)
        return x

class FTOPS_Mlp_Autoencoder(BaseNet):

    def __init__(self, **kwargs):
        super().__init__()

        self.num_features = kwargs['num_features']
        self.rep_dim = kwargs['rep_dim']


        #encoder
        self.fc1 = nn.Linear(self.num_features, 32, bias=False)
        self.fc2 = nn.Linear(32, 16, bias=False)
        self.fc3 = nn.Linear(16, self.rep_dim, bias=False)
       
        #decoder
        self.fc4 = nn.Linear(self.rep_dim, 16, bias=False)
        self.fc5 = nn.Linear(16, 32, bias=False)
        self.fc6 = nn.Linear(32, self.num_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = F.leaky_relu(x)
        x = self.fc5(x)
        x = F.leaky_relu(x)
        x = self.fc6(x)
 
        return x


