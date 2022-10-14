"""
# author: Zhaoyang Li
# 2022 10 13
# Central South University
"""
import torch
import torch.nn as nn

class ProxyMLP(nn.Module):
    def __init__(self, in_features:int=8, out_responses:int=2):
        super().__init__()
        self.proxymlp = nn.Sequential(nn.Linear(in_features=in_features, out_features=32),
                                     nn.ELU(),
                                     nn.Linear(in_features=32, out_features=32),
                                     nn.ELU(),
                                     nn.Linear(in_features=32, out_features=32),
                                     nn.ELU(),
                                     nn.Linear(in_features=32, out_features=32),
                                     nn.ELU(),
                                     nn.Linear(in_features=32, out_features=out_responses),
                                     nn.Sigmoid(),)
    def forward(self, x):
        return self.proxymlp(x)