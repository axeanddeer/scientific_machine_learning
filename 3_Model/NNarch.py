import torch
import torch.nn  as nn
import torch.nn.functional as F
#Filter response normalization
#from frn import FRN, TLU
#Cosine annealing warm restarts for learning rate
from utils import setLR, cosineSGDR
from torch.nn.utils import spectral_norm
from netCDF4 import Dataset
from torch.utils import data
import pandas as pd
from tqdm import tqdm

# class some{defaults
#           def fun()}
# ob = some()
# ob.forward(x) = ob(x)
def iden(x):
    return x

class FcLayer(nn.Module):

    def __init__(self, nhidden=100, drop = 0.3, res = True, norm = False):
        super().__init__()
        self.res = res

        self.lin = nn.Linear(nhidden, nhidden)
        self.drop = nn.Dropout(p=drop)
        if norm:
            self.norm = nn.LayerNorm(nhidden)
        else:
            self.norm = iden
        #SiLU is an activation function x * torch.sigmoid(x)
        self.activ = nn.SiLU()

    def forward(self, x):
        if self.res:
            return x + self.drop(self.norm(self.activ(self.lin(x))))
        else:
            return self.drop(self.norm(self.activ(self.lin(x))))


class ResMLP(nn.Module):
    def __init__(self, numLayers=3, nhidden = 50, ninp = 4, nout  = 2, drop = 0.15, attn = False):
        super(ResMLP, self).__init__()
        #self.dropInp = nn.Dropout(p=0.1)
        self.fci = nn.Linear(ninp, nhidden)
        self.activ = nn.SiLU()
        self.drop = nn.Dropout(p=drop)
        self.numLayers = numLayers
        if numLayers>0:
            self.fc_root = self._make_layers(numLayers, nhidden, drop)
        self.fco = nn.Linear(nhidden, nout)
        #self.fco2 = nn.Linear(2, 1)

    #Not to be accessed from outside the class with leading _name
    def _make_layers(self, numlayers, nhidden, drop):
        layers = []
        for _ in range(numlayers):
            layers.append(FcLayer(nhidden, drop))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activ(self.fci(x))
        if self.numLayers>0:
            out = self.fc_root(out)
        out = self.fco(out)
        return out 

class MLP(nn.Module):
    def __init__(self, numLayers=3, nhidden = 50, ninp = 10, drop = 0.15):
        super(MLP, self).__init__()
        self.activ = nn.SiLU()
        self.fci = nn.Linear(ninp, nhidden)
        self.numLayers = numLayers
        if numLayers>0:
            self.fclayers = self._make_layers(numLayers, nhidden, drop)
        self.fco = nn.Linear(nhidden, ninp)

    def forward(self, x):
        out = self.activ(self.fci(x))
        if self.numLayers>0:
           out = self.fclayers(out)
        return self.fco(out)   
    
    def _make_layers(self, numlayers, nhidden, drop):
        layers = []
        for _ in range(numlayers):
            layers.append(FcLayer(nhidden, drop))
        return nn.Sequential(*layers)

class Linear(nn.Module):
    def __init__(self, ninp = 10 ):
        super(Linear, self).__init__()
        self.fci = nn.Linear(ninp, 1)
    def forward(self, x):
        return self.fci(x)
        
class SelfFish(nn.Module):
    def __init__(self, numLayers=3, nhidden = 50, ninp = 10, nout  = 1, drop = 0.15):
        super(SelfFish, self).__init__()
        self.W1 = nn.Linear(ninp, nhidden)
        self.W2 = nn.Linear(nhidden, ninp)
        self.drop = nn.Dropout(p = drop)
        self.fci = nn.Linear(ninp, nhidden)
        self.sig = nn.Sigmoid()
        self.silu = nn.SiLU()
        self.numLayers = numLayers
        if numLayers>0:
            self.fclayers = self._make_layers(numLayers, nhidden, drop)
        self.fco = nn.Linear(nhidden, nout)
    
    def forward(self, x):
        switch = self.sig(self.W2(self.drop(self.silu(self.W1(x)))))
        out = x * switch
        out = self.fci(out)
        if self.numLayers>0:
           out = self.fclayers(out)
        return self.fco(out), switch

    def _make_layers(self, numlayers, nhidden, drop):
        layers = []
        for _ in range(numlayers):
            layers.append(FcLayer(nhidden, drop))
        return nn.Sequential(*layers)
