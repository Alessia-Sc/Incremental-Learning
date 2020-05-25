import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
import numpy as np

num_class=100
lr=2
weight_dec=0.2

class iCaRL():
  def __init__(self, k=2000,n_classes):
    self.k=k
    self.n_classes=n_classes
  
  def constructExemplar(self, data, net):
    m=(self.k)/self.n_classes
    
    means=[]
    exemplars=[]
    
    
