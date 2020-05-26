import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
import numpy as np

num_class=100
lr=2
weight_dec=0.2

class iCaRL():
  def __init__(self, k=2000, n_classes, device):
    self.k=k
    self.n_classes=n_classes
    self.device=device
  
  def constructExemplar(self, data, net, current_n_class): #Per una classe set di exemplars
    m=(self.k)/current_n_class
    
    means=[]
    exemplars=[]
    outputs=[]
    sum=0
    
    net.train(False)
    for img, _ in data:
      img.to(self.device)
      
      feature = net(img,feature=True)
      feature = feature.to(self.device)/feature.to(self.device).norm()
      features.append(feature[
      for outp in out:
        mean+=outp
      mean=(sum/len(output))
      mean= mean/mean.norm()
      
    for i in range(m):
      
      
      
      
    
    
    
