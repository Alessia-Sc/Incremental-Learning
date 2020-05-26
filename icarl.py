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
    
    exemplars=[]
    feature=[]
    mean = torch.zeros((1,64), device=self.device) 
    
    net.train(False)
    for img, _ in data:
      img.to(self.device)
      
      features = net(img,feature=True)
      feature.append(features)
      mean += features
    mean = (mean/len(data))
    mean = mean/mean.norm()
        
    sigm = nn.Sigmoid()
    for x in feature:
      outputs.append(sigm(feature))
      
    for i in range(m):
      minimum=10000
      summ = sum(outputs)
      exemplar = mean - 1/i *(feature[i] + 
      
      
      
      
      
      
    
    
    
