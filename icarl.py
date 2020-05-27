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
  
  
  def Classifier(self, data
  
  
  
  def constructExemplar(self, data, net, current_class): #Per una classe set di exemplars
    m=(self.k)/current_class
    
    items=dict((idx,[]) for idx in range(current_class-10,current_class))
    exemplars=dict((idx,[]) for idx in range(current_class-10,current_class))
    
    for img in data:
      for idx in items:
        if img[1] == idx:
          items[idx].append(img)
    
    
    for label in items:           #For each class
      data_loader = get_dataloader(items[label])    #Dataloader
      features=[]
      mean = torch.zeros((1,64), device=self.device)  #mean=0
    
      net.train(False)
      for img, _ in data_loader:    
        img.to(self.device)
      
        feature = net(img,feature=True)      #Extract feature
        features.append(feature)
        mean += feature
      mean = (mean/len(features))
      mean = mean/mean.norm()
        
      sigm = nn.Sigmoid()
   
      outputs=[]
      for i in range(m):
        minimum=10000
        summ = sum(outputs)
        for index,instance in enumerate(features):
          ph = (instance + summ)/(i+1)
          ph = ph/ph.norm()
          
          if torch.dist(ph,mean)<minimum:
            minimum=torch.dist(ph,mean)
            min_index=index
        outputs.append(sigm(instance))
        exemplars[label].append(items[label][min_index])
    
    return exemplars
  
  
  
   
      
      
      
      
      
      
      
      
      
      
    
    
    
