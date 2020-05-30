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
    
    
  def Increm_train(self, net, device, exemplars, parameters):
    
  
  
  def NCM_Classifier(self, data, exemplars, net, current_class, classifier):
    net.train(False)
    
    preds=dict((k, []) for k in range(current_class-10,current_class))
    
    for label in exemplars:
      data_load=DataLoader(exemplars[label],batch_size=128,shuffle=False, num_workers=4, drop_last=False) 
      mean=torch.zeros((1,64), device=self.device)
      for img, _ in data_load:
        with torch.no_grad():
          img = img.to(self.device)
          feature = net(img,features=True)
          features.append(feature)
          mean += feature
      mean = mean/ len(exemplars[label])
      means[label] = mean / mean.norm()
    
    data_loader = get_dataloadet(data)
    running_corrects=0
    for img,label in data_loader:
      with torch.no_grad():
        img=img.to(self.device)
        feature = net(img,features=True)
        minimum=10000
        for label in means:
          if torch.dist(feature, means[label])<minimum):
            prediction=label
            minimum=torch.dist(feature,means[label])
        if prediction == label:
          running_corrects+=1
    acc = running_corrects/len(data)
    
    print(f"Accuracy: {acc}")
    return acc    
    
      
     
  def constructExemplar(self, data, net, current_class): #Per una classe set di exemplars
    m=(self.k)/current_class
    
    items=dict((idx,[]) for idx in range(current_class-10,current_class))
    exemplars=dict((idx,[]) for idx in range(current_class-10,current_class))
    
    for img in data:
      for idx in items:
        if img[1] == idx:
          items[idx].append(img)
    
    
    for label in items:           #For each class
      data_loader = DataLoader(items[label],batch_size=128,shuffle=False, num_workers=4, drop_last=False)    #Dataloader
      features=[]
      mean = torch.zeros((1,64), device=self.device)  #mean=0
    
      net.train(False)
      for img, _ in data_loader:  
        with torch.no_grad():
          img.to(self.device)
      
          feature = net(img,features=True)      #Extract feature
          features.append(feature)
          mean += feature
      mean = (mean/len(features))
      mean = mean/mean.norm()
   
      
      for i in range(m):
        minimum=10000
        summ = sum(exemplars[label])
        for index,instance in enumerate(features):
          ph = (instance + summ)/(i+1)
          ph = ph/ph.norm()
          
          if torch.dist(ph,mean)<minimum:
            minimum=torch.dist(ph,mean)
            min_index=index
        exemplars[label].append(items[label][min_index])
        features.drop(min_index)
    return exemplars
  
  
  
   
      
      
      
      
      
      
      
      
      
      
    
    
    
