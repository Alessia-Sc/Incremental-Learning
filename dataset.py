from torchvision.datasets import VisionDataset
from torchvision import transforms
import numpy as np

class cifar100(VisionDataset):
  def __init__(self, root, train, transform=None, target_transform=None):
    super(cifar100, self).__init__(root, transform=transform, target_transform=target_transform)
        
    self.dataset = CIFAR100(root=root, train=train, download=True, transform=transform)
    classes=range(100)
    self.class_groups = {}
        
    for i, idx in zip(range(10),range(0,100,10)):
      self.class_groups[i]=classes[idx:idx+10]
         
    self.groups=dict((k,[]) for k in range(10)) 
    for idx,img in enumerate(self.dataset):
      for i in range(10):
        if img[1] in self.class_groups[i]:
          self.groups[i].append(idx)
              
              
  def __len__(self):
    return len(self.dataset)
          
  def __getitem__(self,index):
    image, label = self.dataset[index]
    return image, label
        
  def __getGroup__(self,index):
    return self.groups[index]
        
        
