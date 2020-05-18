import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from torchvision.utils import save_image
import torchvision.utils as utils
from torchvision import transforms
from torchsummary import summary

import matplotlib.pyplot as plt
import numpy as np

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.maxpool1 = nn.MaxPool2d(2, return_indices=True)
    self.tanh1 = nn.Tanh()
    self.conv2 = nn.Conv2d(6, 12, 5)
    self.maxpool2 = nn.MaxPool2d(2, return_indices=True)
    self.tanh2 = nn.Tanh()
    self.conv3 = nn.Conv2d(12,16, 5)
    self.tanh3 = nn.Tanh()

  def forward(self, x):
    indices=[]
    x = self.conv1(x)
    x , ind = self.maxpool1(x)
    indices.append(ind)
    x = self.tanh1(x)
    x = self.conv2(x)
    x , ind = self.maxpool2(x)
    indices.append(ind)
    x = self.tanh2(x)
    x = self.conv3(x)
    x = self.tanh3(x)
    return x,indices

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    self.convT3 = nn.ConvTranspose2d(16, 12, 5)
    self.tanh2 = nn.Tanh()
    self.maxunpool2 = nn.MaxUnpool2d(2)
    self.convT2 = nn.ConvTranspose2d(12 , 6, 5)
    self.tanh1 = nn.Tanh()
    self.maxunpool1 = nn.MaxUnpool2d(2)
    self.convT1 = nn.ConvTranspose2d(6,3,5)
    self.tanh0 = nn.Tanh()
  
  def forward(self, x,indices):
    x = self.convT3(x)
    x = self.tanh2(x)
    x = self.maxunpool2(x, indices[1])
    x = self.convT2(x)
    x = self.tanh1(x)
    x = self.maxunpool1(x, indices[0])
    x = self.convT1(x)
    x = self.tanh0(x)
    return x

class Autoencoder(nn.Module):
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()
  
  def forward(self, x):
    x, indices = self.encoder(x)
    x = self.decoder(x, indices)
    return x


def get_model():
    PATH = './autoencoder_COCO_data.pth'
    net = Autoencoder()
    net.load_state_dict(torch.load(PATH))
    net.eval()
    net1 = nn.Sequential(*list(net.children())[:1])
    return net1

def transform_image(image_size=128):
    transform = transforms.Compose([
                                transforms.Resize((image_size,image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    return transform

if __name__=='__main__':
    model = Autoencoder()
    print(model)
    model = nn.Sequential(*list(model.children())[:1])
    print(model)
