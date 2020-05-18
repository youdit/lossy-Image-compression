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
from PIL import Image

from Models import get_model,transform_image

# img = Image.open("./images/plane.png")

# if Image.Image.getbands(img)==('R', 'G', 'B', 'A'):
#     img.load()
#     img2 = Image.new("RGB", img.size, (255,255,255))
#     img2.paste(img, mask=img.split()[3])
#     img2.show()

# to_tensor = transforms.ToTensor()

# img2 = to_tensor(img2)
# print(img2.size())
# img2 = transforms.ToPILImage()(img2)
# img2.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Run(root,device='cpu', show=False):
    img = Image.open(root)
    # if image is in RGBA so convert into RGB
    if Image.Image.getbands(img)==('R', 'G', 'B', 'A'):
        img.load()
        img2 = Image.new("RGB", img.size, (255,255,255))
        img2.paste(img, mask=img.split()[3])
        img = img2
    if show:
        img.show()

    
    # process image before feeding it network

    img = transform_image()(img)
    img = torch.unsqueeze(img,0)
    img = img.to(device)

    network = get_model().to(device)
    
    output,_ = network(img)
    output = torch.squeeze(img)
    return output

y = Run("./images/plane.png", device=device, show=True)

#save the tensor 
torch.save(y, 'file.pt')
print(y)

# img 3 - RGB RGBA G
# RGB H,W,C  