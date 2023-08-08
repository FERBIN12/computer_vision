
import torch

## used to get dataset and useful image transformations
import torchvision
import torchvision.transforms as transforms


#Optimization library of pytorch and nn- used as basic building block for our Network graphs
import torch.optim as optim
import torch.nn as nn

print("GPU available: {}".format(torch.cuda.is_available()))   ## for printing about the availability of GPU

# #CUDA - Nvidia accleration library which is used to enable GPU performance

if torch.cuda.is_available():
  device='cuda'
else:
  device='cpu'

transform=transforms.Compose([transforms.ToTensor(),   # nothing but matrices in tensor form to be used by the GPU's
                              transforms.Normalize((0.5,0),(0.5,0)) ])

# Load our Training Data and specify what transform to use when loading
trainset = torchvision.datasets.MNIST('mnist', 
                                      train = True, 
                                      download = True,
                                      transform = transform)

# Load our Test Data and specify what transform to use when loading
testset = torchvision.datasets.MNIST('mnist', 
                                     train = False,
                                     download = True,
                                     transform = transform)


print(trainset.data[5])

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Define our imshow function 
def imshow(title="", image = None, size = 6):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

image=trainset.data[5].numpy()
imshow("The sample is",image)