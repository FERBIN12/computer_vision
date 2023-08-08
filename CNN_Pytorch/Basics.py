import torch

## used to get dataset and useful image transformations
import torchvision
import torchvision.transforms as transforms

#Optimization library of pytorch and nn- used as basic building block for our Network graphs
import torch.optim as optim
import torch.nn as nn

print("GPU available: {}".format(torch.cuda.is_available()))   ## for printing about the availability of GPU

#CUDA - Nvidia accleration library which is used to enable GPU performance

