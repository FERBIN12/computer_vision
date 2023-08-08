
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

