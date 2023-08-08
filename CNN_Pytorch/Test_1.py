
## using MNIST dataset 

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
# trainset = torchvision.datasets.MNIST('mnist', 
#                                       train = True, 
#                                       download = True,
#                                       transform = transform)

# # Load our Test Data and specify what transform to use when loading
# testset = torchvision.datasets.MNIST('mnist', 
#                                      train = False,
#                                      download = True,
#                                      transform = transform)


# print(trainset.data[5])

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# # Define our imshow function 
# def imshow(title="", image = None, size = 6):
#     w, h = image.shape[0], image.shape[1]
#     aspect_ratio = w/h
#     plt.figure(figsize=(size * aspect_ratio,size))
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.title(title)
#     plt.show()

# # image=trainset.data[5].numpy()
# # imshow("The sample is",image)

# # Let's view the 50 first images of the MNIST training dataset
# import matplotlib.pyplot as plt

# figure = plt.figure()
# num_of_images = 50 

# for index in range(1, num_of_images + 1):
#     plt.subplot(5, 10, index)
#     plt.axis('off')
#     plt.imshow(trainset.data[index], cmap='gray_r')

# # Prepare train and test loader
# trainloader = torch.utils.data.DataLoader(trainset,
#                                            batch_size = 128,
#                                            shuffle = True,
#                                            num_workers = 0)

# testloader = torch.utils.data.DataLoader(testset,
#                                           batch_size = 128,
#                                           shuffle = False,
#                                           num_workers = 0)

# # import matplotlib.pyplot as plt
# import numpy as np

# # function to show an image
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()                                     ## gotta figure out there is some error in iter and next function 


# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# # show images
# imshow(torchvision.utils.make_grid(images))

# # print labels
# print(''.join('%1s' % labels[j].numpy() for j in range(128)))



import torch.nn as nn 
import torch.nn.functional as F

class Net (nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    # now we set our CNN layers one by one 

  # nn.Conv2d(in_channels=1,
          # out_channels=32,
          # kernel_size=3,
          # stride=1, 
          # padding=1)

    # we are setting our first cnn layer using 32 filters, of 3X3 size and the input dimension as 1 , leaving out the passing and stride to be defaullts ie stride=1 and padding =0
    self.conv1=nn.Conv2d(1,32,3)
    # Refer my notes for detailed explanataion for other layers
    self.conv2=nn.Conv2d(32,64,3)
    # our maxpool is a 2X2 kernel of stride 2
    self.pool = nn.MaxPool2d(2, 2)
    # our first Fully connected layer takes the output from maxpool and connect them to 128 nodes
    self.fc1=nn.Linear(64*12*12,128)
    # our second FC = connect the 128 nodes to 10 node
    self.fc2=nn.Linear(128,10)

  def forward(self,x):
    # here we link the objects which we created in the prev function
    # The order is con1 - Relu - conv2 -Relu - Max pool - Flatten - FC1 - FC2

    x=F.relu(self.conv1(x))
    x=self.pool(F.relu(self.conv2(x)))
    x=x.view(-1,64*12*12)
    x=F.relu(self.fc1(x))
    x=self.fc2(x)

    return x

net=Net()
net.to(device)
print(net)