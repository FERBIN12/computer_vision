# Import PyTorch
import torch

# We use torchvision to get our dataset and useful image transformations
import torchvision
import torchvision.transforms as transforms

# Import PyTorch's optimization libary and nn
# nn is used as the basic building block for our Network graphs
import torch.optim as optim
import torch.nn as nn

# Are we using our GPU?
print("GPU available: {}".format(torch.cuda.is_available()))

if torch.cuda.is_available():
  device = 'cuda' 
else:
  device = 'cpu' 

# Transform to a PyTorch tensors and the normalize our valeus between -1 and +1
transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5, ), (0.5, )) ])

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

# Prepare train and test loader
trainloader = torch.utils.data.DataLoader(trainset,
                                           batch_size = 128,
                                           shuffle = True,
                                           num_workers = 0)

testloader = torch.utils.data.DataLoader(testset,
                                          batch_size = 128,
                                          shuffle = False,
                                          num_workers = 0)

import torch.nn as nn
import torch.nn.functional as F #

# Create our Model using a Python Class
class Net(nn.Module):
    def __init__(self):
        # super is a subclass of the nn.Module and inherits all its methods
        super(Net, self).__init__()

        # We define our layer objects here
        # Our first CNN Layer using 32 Fitlers of 3x3 size, with stride of 1 & padding of 0
        self.conv1 = nn.Conv2d(1, 32, 3)
        # Our second CNN Layer using 64 Fitlers of 3x3 size, with stride of 1 & padding of 0
        self.conv2 = nn.Conv2d(32, 64, 3)
        # Our Max Pool Layer 2 x 2 kernel of stride 2
        self.pool = nn.MaxPool2d(2, 2)
        # Our first Fully Connected Layer (called Linear), takes the output of our Max Pool
        # which is 12 x 12 x 64 and connects it to a set of 128 nodes
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        # Our second Fully Connected Layer, connects the 128 nodes to 10 output nodes (our classes)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # here we define our forward propogation sequence 
        # Remember it's Conv1 - Relu - Conv2 - Relu - Max Pool - Flatten - FC1 - FC2
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the model and move it (memory and operations) to the CUDA device
net = Net()
net.to(device)

# We import our optimizer function
import torch.optim as optim

# We use Cross Entropy Loss as our loss function
criterion = nn.CrossEntropyLoss()

# For our gradient descent algorthim or Optimizer
# We use Stochastic Gradient Descent (SGD) with a learning rate of 0.001
# We set the momentum to be 0.9
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#We loop over the traing dataset multiple times (each time is called an epoch)
# epochs = 10

# # Create some empty arrays to store logs 
# epoch_log = []
# loss_log = []
# accuracy_log = []

# # Iterate for a specified number of epochs
# for epoch in range(epochs):  
#     print(f'Starting Epoch: {epoch+1}...')

#     # We keep adding or accumulating our loss after each mini-batch in running_loss
#     running_loss = 0.0

#     # We iterate through our trainloader iterator
#     # Each cycle is a minibatch
#     for i, data in enumerate(trainloader, 0):
#     #enumerate(trainloader, 0) - The enumerate function is used to iterate over an iterable (in this case, trainloader)
#     #along with an index. The second argument (in this case, 0) specifies the starting index. The purpose of using this index is usually to keep track of the current iteration's count.

#     # data here is a list of [inputs and labels]
#       inputs, labels=data

#     #here we are passing our labels and inputs to the GPU/cpu

#       inputs=inputs.to(device)
#       labels=labels.to(device)

#     # we set the gradient to zero for the inintial start of the loop
#       optimizer.zero_grad()

#     # Then we forwawrd pass -->back propogate and then optimize
#     # Inputs (which represent a batch of input data) are passed through the neural network model (net) to compute the predictions (outputs). This is the forward pass
#       outputs=net(inputs)
#     # compare outputs with the actual labels (ground truth) using a loss function (criterion). The loss function quantifies how well the model's predictions match the true labels.
#       loss=criterion(outputs,labels)
#     ## then we backpropogate
#       loss.backward()
#     # Then we update them using optimizers.step()
#       optimizer.step()

#     #item(): The .item() method is used to extract a scalar value from a tensor. Since loss is likely a tensor representing the loss value, .item() is used to obtain the numeric value of the loss.
#       running_loss += loss.item()

#       if i % 50 ==49:
#        ## show our loss every 50 mini batches
#        ## initialize a variable to hold correct value and total value
#          correct=0
#          total=0

#       # disablse the tracking of any calculations required to later calculate the gradient
#          with torch.no_grad():

#           for data in testloader:

#             images, labels=data

#             images=images.to(device)
#             labels=labels.to(device)

#           # forward propogate
#             outputs=net(images)
#           #the line of code takes the model's predictions (outputs), finds the class indices with the highest values
#             _, predicted=torch.max(outputs.data,dim=1) #outputs.data: The .data attribute is used to access the tensor without any gradients.

#           #_ and predicted: The result of torch.max is a tuple containing two tensors: the maximum values and their indices. In this case, you are interested in the indices,
#           # which represent the predicted class labels.
#           #_: The underscore _ is used as a convention to discard the value that is assigned to it. Since you are only interested in the predicted tensor, the _ is used to ignore the maximum values.
#           #predicted: This variable holds the tensor containing the predicted class indices for each sample in the batch.

#           # then we add label size to total
#             total += labels.size(0)

#           # keep a running total of the number of predictions predicted correctly

#             correct += (predicted == labels).sum().item()
#           accuracy= 100 * correct/total
#           epoch_num=epoch + 1
#           actual_loss=running_loss/50
#           print(f'Epoch: {epoch_num},Mini-Batches Completed: {(i+1)},Loss:{actual_loss:.3f} , Test Accuracy = {accuracy:.3f}%' )
#           running_loss=0.0

#     epoch_log.append(epoch_num)
#     loss_log.append(loss)
#     accuracy_log.append(accuracy)

# print("Training Finished")  

PATH='./mnist_cnn_torch.pth'
torch.save(net.state_dict(),PATH)

## reloading the model

net=Net()
net.to(device)

net.load_state_dict(torch.load(PATH))
