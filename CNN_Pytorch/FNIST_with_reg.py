import torch
import PIL
import numpy as np

import torchvision
import torchvision.transforms as transforms

import torch.optim as optim
import torch.nn as nn

print(f"GPU available: {torch.cuda.is_available()}")

device='cuda'

# performing data augmentation
# note that the augmentationns hshould be applied only to the train set and not the test so we 
# we set the transforms seperately for them in the transforms list
data_transforms={
    'train' :transforms.Compose([
        transforms.RandomAffine(degrees=10,translate=(0.05,0.05),shear=5),
        transforms.ColorJitter(hue=0.05,saturation=0.05),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15,PIL.Image.BILINEAR),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ),(0.5, ))
        ]),
    'test' : transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ),(0.5, ))
    ])
}

# fetch our train and test set and data loader

trainset=torchvision.datasets.FashionMNIST(root='./data',train=True,
                                           download=True,transform=data_transforms['train'])
testset=torchvision.datasets.FashionMNIST(root='./data',train=False,
                                          download=True,transform=data_transforms['test'])

trainloader=torch.utils.data.DataLoader(trainset,batch_size=32,
                                              shuffle=True,num_workers=2)
testloader=torch.utils.data.DataLoader(testset,batch_size=32,
                                             shuffle=False,num_workers=2)   

# now we look out how to add dropout and batch norm in our model
import torch.nn as nn 
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.Conv1=nn.Conv2d(1,32,3)
    # next we set the batch normlization to this layer and give the output of this layer in it
    self.Conv1_bn=nn.BatchNorm2d(32)
    self.Conv2=nn.Conv2d(32,64,3)
    # now we set the second conv layer 
    self.Conv2_bn=nn.BatchNorm2d(64)
    self.pool=nn.MaxPool2d(2,2) # we set the kernel size and stride for pool layer
    self.fc1=nn.Linear(64*12*12,128)
    self.fc2=nn.Linear(128,10)
    # now we set the dropout for the model
    # in the dropout the parameter 0.2 refers to 20% of the model being dropped
    self.dropout=nn.Dropout(0.2)

  def forward(self,x):
    x=F.relu(self.Conv1_bn(self.Conv1(x)))
    x=self.dropout(x)
    x=self.dropout(F.relu(self.Conv2_bn(self.Conv2(x))))
    x=self.pool(x)
    x=x.view(-1,64*12*12)
    x=F.relu(self.fc1(x))
    x=self.fc2(x)

    return x

net=Net()
net.to(device)

# adding the loss function and optimizer 
criterion=nn.CrossEntropyLoss()

optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9,weight_decay=0.001)

# now here we go the training of the model
epochs=15

# set the empty logs for storing the accuracy,loss,epoch logs

epoch_log=[]
accuracy_log=[]
loss_log=[]

# iterate through the epochs
for epoch in range(epochs):
  print(f'starting epochs :{epoch+1}.....')
  running_loss=0.0
    # we set the running loss to be 0 in every loop  
  for i,data in enumerate(trainloader,0):

    inputs,labels=data

    inputs=inputs.to(device)
    labels=labels.to(device)

    ## we clear the optimizer to 0 grad to start fresh
    optimizer.zero_grad()

    # now forward propogate get the losses , back prop and update the valuess
    outputs=net(inputs)
    loss=criterion(outputs,labels)
    loss.backward()
    optimizer.step() # updating the weights

    running_loss += loss.item()
    #item() is used to extract the loss value as a scalar Python number from the tensor.

    if i % 100 == 90:
      correct=0 # in this condition the model loops over 100 batches and so now we are
      total=0   # setting the correct-predtions and total - labels to 0 for next loop

      # Now  we validate the model usinf test data
      # we do them with no grads for data saving purpose 

      with torch.no_grad():
        for data in testloader:

          images,labels=data
          images=images.to(device)
          labels=labels.to(device)

          # now we forward propogate 
          outputs=net(images)

          _,predicted=torch.max(outputs.data,1)
          # dimension 1. This index corresponds to the predicted class label for each sample.

          total += labels.size(0)

          correct +=(predicted == labels).sum().item()
          # sum() sums the correctly predicted samples and item converts them to scalar

      accuracy=100 *correct/total
      epoch_num = epoch+1
      actual_loss=running_loss/100
      print(f'Epoch:{epoch_num} Mini-Batches completed:{i+1},Loss:{actual_loss:.3f},Test accuracy:{accuracy:.3f}%')
      running_loss=0.0

  epoch_log.append(epoch_num)
  loss_log.append(actual_loss)
  accuracy_log.append(accuracy)

print('FINISHED training booyah')


# we plot our training data

import matplotlib.pyplot as plt

# to create a plot with secondary y axis we need to create a subplot
fig, ax1=plt.subplots()

# set the title for our plot and 
plt.title('Accuracy & loss vs Mini-Batches')

# we use twinx function to create a secondary plot
ax2=ax1.twinx()

#we create the plots for comparing the loss and accuracy
ax1.plot(epoch_log,loss_log,'g-')
ax2.plot(epoch_log,accuracy_log,'b-')

# set the labels
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss',color='g')
ax2.set_ylabel('Test accuracy',color='b')

          
        