## downloading the requirements
# !git clone https://github.com/timesler/facenet-pytorch.git
# !pip install facenet-pytorch

from facenet_pytorch import MTCNN,InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os

workers = 0 if os.name == 'nt' else 4 # nt refers to windows so if the os is windows it uses 0 workers and 4 for others

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}')

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device)
resnet=InceptionResnetV1(pretrained='vggface2').eval().to(device)

#define the dataset and the dataloader

def collate_fn(x):
  return x[0]

dataset=datasets.ImageFolder('facenet-pytorch/data/test_images')
dataset.idx_to_class={ i:c for c,i in dataset.class_to_idx.items()}   # is used for providing labels for the classes 
loader=DataLoader(dataset,collate_fn=collate_fn,num_workers=workers)

# now we print the images
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

def imgshow(title=" ",image=None,size=6):
  if image.any():
    w,h=image.shape[0:2]
    aspect_ratio=w/h
    plt.figure(figsize=(size*aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
  else:
    print('Image not found')
for f in glob.glob('./facenet-pytorch/data/test_images/**/*.jpg',recursive=True):
  image=cv2.imread(f)
  imgshow(f,image) 

# Perform the MTCNN
names = []
aligned = []

for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        print(f'Face detected with probability: {prob:.8f}')
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])

aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()

dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
pd.DataFrame(dists,columns=names,index=names)
