import numpy as np
import time
import cv2
import os
from os import listdir   # used to get list of files in a directory
from os.path import isfile,join  # is file is used to check if the files exist
from matplotlib import pyplot as plt   # join 

def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


## load the coco class labels our YOLO model was trained on "Trained model"

labelspath="/home/kanja-koduki/Downloads/YOLO/YOLO/yolo/coco.names"
# Reads the content of the label file, removes leading/trailing whitespaces, and splits the lines into a list of labels.
LABELS=open(labelspath).read().strip().split("\n")

## now we set random colors for the labels to identify them
COLORS=np.random.randint(0,255,size=(len(LABELS),3),dtype="uint8")

weights_path="/home/kanja-koduki/Downloads/YOLO/YOLO/yolo/yolov3.weights"
cfg_path="/home/kanja-koduki/Downloads/YOLO/YOLO/yolo/yolov3.cfg"

# Loads the Yolo model from cfg and weight files
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

# Sets the backend for the neural network to OpenCV.
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

print("Our YOLO Layers")
ln = net.getLayerNames() ## retrives the names of the layers in the neural network

# There are 254 Layers
print(len(ln), ln)

print("Starting Detections...")

## set the path file in which we can find the images to be processed
mypath="/home/kanja-koduki/Downloads/YOLO/YOLO/images"

# creates an list using list comprehension and using join and isfile method 
file_names=[f for f in list(mypath) if isfile(join(mypath,f))] 
# is file checks if we have a file / not and join function is used to join the file 

for file in file_names: # iterating over each file in the list 
    

    image=cv2.imread(mypath+file)
    (H,W)=image.shape[:2]

    # we want only the *output * layer names that we need from YOLO

    ln=net.getLayerNames() # retrives the namse of all the layers
    ln=[ln[i[0] -1] for i in net.getUnconnectedOutlayers()] # retrives the names of the output layers

    #Now we construct our blob from ip image

    blob=cv2.dnn.blobFromImage(image,1/255,(416,416),swapRB=True,crop=False)
    #Creates a blob from the image. A blob is a preprocessed image that can be fed into
    #the neural network for inference. The function scales the pixel values, resizes the 
    # image to (416, 416), swaps the Red and Blue channels, and disables cropping.

    # we set our input to our image blob
    net.setInput(blob)

    #then we run a forward pass through the network to obtain the output of specified layer
    layerOutputs=net.forward(ln)

    # we initialize our lists
    boxes=[]  ## to store bounding boxes coordinates
    confidences=[] # confidence score of the detection 
    IDs=[] # class IDs of the detection

    # inintially we iterate over the outpputs 
    for output in layerOutputs:

        # Loop over each detection
        for detection in output:
            # Obtain class ID and probality of detection
            scores = detection[5:] # confidence score of each detection
            classID = np.argmax(scores) # Determines the class ID with the highest score
            # by finding the index of the maximum value in the scores array using np.argmax.
            confidence = scores[classID]

            # We keep only the most probably predictions
            if confidence > 0.75:
                # We scale the bounding box coordinates relative to the image
                # Note: YOLO actually returns the center (x, y) of the bounding
                # box followed by the width and height of the box
                box = detection[0:4] * np.array([W, H, W, H])
                # Extracts the center coordinates, width, and 
                # height of the bounding box and converts them to integers.
                (centerX, centerY, width, height) = box.astype("int")

                # Get the the top and left corner of the bounding box
                # Remember we alredy have the width and height
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Append our list of bounding box coordinates, confidences and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                IDs.append(classID)

    # Applies non-maxima suppression to the detected bounding boxes to remove
    # overlapping boxes. The function returns the indices of the boxes that are kept.
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # We proceed once a detection has been found
    if len(idxs) > 0:
        # iterate over the indexes we are keeping
        for i in idxs.flatten():
            #the two-dimensional array is flattened into a one-dimensional array. This is done
            #  because the subsequent code expects a one-dimensional array of indices to iterate over.
            # Get the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1]) # top left corner
            (w, h) = (boxes[i][2], boxes[i][3]) # width and height

            #  Retrieves the color assigned to the class ID and converts it to a list of integers.
            color = [int(c) for c in COLORS[IDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
            # Creates a text string with the class label and confidence score.
            text = "{}: {:.4f}".format(LABELS[IDs[i]], confidences[i])
            # cv2.putText(text, confidence) puts the text in the imageg
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # show the output image
    imshow("YOLO Detections", image, size = 12)












