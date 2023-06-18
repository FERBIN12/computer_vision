



# ## for basics the class notes is available in the google colab account docs
#   # ie the usage of aspect ratio , imshow numpy , and matplotlib

# import cv2
# import numpy as np                                                      
# from matplotlib import pyplot as plt                                    # for creating low level graphs

# def imshow(title="Image",image=None,size=10):
#   w,h=image.shape[0],image.shape[1]
#   aspect_ratio=w/h
#   plt.figure(figsize=(size*aspect_ratio,size))
#   plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
#   plt.title(title)
#   plt.show()

#   # putting text in a image




# # creating black image using RGB as well as greyscale

# # image1=np.zeros((512,512,3),np.uint8)   # rgb takes more memory space 
# # image2=np.zeros((512,512),np.uint8)

# # imshow("Black canvas-RGB",image1)
# # imshow("Black canvas-Greyscale",image2)

# # line in our image

# # cv2.line(image,starting coordinates,ending,color,thickness)

# # cv2.line(image1,(0,0),(511,511),(255,127,0),5)
# # imshow("Black pic with blue line",image1)


# # for drawing rectangle in a image

# # for cv2.rectangle(image,starting,ending_co-ordinates,color,thickness)

# # cv2.rectangle(image1,(100,100),(300,300),(127,50,127),5)
# # imshow("Black canvas with pink rectangle",image1)


# # for drawing circle
# # image1=np.zeros((512,512,3),np.uint8)
# #                                                     # cv2.circle(image,center,radius,color,fill)
# # cv2.circle(image1,(350,350),100,(15,150,50), 12)   # for complete fill we use -1 and 1 for line
# # imshow("Black canvas with green circle",image1)


# # for Writing text inside a image file
# # image=np.zeros((1000,1000,3),np.uint8)
# # ourstr="Thank You Jesus"

# # # cv2.putText(image,text,bottom left point,font,font size,color,thickness)
# # cv2.putText(image,ourstr,(200,100),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,3,(40,200,7),3)
# # imshow("Messing with the images",image)

# ## transformation

# # image=cv2.imread('images/Volleyball.jpeg')
# # imshow("Blah",image)                # loading image

# # # store the height and width of the image

# # height,width=image.shape[:2]

# # # or height,width =image.shape[0],image.width[1]

# # changed_h,changed_w=height/4,width/4

# # # Translation matrix

# # T=np.float32([[1, 0,changed_w],[0, 1,changed_h]])

# # # cv2.warpAffine is used to transform images

# # image_T=cv2.warpAffine(image,T,(width,height))
# # imshow("2000 decades later ",image_T)

# # # cv2.getRotationMatrix2D((centre_point),angle of rottaion,scale)

# # rotation_image=cv2.getRotationMatrix2D((width/2,height/2),90,0.5)
# # rotated_image=cv2.warpAffine(image,rotation_image,(width,height))
# # imshow("Rotated",rotated_image)


# # rotated_image2=cv2.transpose(image)
# # imshow("Rotated_uisng_transpose",rotated_image2)

# # # horizontal_fix

# # flipped=cv2.flip(image,1)
# # imshow("Hori_flipped",flipped)

# # If you're wondering why only two dimensions, well this is a grayscale image, 

# # Making a square
# square = np.zeros((300, 300), np.uint8)
# cv2.rectangle(square, (50, 50), (250, 250), 255, -2)
# imshow("square", square)

# # Making a ellipse
# ellipse = np.zeros((300, 300), np.uint8)
# cv2.ellipse(ellipse, (150, 150), (150, 150), 30, 0, 180, 255, -1)
# imshow("ellipse", ellipse)



import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')

def face_detector(img, size=0.5):
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img
    
    for (x,y,w,h) in faces:
        x = x - 50
        w = w + 50
        y = y - 50
        h = h + 50
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray)
        
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2) 
            
    roi_color = cv2.flip(roi_color,1)
    return roi_color

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    cv2.imshow('Our Face Extractor', face_detector(frame))
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()      