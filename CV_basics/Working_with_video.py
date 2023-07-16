import cv2
# import numpy as np

# cap=cv2.VideoCapture(0)

# while True:
#     ret,frame=cap.read()

#     cv2.imshow('Our live video ',frame)

#     if cv2.waitKey(1) == 13:
#         break 

# cap.release()
# cap.destroyAllWindows()

# def sketch(image):
#     img_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#     img_gray_blur=cv2.GaussianBlur(img_gray,(5,5),0)

#     # extarct the edges using the canny edge
#     cannny_edges=cv2.Canny(img_gray_blur,10,70)

#     ret, mask=cv2.threshold(cannny_edges,70,255,cv2.THRESH_BINARY_INV)

#     return mask

# cap=cv2.VideoCapture(0)

# while True:
#     ret, frame=cap.read()
#     cv2.imshow('Our Live Skecther ',sketch(frame))
#     if cv2.waitKey(1) == 13:
#         break
# cap.release()
# cap.destroyAllWindows()


cap=cv2.VideoCapture('/home/kanja-koduki/computer_vision/Working with Video/videos/drummer.mp4')

while cap.isOpened():
    
    ret,frame=cap.read()

    if not ret:
        print("Video stopped")
        break
    cv2.imshow('Video',frame)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cap.destroyAllWindows()

    

