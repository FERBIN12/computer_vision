import numpy as np
import cv2
from mss import mss
from PIL import Image
import time

bounding_box = {'top': 100, 'left': 0, 'width': 400, 'height': 300}

sct = mss()
frame_count = 0 
last_time = time.time()

while True:
    frame_count += 1
    sct_img = sct.grab(bounding_box)
    cv2.imshow('screen', np.array(sct_img))

    # Display FPS rate
    if frame_count % 30 == 0:
        FPS = 1.0 / (time.time()-last_time)
        print('FPS = {}'.format(FPS))
    last_time = time.time()    

    if cv2.waitKey(1) == 13: #13 is the Enter Key
        print("Exited...")
        break
        
cv2.destroyAllWindows()