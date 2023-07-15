# Our Setup, Import Libaries, Create our Imshow Function and Download our Images


import cv2
import numpy as np
from matplotlib import pyplot as plt

# Define our imshow function 
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

import qrcode
from PIL import Image

qr=qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=4
)

qr.add_data("https://www.instagram.com/ferbin._12/")
qr.make(fit=True)
img=qr.make_image(fill_color="black",back_color="white")
img.save("qrcode.png")

qrcode=cv2.imread("qrcode.png")
imshow("QR Code",qrcode,size=8)