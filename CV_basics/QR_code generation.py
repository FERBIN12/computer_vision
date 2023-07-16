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

from pyzbar.pyzbar import decode
from PIL import Image

img=Image.open('qrcode.png')
result=decode(img)
for i in result:
  print(i.data.decode("utf-8"))

from pyzbar.pyzbar import decode

image=cv2.imread("/home/kanja-koduki/Downloads/1DwED.jpg")
# imshow("The image is ",image)
codes=decode(image)

for bc in codes:
    (x,y,w,h) = bc.rect
    print(bc.polygon)
    pt1,pt2,pt3,pt4=bc.polygon

    pts = np.array( [[pt1.x,pt1.y], [pt2.x,pt2.y], [pt3.x,pt3.y], [pt4.x,pt4.y]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(image, [pts], True, (0,0,255), 3)

    barcode_text=bc.data.decode()
    barcode_type=bc.type

    text = "{} ({})".format(barcode_text, barcode_type)
    cv2.putText(image, barcode_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.putText(image, barcode_type, (x+w, y+h - 15), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    print("QR Code revealed: {}".format(text))

# display our output
imshow("QR Scanner", image, size = 12)


