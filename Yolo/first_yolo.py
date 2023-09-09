## my first yolo pretrained

# here we go installing the required libraries and clone the needed repo
from IPython.display import Image

Image('runs/detect/exp3/bus.jpg')  # image lib can be used for running the image library

from IPython.display import HTML
from base64 import b64encode
import os

# Input video path
save_path = '/home/kanja-koduki/computer_vision/Yolo/yolov7/Video2.mp4'

# Compressed video path
compressed_path = '/yolov7/Video2.mp4'

os.system(f"ffmpeg -i {save_path} -vcodec libx264 {compressed_path}")

# Show video
mp4 = open(compressed_path,'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=400 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)