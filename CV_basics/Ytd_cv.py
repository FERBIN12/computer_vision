# import cv2
# import pafy

# url = 'https://youtu.be/EFEmTsfFL5A'
# video = pafy.new(url)

# best = video.getbest(preftype="mp4")

# capture = cv2.VideoCapture()
# capture.open(best.url)

# while (True):
#     ret, frame = capture.read()
#     if ret == True:
#         cv2.imshow('src', frame)
          
#     if cv2.waitKey(1) == 13: #13 is the Enter Key
#         break
          
#   # Release camera and close windows
# capture.release()
# cv2.destroyAllWindows()  


import pafy

url = 'https://youtu.be/EFEmTsfFL5A'
video = pafy.new(url)

print("Title: {}".format(video.title))
print("Rating: {}".format(video.rating))
print("Viewcount: {}".format(video.viewcount))
print("Author: {}".format(video.author))
print("Length: {}".format(video.length))
print("Duration: {}".format(video.duration))