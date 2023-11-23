import cv2
import matplotlib.pyplot as plt
import numpy as np


# path do video
video_path = './videos/part2(Ermeson).mp4'
video = cv2.VideoCapture(video_path)



# while ret:
#     ret, frame = video.read()

#     if ret:
#         cv2.namedWindow('Nome da Janela', cv2.WINDOW_NORMAL)
#         cv2.imshow('Nome da Janela', frame)
#         if cv2.waitKey(0) & 0xFF == ord('q'):
#             break

img = cv2.imread('videos/img.jpg')
# ret, frame = video.read()
# print("Ret:", ret)
plt.imshow(img)

plt.show()