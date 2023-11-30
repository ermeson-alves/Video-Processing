import cv2
import matplotlib.pyplot as plt
import numpy as np


# path do video
video_path = './videos/part2(Ermeson).mp4'
video = cv2.VideoCapture(video_path)

ret = True

while ret:

    ret, frame = video.read() 
    if ret == True: 
        # Pre-processing
        # r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        # plt.hist(r.ravel(), 256, [0, 256])
        # plt.show()

        # Display the resulting frame 
        cv2.imshow('Frame', frame) 
          
        # Press Q on keyboard to exit 
        if cv2.waitKey(15) & 0xFF == ord('q'): 
            break


video.release()
cv2.destroyAllWindows()
