import cv2
import matplotlib.pyplot as plt
import numpy as np
from preprocessing import canny


# path do video
video_path = './videos/part2(Ermeson).mp4'
video = cv2.VideoCapture(video_path)


# while True:
#     ret, frame = video.read() 
#     if not ret:
#         break 
          
#     # Show:
#     cv2.imshow('Video', frame)
#     # Press Q on keyboard to exit 
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
ret1, frame_1 = video.read()
ret2, frame_2 = video.read()

c_frame_1 = canny(frame_1)
c_frame_2 = canny(frame_2)

# plt.subplot(141); plt.imshow(frame_1-frame_2, cmap='gray'); plt.title('Frame 1 - Frame 2')
# plt.subplot(142); plt.imshow(c_frame_1-frame_2, cmap='gray'); plt.title('CANNY(Frame 1) - Frame 2')
# plt.subplot(143); plt.imshow(frame_1-c_frame_2, cmap='gray'); plt.title('Frame 1 - CANNY(Frame 2)')
# plt.subplot(144); plt.imshow(c_frame_1-c_frame_2, cmap='gray'); plt.title('CANNY(Frame 1) - CANNY(Frame 2)')

# plt.figure(figsize=(20, 10))
# plt.subplot(141); plt.imshow(frame_1[..., 0], cmap='gray'); plt.title('Canal azul')
# plt.subplot(142); plt.imshow(frame_1[..., 2], cmap='gray'); plt.title('Canal vermelho')
# plt.subplot(143); plt.imshow(frame_1[..., 0] - frame_1[..., 2], cmap='gray'); plt.title('Azul - Vermelho')


elementoEstruturante	=	cv2.getStructuringElement(
		cv2.MORPH_ELLIPSE,	(5,5)
)
imagemProcessada	=	cv2.dilate (
		frame_2- frame_1,	elementoEstruturante,	iterations	=	2
)


b,g,r = cv2.split(frame_1)
h, s, v = cv2.split(cv2.cvtColor(frame_1, cv2.COLOR_BGR2HSV))

plt.subplot(231); plt.imshow(b, cmap='gray'); plt.title('B')
plt.subplot(232); plt.imshow(g, cmap='gray'); plt.title('G')
plt.subplot(233); plt.imshow(r, cmap='gray'); plt.title('R')
plt.subplot(234); plt.imshow(h, cmap='gray'); plt.title('H')
plt.subplot(235); plt.imshow(s, cmap='gray'); plt.title('S')
plt.subplot(236); plt.imshow(v, cmap='gray'); plt.title('V')

plt.show()
video.release()
cv2.destroyAllWindows()


