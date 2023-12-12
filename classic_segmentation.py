import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


# path do video
video_path = 'videos\part2(Ermeson).mp4'
video = cv2.VideoCapture(video_path)



# ret1, frame_1 = video.read()
# ret2, frame_2 = video.read()


# kernel	=	cv2.getStructuringElement(
# 		cv2.MORPH_ELLIPSE,	(5,5)
# )



# b,g,r = cv2.split(frame_1)
# h, s, v = cv2.split(cv2.cvtColor(frame_1, cv2.COLOR_BGR2HSV))
# h2, s2, v2 = cv2.split(cv2.cvtColor(frame_2, cv2.COLOR_BGR2HSV))

# ret, h = cv2.threshold(h, 160, 255, cv2.THRESH_BINARY)
# ret, h2 = cv2.threshold(h2, 160, 255, cv2.THRESH_BINARY)


# h = cv2.dilate(h, kernel, iterations=2)
# h2 = cv2.dilate(h2, kernel, iterations=2)


# plt.subplot(331); plt.imshow(b, cmap='gray'); plt.title('B')
# plt.subplot(332); plt.imshow(g, cmap='gray'); plt.title('G')
# plt.subplot(333); plt.imshow(r, cmap='gray'); plt.title('R')
# plt.subplot(334); plt.imshow(h, cmap='gray'); plt.title('H')
# plt.subplot(335); plt.imshow(s, cmap='gray'); plt.title('S')
# plt.subplot(336); plt.imshow(v, cmap='gray'); plt.title('V')
# plt.subplot(337); plt.imshow(h, cmap='gray'); plt.title('H1')
# plt.subplot(338); plt.imshow(h2, cmap='gray'); plt.title('H2')
# plt.subplot(339); plt.imshow(h2 - h, cmap='gray'); plt.title('H2 - H1')

# # plt.imshow(frame_2 - frame_1)
# plt.show()


cor_pixel = 255

cont = 0
while True:
	# calcular a distancia q o navio se moveu a cada 1 frame. Compara o frame n+1 com o n
	ret, frame = video.read()

	if not ret:
		print("Não foi possivel ler o frame. Saindo...")
		break
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
	h, s, v = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
	ret, h = cv2.threshold(h, 160, 255, cv2.THRESH_BINARY)
	h = cv2.dilate(h, kernel, iterations=2)


	pixels_non_zero = cv2.findNonZero(h)
	coord_y = np.max(pixels_non_zero, axis=0)[0,1]

	ponto_inicial = (0, coord_y) 
	ponto_final = (h.shape[1], coord_y)
	cor_linha = 255
	espessura = 2
	cv2.line(h, ponto_inicial, ponto_final, cor_linha, espessura)

	cv2.imshow("Video", h)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



video.release()
cv2.destroyAllWindows()


