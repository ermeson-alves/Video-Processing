import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


# path do video
video_path = 'videos\part2(Ermeson).mp4'
video = cv2.VideoCapture(video_path)


def view_colors_channels(video):
	"""Com base em um objeto cv2.VideoCapture, exibe diferentes canais de cores
	pra cada frame desse objeto"""
	while True:
		ret, frame = video.read()

		if not ret:
			print("Não foi possivel ler o frame. Saindo...")
			break 

		b, g, r = cv2.split(frame)
		h, s, v = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))


		function = lambda name, img: cv2.imshow(name, cv2.resize(img, (600, 350)))
		function("Blue",       b)
		function("Green",      g)
		function("Red",        b)
		function("Hue",        h)
		function("Saturation", s)
		function("Value",      v)

		cv2.moveWindow("Blue",       50,   0)
		cv2.moveWindow("Green",      692,  0)
		cv2.moveWindow("Red",        1334, 0)
		cv2.moveWindow("Hue",        50,   500)
		cv2.moveWindow("Saturation", 692,  500)
		cv2.moveWindow("Value",      1334, 500)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# Liberar janelas
	video.release()
	cv2.destroyAllWindows()
	

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

# ds = 0
# cont = 0

def calc_velocity(video):
	cont = 0
	
	while True:
		# calcular a distancia q o navio se moveu a cada k frames
		ret, frame = video.read()

		if not ret:
			print("Não foi possivel ler o frame. Saindo...")
			break

		# pre-processamento
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
		h, s, v = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
		ret, h = cv2.threshold(h, 160, 255, cv2.THRESH_BINARY)
		h = cv2.dilate(h, kernel, iterations=2)

		pixels_non_zero = cv2.findNonZero(h)
		coord_y = np.max(pixels_non_zero, axis=0)[0,1]

		# configurações da linha
		ponto_inicial = (0, coord_y) 
		ponto_final = (h.shape[1], coord_y)
		cor_linha = 255
		espessura = 1
		cv2.line(h, ponto_inicial, ponto_final, cor_linha, espessura)



		if cont == 0:
			# deslocamento inicial
			ds_i = coord_y 
			ds = ds_i

		# if cont %  == 0: # se passaram 5 frames
		ds = coord_y - ds # calcula deslocamento
		if ds == ds_i:
			ds = 0
		
		print(f"Coordenada y: {coord_y}, DS: {ds}, DS_i: {ds_i}")

		cv2.imshow("Video", h)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		cont+=1



	video.release()
	cv2.destroyAllWindows()


# view_colors_channels(video)
calc_velocity(video)