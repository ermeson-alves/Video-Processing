import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from preprocessing import *
import requests
import json

# path do video
video_path = 'videos\part3(Luis Carlos).mp4'
video = cv2.VideoCapture(video_path)
# informações do video:
fps = int(video.get(cv2.CAP_PROP_FPS))
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))



def view_colors_channels(video, plot: bool = False):
	"""Com base em um objeto cv2.VideoCapture, exibe diferentes canais de cores
	pra cada frame desse objeto
	
	Se plot = True, quando o video for interrompido uma imagem representando o
	frame atual é construída.
	"""
	while True:
		ret, frame = video.read()

		if not ret:
			print("Não foi possivel ler o frame. Saindo...")
			break 

		b, g, r = cv2.split(frame)
		h, s, v = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
		l, a, b_lab = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))


		function = lambda name, img: cv2.imshow(name, cv2.resize(img, (600, 400)))
		function("Blue",       b)
		function("Green",      g)
		function("Red",        b)
		function("Hue",        h)
		function("Saturation", s)
		function("Value",      v)

		cv2.moveWindow("Blue",       30,   0)
		cv2.moveWindow("Green",      672,  0)
		cv2.moveWindow("Red",        1314, 0)
		cv2.moveWindow("Hue",        30,   480)
		cv2.moveWindow("Saturation", 672,  480)
		cv2.moveWindow("Value",      1314, 480)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# Liberar janelas
	video.release()
	cv2.destroyAllWindows()

	if plot:
		num_lin = 2
		num_col = 3
		plt.subplot(num_lin, num_col, 1); plt.imshow(b, cmap='gray'); plt.title('B')
		plt.subplot(num_lin, num_col, 2); plt.imshow(g, cmap='gray'); plt.title('G')
		plt.subplot(num_lin, num_col, 3); plt.imshow(r, cmap='gray'); plt.title('R')
		plt.subplot(num_lin, num_col, 4); plt.imshow(h, cmap='gray'); plt.title('H')
		plt.subplot(num_lin, num_col, 5); plt.imshow(s, cmap='gray'); plt.title('S')
		plt.subplot(num_lin, num_col, 6); plt.imshow(v, cmap='gray'); plt.title('V')
		plt.show()



def calc_velocity_video(video:cv2.VideoCapture, scale:float = 0.32, period:int =10):
	"""Realiza o processo de segmentação do navio e calculo da velocidade a cada
	period segundos. Também faz o envio da posição e velocidade atual para a
	interface.
	
	Args:
		video: Objeto que representa o video dentro do contexto da opencv.
		scale: Quantidade equivalente de metros para um pixel.
		period: Diz a cada quantos segundos a velocidade será calculada.
	
	"""

	flask_url = 'http://localhost:8000/dados'
	cont = 0
	ds = 0


	while True:
		ret, frame = video.read()

		if not ret:
			print("Não foi possivel ler o frame. Saindo...")
			break

		# pre-processamento
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
		h, s, v = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
		ret_th_otsu, h_th_otsu = cv2.threshold(h, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
		dil_th = cv2.dilate(h_th_otsu, kernel, iterations=2)

		pixels_non_zero = cv2.findNonZero(dil_th)
		coord_y = np.max(pixels_non_zero, axis=0)[0,1]
		coord_y = int(coord_y)

		# configurações da linha
		ponto_inicial = (0, coord_y) 
		ponto_final = (h.shape[1], coord_y)
		cor_linha = 255
		espessura = 1
		cv2.line(h_th_otsu, ponto_inicial, ponto_final, cor_linha, espessura)

		# deslocamento inicial
		if cont == 0:
			coord_y_i = coord_y
		# caso tenha passado uma quantidade de frames equivalentes a 15s de video:
		if cont % (fps*period) == 0:
			ds = (coord_y - coord_y_i) - ds
			speed = (ds*0.32)/period
			
			print(f"{(cont/(fps*period))*period}s de video | Velocidade: {speed:4f}m/s")

			 # Enviar dados para o servidor Flask
			print(type(coord_y), type(speed))
			data = {'present-position': coord_y, 'speed': speed}
			headers = {'Content-Type': 'application/json'}
			try:
				response = requests.post(flask_url, data=json.dumps(data), headers=headers)
				response.raise_for_status()
				print('Dados enviados com sucesso!')
			except requests.exceptions.RequestException as e:
				print(f'Erro ao enviar dados para o servidor Flask: {e}')

		
		# print(f"Coordenada y: {coord_y}, DS: {ds}, DS_i: {coord_y_i}")

		cv2.imshow("Video", h_th_otsu)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		cont+=1
	
	
	video.release()
	cv2.destroyAllWindows()



def calc_velocity_frames(img1, img2):
	"""Faz o processamento de duas imagens, que podem ser dois frames
	do video em questão."""

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
	# pre-processamento frame 1
	h, s, v = cv2.split(cv2.cvtColor(img1, cv2.COLOR_BGR2HSV))
	ret_th_manual,   h_th_manual   = cv2.threshold(h, 160, 255, cv2.THRESH_BINARY)
	ret_th_otsu,     h_th_otsu     = cv2.threshold(h, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	h_th_adaptive                  = cv2.adaptiveThreshold(h, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 15)

	dil_h 			  = cv2.dilate(h, kernel, iterations=2)
	dil_h_th_manual   = cv2.dilate(h_th_manual, kernel, iterations=2)
	dil_h_th_otsu     = cv2.dilate(h_th_otsu, kernel, iterations=2)
	dil_h_th_adaptive = cv2.dilate(h_th_adaptive, kernel, iterations=2)

	# pre-processamento frame 2
	h2, s2, v2 = cv2.split(cv2.cvtColor(img2, cv2.COLOR_BGR2HSV))
	ret2_th_manual, h2_th_manual = cv2.threshold(h2, 160, 255, cv2.THRESH_BINARY)
	ret2_th_otsu, h2_th_otsu = cv2.threshold(h2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	h2_th_adaptive = cv2.adaptiveThreshold(h2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 15)

	dil_h2 			  = cv2.dilate(h, kernel, iterations=2)
	dil_h2_th_manual   = cv2.dilate(h_th_manual, kernel, iterations=2)
	dil_h2_th_otsu     = cv2.dilate(h_th_otsu, kernel, iterations=2)
	dil_h2_th_adaptive = cv2.dilate(h_th_adaptive, kernel, iterations=2)

	plt.subplot(241); plt.imshow(h,                 cmap='gray'); plt.title('H img1')
	plt.subplot(242); plt.imshow(h_th_manual,       cmap='gray'); plt.title('Limiarização Manual (>160)\nTHRESH_BINNARY img1')
	plt.subplot(243); plt.imshow(h_th_otsu,         cmap='gray'); plt.title('Limiarização de Otsu\nTHRESH_BINNARY_INV\nTHRESH_OTSU img1')
	plt.subplot(244); plt.imshow(h_th_adaptive,     cmap='gray'); plt.title('Limiarização Adaptativa\nTHRESH_BINNARY_INV\nMEAN\ntam_mask = 5\ncte_subtraida = 15 img1')
	plt.subplot(245); plt.imshow(h2,                cmap='gray'); plt.title('H com dilatação img2')
	plt.subplot(246); plt.imshow(h2_th_manual,      cmap='gray'); plt.title('Limiarização Manual\ncom dilatação img2')
	plt.subplot(247); plt.imshow(h2_th_otsu,        cmap='gray'); plt.title('Limiarização de Otsu\ncom dilatação img2')
	plt.subplot(248); plt.imshow(h2_th_adaptive,    cmap='gray'); plt.title('Limiarização Adaptativa\ncom dilatação img2')
	plt.show()





# view_colors_channels(video, True)
# _, img1 = video.read()
# _, img2 = video.read()
# calc_velocity_frames(img1, img2)


calc_velocity_video(video, scale=0.32, period=10)


