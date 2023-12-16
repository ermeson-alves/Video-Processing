import cv2
import numpy as np
import requests
import json

def trackShip(img, column):

    # funcao que procura um pixel do navio partindo do pier, contando pixels de baixo para cima numa coluna

    # pixel acima do qual o algoritmo ira procurar o navio (abaixo deste pixel se encontra o pier)
    counter = 520

    diff = 0

    # cada pixel tem seus valores de cor comparado com o pixel acima
    # caso haja uma diferenca acima do threshhold, foi encontrado um pixel do navcio
    while (diff < 280):

        if img[counter, column, 0] > img[counter - 1, column, 0]:
            bigger0 = img[counter, column, 0]
            smaller0 = img[counter - 1, column, 0]
        else:
            bigger0 = img[counter - 1, column, 0]
            smaller0 = img[counter, column, 0]

        if img[counter, column, 1] > img[counter - 1, column, 1]:
            bigger1 = img[counter, column, 1]
            smaller1 = img[counter - 1, column, 1]
        else:
            bigger1 = img[counter - 1, column, 1]
            smaller1 = img[counter, column, 1]

        if img[counter, column, 2] > img[counter - 1, column, 2]:
            bigger2 = img[counter, column, 2]
            smaller2 = img[counter - 1, column, 2]
        else:
            bigger2 = img[counter - 1, column, 2]
            smaller2 = img[counter, column, 2]

        diff = (bigger0 - smaller0) + (bigger1 - smaller1) + (bigger2 - smaller2) * 2

        counter -= 1

        # caso o navio nao tenha sido encontrado, retorna 0
        if counter == 0:
            return 0

    return counter


# URL do servidor Flask
flask_url = 'http://127.0.0.1:8000/dados'

# path do video
video_path = 'part3.mp4'
cap = cv2.VideoCapture(video_path)

# kernel para realcar a imagem, deixando a diferenca de pixels mais acentuada
kernel = np.array([[0.0, -1.0, 0.0],
                   [-1.0, 5.0, -1.0],
                   [0.0, -1.0, 0.0]])

kernel = kernel/(np.sum(kernel) if np.sum(kernel) != 0 else 1)

ret = True
ship_found = False

frame_counter = 0
frame_array = []

# enquanto houverem frames do video, o loop ira continuar
# o loop pode ser encerrado apertando a tecla "q" na janela do video
while ret:
    # ret = true se ha mais um frame no video
    # frame = imagem do frame do video
    ret, frame = cap.read()

    if ret:

        # print('TRACKING NEXT FRAME')

        # nesse array serao armazenados os pixels das bordas do navio encontradas
        borders = []

        # aplicando o filtro de realce
        img = cv2.filter2D(frame, -1, kernel)

        # 20 colunas de pixels sao analisadas
        for column in range(500, 861, 18):
            counter = trackShip(img, column)
            borders.append(counter)

        # ordena o array de pixels da borda do navio e leva em consideracao os 7 pixels centrais para encontrar a media
        # esta operacao serve para retirar os pixels outliers e estabilizar o tracking do navio
        borders.sort()
        location = np.mean(borders[6:14])

        # converte a medida de pixels para metros
        position = (520-location)*0.32

        # valores de posicao dos ultimos 10 segundos sao levados em consideracao para calculo da velocidade
        if frame_counter < 240:
            frame_counter += 1
            frame_array.append(position)
        else:
            # velocidade do navio (metros por segundo)
            speed = (abs(np.mean(frame_array[:119])-np.mean(frame_array[120:]))/5)*100
            # posicao do navio (distancia em metros do pier ao navio)
            present_position = np.mean(frame_array[120:])
            print(present_position)
            print(speed)
            frame_array.pop(0)
            frame_array.append(position)

            # Enviar dados para o servidor Flask
            data = {'present_position': present_position, 'speed': speed}
            headers = {'Content-Type': 'application/json'}  # Definir o cabeçalho adequado
            try:
                response = requests.post(flask_url, data=json.dumps(data), headers=headers)
                response.raise_for_status()
                print('Dados enviados com sucesso!')
            except requests.exceptions.RequestException as e:
                print(f'Erro ao enviar dados para o servidor Flask: {e}')

        # insere na imagem do video uma linha horizontal vermelha indicando como esta sendo feito o tracking
        frame[(round(location) - 1):(round(location) + 1), (680-180):(680+180)] = [0, 0, 255]

        # mostra a imagem do frame na janela de video
        cv2.imshow('frame', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
