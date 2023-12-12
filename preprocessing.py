import cv2

# Canny
def canny(img):
    bordas_canny = cv2.Canny(img, threshold1=100, threshold2=200)
    bordas_canny_colorida = cv2.cvtColor(bordas_canny, cv2.COLOR_GRAY2BGR)
    img_canny = cv2.addWeighted(img, 1, bordas_canny_colorida, 0.5, 0)
    return img_canny

# filtro Laplaciano(aguçamento) + filtro gaussiano (suavização)
def laplacian_gaussian(img):
    img = cv2.GaussianBlur(img, (3, 1), cv2.CV_64F)  # Pode ajustar o tamanho do kernel (5, 5) conforme necessário

    # Aplica o filtro Laplaciano em cada canal de cor separadamente
    laplaciano_r = cv2.Laplacian(img[:, :, 0], cv2.CV_64F)
    laplaciano_g = cv2.Laplacian(img[:, :, 1], cv2.CV_64F)
    laplaciano_b = cv2.Laplacian(img[:, :, 2], cv2.CV_64F)

    # Convertendo para o tipo de dado uint8 e combinando os canais
    laplaciano_r_uint8 = cv2.convertScaleAbs(laplaciano_r)
    laplaciano_g_uint8 = cv2.convertScaleAbs(laplaciano_g)
    laplaciano_b_uint8 = cv2.convertScaleAbs(laplaciano_b)

    # Combinando os canais para formar a imagem colorida final
    laplaciano_colorida = cv2.merge((laplaciano_r_uint8, laplaciano_g_uint8, laplaciano_b_uint8))

    return cv2.add(laplaciano_colorida, img)



