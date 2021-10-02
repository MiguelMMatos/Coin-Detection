import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[23]:


def printImg(img, name, imgSize, titleSize):
    plt.figure(figsize=(imgSize, imgSize))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.title(name, fontsize=titleSize)
    plt.show()


def print2Img(img, imgg, name1, name2):
    plt.figure(figsize=(20, 20))
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis(False)
    plt.title(name1, fontsize=17)

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB))
    plt.axis(False)
    plt.title(name2, fontsize=17)
    plt.show()


def print3Imgs(img, name1, imgg, name2, imggg, name3):
    plt.figure(figsize=(20, 20))
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis(False)
    plt.title(name1, fontsize=19)

    plt.subplot(1, 3, 2)
    plt.imshow(imgg, cmap='gray', interpolation='bicubic')
    plt.axis(False)
    plt.title(name2, fontsize=19)

    plt.subplot(1, 3, 3)
    plt.imshow(imggg, cmap='gray', interpolation='bicubic')
    plt.axis(False)
    plt.title(name3, fontsize=19)
    plt.show()


# In[24]:


def histRGB(img, name1, name2):
    plt.figure(figsize=(15, 5))
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis(False)
    plt.title(name1)
    color = ('b', 'g', 'r')
    plt.subplot(1, 2, 2)
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.title(name2)
    plt.show()


def histRGB2(img, var, name1, name2):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        if (var == col):
            plt.figure(figsize=(15, 5))
            plt.xticks([]), plt.yticks([])
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis(False)
            plt.title(name1)

            plt.subplot(1, 2, 2)
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
            plt.title(name2)
            plt.show()

    plt.show()


def ColorRGB(img, var):
    b, g, r = cv2.split(img)
    if (var == 'r'):
        return r
    elif (var == 'g'):
        return g
    elif (var == 'b'):
        return b


def threshold(img, minValue, modo, maxValue=255):
    thres, bw = cv2.threshold(img, minValue, maxValue, modo)
    return bw


source = "trainImages/"
img = cv2.imread(source + '1.jpg')
histRGB(img, "Original", "RGB")

img = cv2.imread(source + '1.jpg')
print3Imgs(ColorRGB(img, 'r'), "Red channel",
           ColorRGB(img, 'g'), "Green channel",
           ColorRGB(img, 'b'), "Blue channel")

img = cv2.imread('images/blue.png')
# printImg(img,"blue",8,20)
histRGB2(img, 'r', "Imagem editada do fundo", "Histograma Red Channel da Imagem editada do fundo")

img = cv2.imread('images/coins.png')
# printImg(img,"coins",10,20)
histRGB2(img, 'r', "Imagem editada de moedas", "Histograma Red Channel da Imagem editada de moedas")

img = cv2.blur(cv2.imread('images/blue.png'), (4, 4))
imgg = threshold(ColorRGB(cv2.imread('images/blue.png'), 'r'), 100, cv2.THRESH_BINARY)
print2Img(img, imgg, "Imagem editada do fundo", "Threshold com valor de 100 da Imagem editada do fundo")

img = cv2.blur(cv2.imread('images/coins.png'), (4, 4))
imgg = threshold(ColorRGB(cv2.imread('images/coins.png'), 'r'), 100, cv2.THRESH_BINARY)
print2Img(img, imgg, "Imagem editada de moedas", "Threshold com valor de 100 da Imagem editada de moedas")

img = cv2.imread(source + '1.jpg')
imgg = threshold(ColorRGB(img, 'r'), 100, cv2.THRESH_BINARY)
print2Img(img, imgg, "Original", "Red channel")

for x in range(5, 9):
    img = cv2.imread(source + str(x) + '.jpg')
    noBlur = threshold(ColorRGB(img, 'r'), 100, cv2.THRESH_BINARY)
    blur = threshold(ColorRGB(cv2.blur(img, (4, 4)), 'r'), 100, cv2.THRESH_BINARY)
    print2Img(noBlur, blur, "No Blur", "Blur")


def metodo1(teste2):
    M = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11), (-1, -1))
    bw = cv2.erode(teste2, M, iterations=2)
    bw = cv2.dilate(bw, M, iterations=1)
    bw = cv2.erode(bw, M, iterations=3)

    return bw


for x in range(1, 10):
    img = cv2.imread(source + str(x) + '.jpg')
    print2Img(img, metodo1(
        threshold(ColorRGB(cv2.blur(img, (4, 4)), 'r'), 100,
                  cv2.THRESH_BINARY)), "Original", "Threshold com transformações morfologicas")

dicionarioMoedas = {
    "1 centimo": [0.01, 245, 260],
    "2 centimos": [0.02, 307, 324],
    "10 centimos": [0.10, 325, 347],
    "5 centimos": [0.05, 355, 374],
    "20 centimos": [0.20, 381, 405],
    "1 euro": [1, 410, 431],
    "50 centimos": [0.50, 433, 452],
    "2 euros": [2.0, 454, 480]
}


# In[12]:


def FindAllCountors(bw, image):
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont = 0
    money = 0
    valorMinimo = dicionarioMoedas.get("1 centimo")[1] - 1

    for c in contours:
        if (hierarchy[0][cont][2] != -1 or hierarchy[0][cont - 1][2] != -1):
            cont += 1
            continue

        cont += 1

        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        perimetro = cv2.arcLength(c, True)
        area = cv2.contourArea(c)

        if (perimetro > valorMinimo):  # Impossivel haver moedas inferiores a 244 neste contexto
            raio1 = int(perimetro / (2 * np.pi))
            raio2 = int(np.sqrt(area / (np.pi)))
            if (abs(raio1 - raio2) <= 5):
                for key, value in dicionarioMoedas.items():
                    if perimetro >= value[1] and perimetro <= value[2]:
                        money += value[0]
                        cv2.circle(image, (cX, cY), int(perimetro / (2 * np.pi)) + 16, (0, 255, 0), 2)
                        cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)
                        # cv2.putText(image, "raio1: " + str(raio1) + " | raio2: "+ str(int(raio2)), (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(image, str(key), (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(image, "Total: " + str(round(money, 2)) + " Euros", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                2)
    return image


for x in range(1, 10):
    image1_Ori = cv2.imread(source + str(x) + '.jpg')
    image2 = metodo1(threshold(ColorRGB(cv2.blur(image1_Ori, (4, 4)), 'r'), 100, cv2.THRESH_BINARY))
    printImg(FindAllCountors(image2, image1_Ori), "", 15, 0)






