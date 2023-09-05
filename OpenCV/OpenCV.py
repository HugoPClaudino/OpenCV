import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import *
import matplotlib.widgets 
from matplotlib.widgets import Button, CheckButtons , RadioButtons
# Upload settings

carrega_algoritimo = cv2.CascadeClassifier('haarcascades\haarcascade_frontalface_default.xml')
imagem = cv2.imread('fotos_opencv/2.jpg')

# Effects
#videocinza = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)
foto_infravermelha = cv2.applyColorMap(rgb, cv2.COLORMAP_JET)
escalacinza = cv2.applyColorMap(rgb, cv2.COLORMAP_BONE)
foto_cinza = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

# Recon faces
faces1 = carrega_algoritimo.detectMultiScale(escalacinza)

# Shows faces in terminal
print (faces1)

# Create a box at the face
for(x , y , l , a) in faces1:
  cv2.rectangle(imagem, (x , y) , (x + l , y + a) , (0,255,0), 2)
  
# Create subplots for displaying images
fig, axs = plt.subplots(3, 2, figsize=(6, 6))

# Display the original image
axs[2, 1].imshow(imagem ,)
axs[2, 1].set_title('Detector de rostos', pad=10)
axs[2, 1].axis('off')

axs[0, 1].imshow(foto_infravermelha,)
axs[0, 1].set_title('Visão termica',pad=8)
axs[0, 1].axis('off')

axs[1, 1].imshow(lab)
axs[1, 1].set_title('Lab',pad=8)
axs[1, 1].axis('off')

axs[1, 0].imshow(hsv)
axs[1, 0].set_title('Hsv',pad=8)
axs[1, 0].axis('off')

axs[2, 0].imshow(escalacinza)
axs[2, 0].set_title('Escala de cinza',pad=8)
axs[2, 0].axis('off')

axs[0, 0].imshow(rgb)
axs[0, 0].set_title('Original',pad=8)
axs[0, 0].axis('off')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plt with all images
plt.show()

#OLD
#cv2.imshow("Infravermelho", foto_infravermelha)
#cv2.imshow("Reconhecimento facial", imagem)
#cv2.imshow("L*a*b*", lab)
#cv2.imshow("HSV", hsv)
#cv2.imshow("faces", escalacinza)
cv2.waitKey()