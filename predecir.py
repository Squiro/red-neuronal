import sys
import os
import numpy as np
#import keras.backend as kerasBackend
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf 
import csv

longitud, altura = 28, 28
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
# Cargamos el modelo de la CNN
cnn = load_model(modelo)
# Cargamos los pesos
cnn.load_weights(pesos_modelo)

# Borramos el txt de predicciones antes de empezar
if (os.path.isfile("prediccion.txt")):
    os.remove("prediccion.txt")
# Abrimos / creamos el txt de predicciones
preddicionesTxt = open("prediccion.txt", "a+")

# Borramos el csv de predicciones antes de empezar
if (os.path.isfile("predcsv.csv")):
    os.remove("predcsv.csv")
# Abrimos / creamos el txt de predicciones
predcsv = open("predcsv.csv", "a+")
# Creamos un writer para poder escribir en el csv
predcsv_writer = csv.writer(predcsv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
predcsv_writer.writerow(['Imagen', 'Prob. Sano', 'Prob. Enfermo'])

def predecir(imagepath, imagename):
    # Cargamos la imagen
    image = load_img(imagepath, target_size=(longitud, altura))
    # Convertimos la imagen en un arreglo de valores que la representa 
    image = img_to_array(image)
    # Necesitamos normalizar la imagen, de la misma forma que la normalizamos con ImageDataGenerator en el script de entrenamiento
    image = image.astype('float32')
    # Dividimos la imagen por 255, por la razon de arriba ^
    image /= 255

    # Añadimos una dimensión extra a la imagen
    image = np.expand_dims(image, axis=0)
    # Hacemos la prediccion sobre la imagen, esto nos devuelve un array
    # de dos dimensiones. El valor que la CNN cree que es correcto estará en la primera
    # dimensión. e.g. [[1,0,0]]
    array = cnn.predict(image)
    # Tomamos la dimensión la primera dimensión. e.g. [1,0,0]
    result = array[0]
    
    print("Probabilidades en base a las clases: " + str(result))
    preddicionesTxt.write("Probabilidades en base a las clases: " + str(result) + "\n")
    
    predcsv_writer.writerow([imagename, str(result[0]), str(result[1])])
    # Toma el valor más alto del array y nos devuelve la posición en donde se encuentra
    # el mismo (el índice)
    answer = np.argmax(result)
    # "If it’s a classification problem,then what the model outputs will corresponds to what you trained it on.
    # If you do flow from directory, then the classes will be in alphabetical order.""

    # Nosotros utilizamos flow from directory al momento de alimentar las imágenes a
    # la CNN. Entonces, si está en orden alfabético, los indices serán
    # A -- 0
    # B -- 1
    # Y así.

    return answer

def main(): 
    pathFalse="./test-images/false/"
    classIndexFalse=0
    pathTrue="./test-images/true/"
    classIndexTrue=1

    preddicionesTxt.write("------------------------------- \n")
    preddicionesTxt.write("Predicciones en base a imágenes de FALSE (Sanos): \n")
    predecirClases(classIndexFalse, "false", pathFalse)

    preddicionesTxt.write("------------------------------- \n")
    preddicionesTxt.write("Predicciones en base a imágenes de TRUE (Enfermos): \n")
    predecirClases(classIndexTrue, "true", pathTrue)

def predecirClases(classIndex, className, path): 
    cantImg=0
    cantAciertos=0
    for image in os.listdir(path):
        preddicionesTxt.write("Imagen: " + image + "\n")
        if (predecir(path+image, image) == classIndex):
            preddicionesTxt.write("Predicción: es " + className + "\n")
            cantAciertos+=1
        else:
            preddicionesTxt.write("No lo predije correctamente :( \n")
        cantImg+=1
        preddicionesTxt.write("\n\n")
    preddicionesTxt.write("Cantidad: " + str(cantImg) + "     Aciertos: " + str(cantAciertos) + "    Ratio: " + str((cantAciertos/cantImg)) + "\n")

main()