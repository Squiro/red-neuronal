import sys
import os
import numpy as np
import keras.backend as kerasBackend
from keras.preprocessing.image import load_img, img_to_array
#from keras.models import load_model
import tensorflow as tf 

archivo = open("prediccion.txt", "a+")
longitud, altura = 100, 100
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
cnn = tf.keras.models.load_model(modelo)#= load_model(modelo)
cnn.load_weights(pesos_modelo)

def predecir(imagepath):
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
    #print("Probabilidades en base a las clases: " + str(result))
    archivo.write("Probabilidades en base a las clases: " + str(result) + "\n")
    # Toma el valor más alto del array y nos devuelve la posición en donde se encuentra
    # el mismo (el índice)
    answer = np.argmax(result)
    # If it’s a classification problem,then what the model outputs will corresponds to what you trained it on.
    # If you do flow from directory, then the classes will be in alphabetical order.

    # Nosotros utilizamos flow from directory al momento de alimentar las imágenes a
    # la CNN. Entonces, si está en orden alfabético, los indices serán
    # A -- 0
    # B -- 1
    # Y así.

    #if answer == 0:
    #    print("Predicción: es Ada")
    #elif answer == 1:
    #    print("Predicción: es Kira")
    return answer

def main(): 
    pathAda="./test-images/ada/"
    classIndexAda=0
    pathKira="./test-images/kira/"
    classIndexKira=1
    #print("-------------------------------")
    #print("Predicciones en base a imágenes de Ada:")
    archivo.write("------------------------------- \n")
    archivo.write("Predicciones en base a imágenes de Ada: \n")
    predecirClases(classIndexAda, "Ada", pathAda)
 
    #print("-------------------------------")
    #print("Predicciones en base a imágenes de Kira:")
    archivo.write("------------------------------- \n")
    archivo.write("Predicciones en base a imágenes de Kira: \n")
    predecirClases(classIndexKira, "Kira", pathKira)

def predecirClases(classIndex, className, path): 
    cantImg=0
    cantAciertos=0
    for image in os.listdir(path):
        #print("Imagen: " + image)
        archivo.write("Imagen: " + image + "\n")
        if (predecir(path+image) == classIndex):
            #print("Predicción: es " + className)
            archivo.write("Predicción: es " + className + "\n")
            cantAciertos+=1
        else:
            #print("No lo predije correctamente :(")
            archivo.write("No lo predije correctamente :( \n")
        cantImg+=1
        #print("\n\n")
        archivo.write("\n\n")
    #print("Cantidad: " + str(cantImg) + "     Aciertos: " + str(cantAciertos) + "    Ratio: " + str((cantAciertos/cantImg)))
    archivo.write("Cantidad: " + str(cantImg) + "     Aciertos: " + str(cantAciertos) + "    Ratio: " + str((cantAciertos/cantImg)) + "\n")

main()