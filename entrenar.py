# Necesitamos los packages: tensorflow, pillow, SciPy
import sys
import os

import numpy as np
from pathlib import Path
from PIL import Image

# Importamos tensorflow
import tensorflow as tf
# Para preprocesar las imágenes que vamos darle a la CNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import optimizers
# Para que nuestra CNN sea sequencial (aplicamos filtros de forma sequencial)
from tensorflow.keras.models import Sequential
# Importamos los diferentes filtros
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
# Para hacer las convoluciones y los Poolings
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
# Para matar sesiones de keras que estén ejecutándose en segundo plano
from tensorflow.keras import backend as K

# Empezamos un entrenamiento desde 0
K.clear_session()

# Path al directorio de entrenamiento y validacion
data_entrenamiento = './data/entrenamiento'
data_validacion = './data/validacion'

# Parametros de la red neuronal

# Numero de veces que vamos a iterar sobre el set de datos durante el entrenamiento
epocas = 30
# Tamaño con el cual vamos a procesar estas imágenes (para el resize de las imagenes)
altura, longitud = 28, 28
# Numero de imágenes que vamos a enviar para procesar en cada uno de los pasos
batch_size = 32
# Número de veces que se va a procesar el set de entrenamiento en cada una de las epocas
# Los pasos están relacionados con el batch size. En general, el valor de los pasos tiene que ser 
# el resultado de la divisón (cantImages/batch_size)
pasos = 4
# Al final de cada una de las epocas, se corren X pasos con el set de validación
# Sucede lo mismo que con el numero de pasos anterior. El valor es recomendable que sea
# el resultado de (cantImagenesValidacion/batch_size)
pasos_validacion = 2

# Número de filtros que vamos a aplicar en cada convolución. 
# Después de cada convolución nuestra imagen quedará con más profundidad.
filtrosConv1 = 16
filtrosConv2 = 32
filtrosConv3 = 64

# Tamaño de los filtros que utilizaremos en cada convolución
tam_filtro = (3,3)

# Tamaño de filtro que vamos a usar en max pooling
tamano_pool = (2,2)

# Número de clases (gato, perro, etc)
clases = 2

# Learning rate, qué tan grandes deben ser los ajustes que realice la red neuronal
# para acercarse a una solución óptima
lr=0.00003	

# Pre procesamiento de imagenes, lo que hacemos es tanto NORMALIZAR el set de datos 
# como AUMENTAR el set (data augmentation). 
entrenamiento_datagen = ImageDataGenerator(
	rescale=1./255, # Hacemos que todos los píxeles estén en un rango de 0 a 1
	brightness_range=(0.3, 1.0) # Cambia el brillo d las imágenes según el rango especificado		
		
	#featurewise_center=True,
	#shear_range=0.3, # Para que algunas imagenes esten inclinadas
	#zoom_range=0.3, # Para que a algunas imagenes les haga zoom
	#horizontal_flip=True, # Invertir horizontalmente las imagenes
	#width_shift_range=0.3, # Mueve la imagen hacia la izquierda o derecha	
)

# Le damos las imágenes tal cual son en la validación, solo que con un rescale
validacion_datagen = ImageDataGenerator(rescale=1./255) #featurewise_center=True)

# Procesa todas las imagenes que esten dentro de las carpetas
imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
	data_entrenamiento,
	target_size=(altura, longitud), 
	batch_size=batch_size,
	color_mode="rgb",
	class_mode='categorical'
)

imagen_validacion = validacion_datagen.flow_from_directory(
	data_validacion,
	target_size=(altura, longitud),
	batch_size=batch_size,
	color_mode="rgb",
	class_mode='categorical'
)

# Creamos la CNN
# Indicamos que nuestra red neuronal es secuencial
cnn = Sequential() 

# Nuestra primera capa va a ser una convolución con 32 filtros
# input_shape indica cual es la altura y longitud de las imagenes, y los canales (rgb)
# Utilizamos relu como activacion
cnn.add(Convolution2D(filtrosConv1, tam_filtro, padding='same', strides=1, input_shape=(altura, longitud, 3), activation='relu'))

# Añadimos una layer de batch normalization para normalizar el lote de datos entre capas
# Hay discusión sobre donde se deben poner las capas de BatchNormalization. Algunos dicen que antes de una activación. Otros dicen que después. 
# El paper que diseñó este tipo de capas dice que deben ser usadas antes de una activación que no sea lineal (como sigmoid, tanh, etc).
# El paper que usamos como base dice que utiliza luego de cada capa.
cnn.add(BatchNormalization())

# Agregamos otra capa de convolucion
cnn.add(Convolution2D(filtrosConv2, tam_filtro, padding='same', strides=1, activation='relu'))

cnn.add(BatchNormalization())
# Agregamos otra capa de convolucion
cnn.add(Convolution2D(filtrosConv3, tam_filtro, padding='same', strides=1, activation='relu'))
cnn.add(BatchNormalization())

# Agregamos una capa de pooling, indicando el tamaño del filtro
cnn.add(MaxPooling2D(pool_size=tamano_pool, strides=2))

# Aplanamos la imagen en una dimension que contendrá toda la información
cnn.add(Flatten())

# Lo mandamos a una capa de 128 neuronas
cnn.add(Dense(128, activation='relu'))
# A la anterior capa le apagamos el 50% de las neuronas en cada paso, de manera aleatoria
#cnn.add(Dropout(0.5))
#cnn.add(BatchNormalization())
# Lo mandamos a una capa de 64
cnn.add(Dense(64, activation='relu'))
cnn.add(BatchNormalization())

# Ultima capa, con cant. neuronas = clases. Softmax nos indica la probabilidad
# de que la imagen sea un gato, perro, gorila...
cnn.add(Dense(clases, activation='sigmoid'))

# Para optimizar nuestro algoritmo. 
# loss es una medida de los errores que produce la red neuronal al intentar predecir la salida, es una suma de dichos errores y no un porcentaje
# se optimiza con Adam
# metrics con accuracy es el porcentaje de qué tan bien está aprendiendo la red neuronal
cnn.compile(loss='categorical_crossentropy', 
	optimizer=optimizers.Adam(lr=lr), 	
	metrics=['accuracy'])

# Entrenamos la red neuronal
cnn.fit(imagen_entrenamiento, #Data Generator
	steps_per_epoch=pasos,
	epochs=epocas, 
	validation_data=imagen_validacion, 
	validation_steps=pasos_validacion)

# Guardamos nuestro modelo en un archivo para no tener que volver a entrenarlo cada vez
# que querramos hacer una prediccion

dir='./modelo/'

if not os.path.exists(dir):
	os.mkdir(dir)

cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')