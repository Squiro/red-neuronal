

import matplotlib.pyplot as plt
import numpy as np
import os, shutil
from datetime import datetime

import tensorflow as tf
from keras.initializers import Constant
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from classification_models.tfkeras import Classifiers

# Para no tener problemas de memoria con la GPU / evitar el problema de cudNN
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
ResNet18, preprocess_input = Classifiers.get('resnet18')

# Directorio donde se guardarán los logs de cada ejecución de este script
log_dir = './logs/'
# Directorio donde se encuentra el dataset de entrenamiento
train_dir = './entrenamiento/'
# Directorio donde se encuentra el dataset de validación
validation_dir = './validacion/'

# Parámetros de la red neuronal
batch_size = 16
IMG_SIZE = (160, 160)
IMG_SHAPE = IMG_SIZE + (3,)
CLASS_MODE = 'binary'
base_learning_rate = 0.0005
initial_epochs = 100


# Leemos todas las imágenes de las carpetas de entrenamiento y validación
train_dataset = image_dataset_from_directory(train_dir,
                                                  shuffle=True,
                                                  batch_size=batch_size,
												  color_mode="rgb",
                                                  image_size=IMG_SIZE)

validation_dataset = image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=batch_size,
												  color_mode="rgb",
                                                  image_size=IMG_SIZE)

# Use buffered prefetching to load images from disk without having I/O become blocking. 
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# Importamos el modelo base desde una red neuronal pre-entrenada
# include_top = false hace que se cargue la red neuronal sin traer la última capa (fully connected layer),
# que es lo que queremos, ya que deseamos pre-entrenar, y no usar las mismas predicciones de la pre-entrenada
# base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
#                                                include_top=False,
#                                                weights='imagenet')

# base_model = tf.keras.applications.ResNet152V2(
# 	include_top=False,
# 	weights="imagenet",
# 	input_shape=IMG_SHAPE
# )

base_model = ResNet18(input_shape=IMG_SHAPE, weights='imagenet', include_top=False)

# Extraemos las caracteristicas / feature extractor
image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)

# Congelamos todas las capas de la red pre-entrenada para prevenir que se entrenen
base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1)

# Input crea un tensor 
inputs = tf.keras.Input(shape=(160, 160, 3))
# x = data_augmentation(inputs)
# Pre-procesamos el input para que se adapte a lo que necesita RESNET18, esto es necesario porque la red 
# fue entrenada con un input totalmente diferente al que vamos usar nosotros
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
#x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.PReLU(alpha_initializer=Constant(value=0.25))(x)
x = tf.keras.layers.PReLU(alpha_initializer=Constant(value=0.25))(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# Configuramos el modelo, indicando su learning reate, la funcion que va a ser utilizada en las metricas, y la metrica de accuracy
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
 
#loss0, accuracy0 = model.evaluate(validation_dataset)

history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)

# Creamos una carpeta donde vamos a guardar los registros de esta sesion

sessionLog_dir = log_dir + str(datetime.today().strftime('%d-%m-%Y-%Hh%Mm%Ss'))
os.mkdir(sessionLog_dir)

# Generación del gráfico de métricas
print('Generando gráfico de métricas...')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
# plt.show()

plt.savefig(sessionLog_dir + '/metricas' + '.png')

print('Gráfico generado')

# Copiamos el set de datos usado en esta sesión

print('Copiando dataset utilizado en esta sesión')

os.mkdir(sessionLog_dir + '/dataset/')
# os.mkdir(sessionLog_dir + '/dataset/entrenamiento')
# os.mkdir(sessionLog_dir + '/dataset/validacion')

# source, dest
shutil.copytree(train_dir, sessionLog_dir + '/dataset/entrenamiento')
shutil.copytree(validation_dir, sessionLog_dir + '/dataset/validacion')

print('Dataset copiado')

# Generamos un archivo de texto con los parámetros utilizados en esta sesión

print('Creando archivo de paramétros')

# a+ crea un archivo de lectura y escritura
paramFile = open(sessionLog_dir + "/parametros.txt", "a+")
paramFile.write("batch_size: " + str(batch_size) + "\n")
paramFile.write("IMG_SIZE: " + str(IMG_SIZE) + "\n")
paramFile.write("IMG_SHAPE: " + str(IMG_SHAPE) + "\n")
paramFile.write("CLASS_MODE: " + str(CLASS_MODE) + "\n")
paramFile.write("base_learning_rate: " + str(base_learning_rate) + "\n")
paramFile.write("initial_epochs: " + str(initial_epochs) + "\n")
paramFile.close()

print('Archivo de paramétros generado')
