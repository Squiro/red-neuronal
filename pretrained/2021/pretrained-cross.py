

import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import tensorflow as tf
from datetime import datetime
from keras.initializers import Constant
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import PReLU, Dropout, GlobalAveragePooling2D, Dense
from classification_models.tfkeras import Classifiers
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from keras.applications.resnet50 import ResNet50

# Para no tener problemas de memoria con la GPU / evitar problemas de cudNN
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

log_dir = './logs/'
dataset_dir = './dataset/'

# Parámetros de la red neuronal
batch_size = 16
IMG_SIZE = (160, 160)
IMG_SHAPE = IMG_SIZE + (3,)
CLASS_MODE = 'binary'
base_learning_rate = 0.0005
initial_epochs = 20

# K fold parameters
num_folds = 10

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

# Leemos todas las imágenes de las carpetas de entrenamiento y validación
dataset = image_dataset_from_directory(dataset_dir,
                                       shuffle=True,
                                       batch_size=batch_size,
                                       color_mode="rgb",
                                       image_size=IMG_SIZE)

# for images, labels in dataset.take(-1): #-1 takes all elements
#   print(images.numpy())
#   numpy_images = images.numpy()
#   numpy_labels = labels.numpy()

numpy_images = np.array([])
numpy_labels = np.array([])

for images, labels in dataset.as_numpy_iterator():
    numpy_images = np.concatenate(
        (numpy_images, images)) if numpy_images.size else images
    numpy_labels = np.concatenate(
        (numpy_labels, labels)) if numpy_labels.size else labels

# Importamos un "Modelo base" generado por una red neuronal que ya ha sido entrenada
# include_top = false toma la arquitectura de la red sin traer la última capa (classification layer),
# debido a que las capas anteriores poseen mas generalidad (es decir, dependen menos del dataset con el que fueron entrenadas originalmente)
#base_model = ResNet18(input_shape=IMG_SHAPE, weights='imagenet', include_top=False)
base_model = ResNet50(input_shape=IMG_SHAPE,
                      weights='imagenet', include_top=False)

# La funcion de pre-procesado debe ser modificada según la red utilizada
preprocess_input = tf.keras.applications.resnet50.preprocess_input

# Congelamos todas las capas de la red pre-entrenada para prevenir que se entrenen
base_model.trainable = False

# La llamada input crea un objeto llamado "Tensor". Utilizamos esto para definir la "forma" (shape) de los inputs con los
# que alimentaremos a la red
inputs = tf.keras.Input(shape=(IMG_SHAPE))

# Si tuvieramos un proceso de data_augmentation, podriamos definirlo de esta forma
# x = data_augmentation(inputs)

# Pre-procesamos el input para que se adapte a lo que necesita la red seleccionada, esto es necesario porque la red
# fue entrenada con un input totalmente diferente al que vamos a usar nosotros
x = preprocess_input(inputs)

# Al hacer una llamada a base_model(), tomamos las capas consideras como Feature Extraction Layers de la arquitectura de la red
# y se la añadimos a nuestro modelo "x"
x = base_model(x, training=False)

# En caso de ser necesario, podemos agregar algunas capas extra de Pooling o Convolution
x = GlobalAveragePooling2D()(x)

# Como el modelo base importado no posee la última capa de clasificación, creamos una personalizada
x = Dense(512, activation=PReLU(alpha_initializer=Constant(value=0.25)))(x)
x = Dense(256, activation=PReLU(alpha_initializer=Constant(value=0.25)))(x)
x = Dropout(0.5)(x)
# Seteamos una capa de predicción de una sola neurona, utilizando sigmoid como función de activación para realizar predicciones binarias
# De esta forma, los resultados de las predicciones estarán dentro del rango [0, 1]
outputs = Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)

# Define the K-fold Cross Validator
kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
# Fold number
fold_no = 1

for train, test in kfold.split(numpy_images, numpy_labels):
    # Model configuration, such as optimizer, metrics, etc
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history = model.fit(numpy_images[train], numpy_labels[train],
                        batch_size=batch_size,
                        epochs=initial_epochs,
                        verbose=1)

    # Generate generalization metrics
    scores = model.evaluate(numpy_images[test], numpy_labels[test], verbose=0)
    print(
        f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1
tf.keras.backend.clear_session()

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(
        f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')


# Creamos una carpeta donde vamos a guardar los registros de esta sesion

sessionLog_dir = log_dir + str(datetime.today().strftime('%d-%m-%Y-%Hh%Mm%Ss'))
os.mkdir(sessionLog_dir)

# Copiamos el set de datos usado en esta sesión

#print('Copiando dataset utilizado en esta sesión')

#os.mkdir(sessionLog_dir + '/dataset/')

# source, dest
#shutil.copytree(dataset_dir, sessionLog_dir + '/dataset/entrenamiento')

#print('Dataset copiado')

# Generamos un archivo de texto con los parámetros utilizados en esta sesión

print('Creando archivo de paramétros')

paramFile = open(sessionLog_dir + "/parametros.txt", "a+")
paramFile.write("batch_size: " + str(batch_size) + "\n")
paramFile.write("IMG_SIZE: " + str(IMG_SIZE) + "\n")
paramFile.write("IMG_SHAPE: " + str(IMG_SHAPE) + "\n")
paramFile.write("CLASS_MODE: " + str(CLASS_MODE) + "\n")
paramFile.write("base_learning_rate: " + str(base_learning_rate) + "\n")
paramFile.write("initial_epochs: " + str(initial_epochs) + "\n")


paramFile.write(
    '------------------------------------------------------------------------\n')
paramFile.write('Score per fold\n')
for i in range(0, len(acc_per_fold)):
    paramFile.write(
        '------------------------------------------------------------------------\n')
    paramFile.write(
        f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}% \n')
paramFile.write(
    '------------------------------------------------------------------------\n')
paramFile.write('Average scores for all folds: \n')
paramFile.write(
    f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)}) \n')
paramFile.write(f'> Loss: {np.mean(loss_per_fold)} \n')
paramFile.write(
    '------------------------------------------------------------------------\n')

paramFile.close()

print('Archivo de paramétros generado')
