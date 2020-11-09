

import matplotlib.pyplot as plt
import numpy as np
import os, shutil
from datetime import datetime

import tensorflow as tf
from keras.initializers import Constant
from tensorflow.keras.preprocessing import image_dataset_from_directory
from classification_models.tfkeras import Classifiers
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

# Para no tener problemas de memoria con la GPU / evitar el problema de cudNN
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#ResNet18, preprocess_input = Classifiers.get('resnet18')

# Directorio donde se guardarán los logs de cada ejecución de este script
log_dir = './logs/'
# Directorio donde se encuentra el dataset
dataset_dir = './dataset/'
colormap_dir="viridis/"

# Parámetros constantes de la red neuronal
IMG_SIZE = (160, 160)
IMG_SHAPE = IMG_SIZE + (3,)
CLASS_MODE = 'binary'

# K fold parameters
num_folds = 10

def main():
  trainVariableLearningRate()
  trainVariableEpochs()

def readDataset(batch_size):
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
    numpy_images = np.concatenate((numpy_images, images)) if numpy_images.size else images
    numpy_labels = np.concatenate((numpy_labels, labels)) if numpy_labels.size else labels

  return numpy_images, numpy_labels

def trainVariableLearningRate():
  epochs=100
  lr_dir="learning-rate/"
  testNumber = 1
  
  numpy_images, numpy_labels = readDataset(16)
  
  train(16, 0.0002, epochs, numpy_images, numpy_labels, lr_dir, testNumber)
  testNumber+=1
  train(16, 0.0005, epochs, numpy_images, numpy_labels, lr_dir, testNumber)
  testNumber+=1
  train(16, 0.0009, epochs, numpy_images, numpy_labels, lr_dir, testNumber)
  testNumber+=1
  
  numpy_images, numpy_labels = readDataset(32)
  
  train(32, 0.0002, epochs, numpy_images, numpy_labels, lr_dir, testNumber)
  testNumber+=1
  train(32, 0.0005, epochs, numpy_images, numpy_labels, lr_dir, testNumber)
  testNumber+=1
  train(32, 0.0009, epochs, numpy_images, numpy_labels, lr_dir, testNumber)


def trainVariableEpochs():
  lr=0.0005
  epoch_dir="epochs/"
  testNumber = 1
  
  numpy_images, numpy_labels = readDataset(16)
  
  train(16, lr, 50, numpy_images, numpy_labels, epoch_dir, testNumber)
  testNumber+=1
  train(16, lr, 250, numpy_images, numpy_labels, epoch_dir, testNumber)
  testNumber+=1
  
  numpy_images, numpy_labels = readDataset(32)
  
  train(32, lr, 50, numpy_images, numpy_labels, epoch_dir, testNumber)
  testNumber+=1
  train(32, lr, 250, numpy_images, numpy_labels, epoch_dir, testNumber)


def train(batch_size, base_learning_rate, initial_epochs, numpy_images, numpy_labels, test_dir, test_number):
  #batch_size = 16
  #base_learning_rate = 0.0002
  #initial_epochs = 100
  model = getModel()
  
  # Define per-fold score containers
  acc_per_fold = []
  loss_per_fold = []

  # Define the K-fold Cross Validator
  #kfold = KFold(n_splits=num_folds, shuffle=True)
  kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)

  # K-fold Cross Validation model evaluation
  fold_no = 1

  for train, test in kfold.split(numpy_images, numpy_labels):
      # preprocesamiento del train en cada iteracion?

      # Configuramos el modelo, indicando su learning reate, la funcion que va a ser utilizada en las metricas, y la metrica de accuracy
      model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

      # Generate a print
      print('------------------------------------------------------------------------')
      print(f'Training for fold {fold_no} ...')

      # Fit data to model
      model.fit(numpy_images[train], 
                numpy_labels[train],
                batch_size=batch_size,
                epochs=initial_epochs,
                verbose=1)

      # Generate generalization metrics
      scores = model.evaluate(numpy_images[test], numpy_labels[test], verbose=0)
      print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
      acc_per_fold.append(scores[1] * 100)
      loss_per_fold.append(scores[0])

      # Increase fold number
      fold_no = fold_no + 1
  tf.keras.backend.clear_session()
  printAverageScores(acc_per_fold, loss_per_fold)
  generateLog(batch_size, base_learning_rate, initial_epochs, acc_per_fold, loss_per_fold, test_dir, test_number)

def getModel():
  # Importamos el modelo base desde una red neuronal pre-entrenada
  # include_top = false hace que se cargue la red neuronal sin traer la última capa (fully connected layer),
  # que es lo que queremos, ya que deseamos pre-entrenar, y no usar las mismas predicciones de la pre-entrenada
  #base_model = ResNet18(input_shape=IMG_SHAPE, weights='imagenet', include_top=False)
  base_model = tf.keras.applications.Xception(input_shape=IMG_SHAPE, weights='imagenet', include_top=False)

  preprocess_input = tf.keras.applications.xception.preprocess_input

  # Congelamos todas las capas de la red pre-entrenada para prevenir que se entrenen
  base_model.trainable = False

  global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
  prediction_layer = tf.keras.layers.Dense(1)

  # Input crea un tensor 
  inputs = tf.keras.Input(shape=(IMG_SHAPE))
  # x = data_augmentation(inputs)
  # Pre-procesamos el input para que se adapte a lo que necesita RESNET18, esto es necesario porque la red 
  # fue entrenada con un input totalmente diferente al que vamos usar nosotros
  x = preprocess_input(inputs)
  # feature extraction layer
  x = base_model(x, training=False)
  x = global_average_layer(x)
  #x = tf.keras.layers.Dropout(0.2)(x)

  x = tf.keras.layers.PReLU(alpha_initializer=Constant(value=0.25))(x)
  x = tf.keras.layers.PReLU(alpha_initializer=Constant(value=0.25))(x)
  x = tf.keras.layers.Dropout(0.5)(x)
  outputs = prediction_layer(x)
  model = tf.keras.Model(inputs, outputs)

  return model

def printAverageScores(acc_per_fold, loss_per_fold):
  # == Provide average scores ==
  print('------------------------------------------------------------------------')
  print('Score per fold')
  for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
  print('------------------------------------------------------------------------')
  print('Average scores for all folds:')
  print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
  print(f'> Loss: {np.mean(loss_per_fold)}')
  print('------------------------------------------------------------------------')

def generateLog(batch_size, base_learning_rate, initial_epochs, acc_per_fold, loss_per_fold, test_dir, test_number):
  # Creamos una carpeta donde vamos a guardar los registros de esta sesion
  sessionLog_dir = log_dir + colormap_dir + test_dir) #str(datetime.today().strftime('%d-%m-%Y-%Hh%Mm%Ss')
  os.makedirs(sessionLog_dir)

  print('Creando archivo de paramétros')

  # a+ crea un archivo de lectura y escritura
  paramFile = open(sessionLog_dir + "/" + str(test_number) + ".txt", "a+")
  paramFile.write("batch_size: " + str(batch_size) + "\n")
  paramFile.write("IMG_SIZE: " + str(IMG_SIZE) + "\n")
  paramFile.write("IMG_SHAPE: " + str(IMG_SHAPE) + "\n")
  paramFile.write("CLASS_MODE: " + str(CLASS_MODE) + "\n")
  paramFile.write("base_learning_rate: " + str(base_learning_rate) + "\n")
  paramFile.write("initial_epochs: " + str(initial_epochs) + "\n")

  # == Write average scores ==
  paramFile.write('------------------------------------------------------------------------\n')
  paramFile.write('Score per fold\n')
  for i in range(0, len(acc_per_fold)):
    paramFile.write('------------------------------------------------------------------------\n')
    paramFile.write(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}% \n')
  paramFile.write('------------------------------------------------------------------------\n')
  paramFile.write('Average scores for all folds: \n')
  paramFile.write(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)}) \n')
  paramFile.write(f'> Loss: {np.mean(loss_per_fold)} \n')
  paramFile.write('------------------------------------------------------------------------\n')

  paramFile.close()

  print('Archivo de paramétros generado')

main()