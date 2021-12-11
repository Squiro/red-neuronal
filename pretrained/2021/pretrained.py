

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
from keras.metrics import Precision, Recall, TrueNegatives, TruePositives, FalseNegatives, FalsePositives
from sklearn.metrics import ConfusionMatrixDisplay

def generateConfusionGraph(tp, tn, fp, fn, dir):
    cm = np.array([[tn,fp], [fn,tp]])
    ls = [0, 1] # your y labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ls)
    disp.plot()
    plt.savefig(f'{dir}/confusion-matrix.png')
    plt.close()

def plotMetrics(history, dir):
    fig, ax = plt.subplots(2, 4, constrained_layout=True, figsize=(15, 10))

    # Accuracy
    ax[0,0].plot(history.history['accuracy'])
    ax[0,0].plot(history.history['val_accuracy'])    
    ax[0,0].set_title('model accuracy')
    ax[0,0].set_ylabel('accuracy')
    ax[0,0].set_xlabel('epoch')
    ax[0,0].legend(['train', 'val'], loc='upper left')   

    # Loss

    ax[0,1].plot(history.history['loss'])
    ax[0,1].plot(history.history['val_loss'])
    ax[0,1].set_title('model loss')
    ax[0,1].set_ylabel('loss')
    ax[0,1].set_xlabel('epoch')
    ax[0,1].legend(['train', 'val'], loc='upper left')


    # Precision
    ax[0,2].plot(history.history['precision'])
    ax[0,2].plot(history.history['val_precision'])
    ax[0,2].set_title('model precision')
    ax[0,2].set_ylabel('precision')
    ax[0,2].set_xlabel('epoch')
    ax[0,2].legend(['train', 'val'], loc='upper left')


    # Recall
    ax[0,3].plot(history.history['recall'])
    ax[0,3].plot(history.history['val_recall'])
    ax[0,3].set_title('model recall')
    ax[0,3].set_ylabel('recall')
    ax[0,3].set_xlabel('epoch')
    ax[0,3].legend(['train', 'val'], loc='upper left')

    # TP
    ax[1,0].plot(history.history['true_positives'])
    ax[1,0].plot(history.history['val_true_positives'])
    ax[1,0].set_title('model true_positives')
    ax[1,0].set_ylabel('true_positives')
    ax[1,0].set_xlabel('epoch')
    ax[1,0].legend(['train', 'val'], loc='upper left')

    # TN
    ax[1,1].plot(history.history['true_negatives'])
    ax[1,1].plot(history.history['val_true_negatives'])
    ax[1,1].set_title('model true_negatives')
    ax[1,1].set_ylabel('true_negatives')
    ax[1,1].set_xlabel('epoch')
    ax[1,1].legend(['train', 'val'], loc='upper left')

    # FP
    ax[1,2].plot(history.history['false_positives'])
    ax[1,2].plot(history.history['val_false_positives'])
    ax[1,2].set_title('model false_positives')
    ax[1,2].set_ylabel('false_positives')
    ax[1,2].set_xlabel('epoch')
    ax[1,2].legend(['train', 'val'], loc='upper left')

    # FN
    ax[1,3].plot(history.history['false_negatives'])
    ax[1,3].plot(history.history['val_false_negatives'])
    ax[1,3].set_title('model false_negatives')
    ax[1,3].set_ylabel('false_negatives')
    ax[1,3].set_xlabel('epoch')
    ax[1,3].legend(['train', 'val'], loc='upper left')
    
    plt.savefig(f'{dir}/metrics.png')
    plt.close()


# Para no tener problemas de memoria con la GPU / evitar problemas de cudNN
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Use buffered prefetching to load images from disk without having I/O become blocking.
AUTOTUNE = tf.data.experimental.AUTOTUNE

log_dir = './logs/'
dataset_dir = './dataset/'
train_dir = './dataset/training/'
validation_dir = './dataset/validation/'
test_dir = './dataset/test/'

# Parámetros de la red neuronal
BATCH_SIZE = 32
IMG_SIZE = (224, 168)
IMG_SHAPE = IMG_SIZE + (3,)
CLASS_MODE = 'binary'
base_learning_rate = 0.00001
initial_epochs = 50

# Leemos todas las imágenes de las carpetas de entrenamiento y validación
train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             color_mode="rgb",
                                             image_size=IMG_SIZE)

validation_dataset = image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  color_mode="rgb",
                                                  image_size=IMG_SIZE)

test_dataset = image_dataset_from_directory(test_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  color_mode="rgb",
                                                  image_size=IMG_SIZE)

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

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

# Model configuration, such as optimizer, metrics, etc
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy', Precision(), Recall(), TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives()])

# Fit data to model
history = model.fit(train_dataset,
                    validation_data=validation_dataset,
                    batch_size=BATCH_SIZE,
                    epochs=initial_epochs,
                    verbose=1)


print("TESTING: Evaluación del modelo con dataset de testing")
results = model.evaluate(test_dataset, batch_size=BATCH_SIZE)
print("Testing metrics")
print("Loss: ", results[0])
print("Accuracy: ", results[1])
print("Precision: ", results[2])
print("Recall: ", results[3])
print("True positives: ", results[4])
print("True negatives: ", results[5])
print("False positives: ", results[6])
print("False negatives: ", results[7])

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
# print("Generate predictions for 3 samples")
# predictions = model.predict(test_dataset[:3])
# print("predictions shape:", predictions.shape)

tf.keras.backend.clear_session()

# Creamos una carpeta donde vamos a guardar los registros de esta sesion

sessionLog_dir = log_dir + str(datetime.today().strftime('%d-%m-%Y-%Hh%Mm%Ss'))
os.mkdir(sessionLog_dir)

generateConfusionGraph(results[4], results[5], results[6], results[7], sessionLog_dir)
plotMetrics(history, sessionLog_dir)

# Copiamos el set de datos usado en esta sesión

#print('Copiando dataset utilizado en esta sesión')

#os.mkdir(sessionLog_dir + '/dataset/')

# source, dest
#shutil.copytree(dataset_dir, sessionLog_dir + '/dataset/entrenamiento')

#print('Dataset copiado')

# Generamos un archivo de texto con los parámetros utilizados en esta sesión

print('Creando archivo de paramétros')

paramFile = open(sessionLog_dir + "/parametros.txt", "a+")
paramFile.write("batch_size: " + str(BATCH_SIZE) + "\n")
paramFile.write("IMG_SIZE: " + str(IMG_SIZE) + "\n")
paramFile.write("IMG_SHAPE: " + str(IMG_SHAPE) + "\n")
paramFile.write("CLASS_MODE: " + str(CLASS_MODE) + "\n")
paramFile.write("base_learning_rate: " + str(base_learning_rate) + "\n")
paramFile.write("initial_epochs: " + str(initial_epochs) + "\n")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
precision = history.history['precision']
val_precision = history.history['val_precision']
recall = history.history['recall']
val_recall = history.history['val_recall']
true_positives = history.history['true_positives']
val_true_positives = history.history['val_true_positives']
true_negatives = history.history['true_negatives']
val_true_negatives = history.history['val_true_negatives']
false_negatives = history.history['false_negatives']
val_false_negatives = history.history['val_false_negatives']
false_positives = history.history['false_positives']
val_false_positives = history.history['val_false_positives']

paramFile.write('\n\n')
paramFile.write('Training\n')
paramFile.write(f'Accuracy: {np.mean(acc):.3f} (+- {np.std(acc):.3f}) \n')
paramFile.write(f'Loss: {np.mean(loss):.3f} \n')
paramFile.write(f'Precision: {np.mean(precision):.3f} (+- {np.std(precision):.3f}) \n')
paramFile.write(f'Recall: {np.mean(recall):.3f} (+- {np.std(recall):.3f}) \n')
paramFile.write(f'True positives: {np.mean(true_positives):.3f} (+- {np.std(true_positives):.3f}) \n')
paramFile.write(f'True negatives: {np.mean(true_negatives):.3f} (+- {np.std(true_negatives):.3f}) \n')
paramFile.write(f'False positives: {np.mean(false_positives):.3f} (+- {np.std(false_positives):.3f}) \n')
paramFile.write(f'False negatives: {np.mean(false_negatives):.3f} (+- {np.std(false_negatives):.3f}) \n')

paramFile.write('\n\n')
paramFile.write('Validation\n')
paramFile.write(f'Accuracy: {np.mean(val_acc):.3f} (+- {np.std(val_acc):.3f}) \n')
paramFile.write(f'Loss: {np.mean(val_loss):.3f} \n')
paramFile.write(f'Precision: {np.mean(val_precision):.3f} (+- {np.std(val_precision):.3f}) \n')
paramFile.write(f'Recall: {np.mean(val_recall):.3f} (+- {np.std(val_recall):.3f}) \n')
paramFile.write(f'True positives: {np.mean(val_true_positives):.3f} (+- {np.std(val_true_positives):.3f}) \n')
paramFile.write(f'True negatives: {np.mean(val_true_negatives):.3f} (+- {np.std(val_true_negatives):.3f}) \n')
paramFile.write(f'False positives: {np.mean(val_false_positives):.3f} (+- {np.std(val_false_positives):.3f}) \n')
paramFile.write(f'False negatives: {np.mean(val_false_negatives):.3f} (+- {np.std(val_false_negatives):.3f}) \n')

paramFile.write('\n\n')
paramFile.write('Testing\n')
paramFile.write(f'Accuracy: {results[1]:.3f} \n')
paramFile.write(f'Loss: {results[0]:.3f} \n')
paramFile.write(f'Precision: {results[2]:.3f} \n')
paramFile.write(f'Recall: {results[3]:.3f}  \n')
paramFile.write(f'True positives: {results[4]:.3f}  \n')
paramFile.write(f'True negatives: {results[5]:.3f}  \n')
paramFile.write(f'False positives: {results[6]:.3f}  \n')
paramFile.write(f'False negatives: {results[7]:.3f}  \n')

paramFile.close()

print('Archivo de paramétros generado')

