import numpy as np
from keras.preprocessing.image import load_img, img_to_array
#from keras.models import load_model
import tensorflow as tf 


longitud, altura = 100, 100
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
cnn = tf.keras.models.load_model(modelo)#= load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(imagepath):
    # Cargamos la imagen
    image = load_img(imagepath, target_size=(longitud, altura))
    # Convertimos la imagen en un arreglo de valores que la representa 
    image = img_to_array(image)
    # Añadimos una dimensión extra a la imagen
    image = np.expand_dims(image, axis=0)
    # Hacemos la prediccion sobre la imagen, esto nos devuelve un array
    # de dos dimensiones. El valor que la CNN cree que es correcto estará en la primera
    # dimensión. e.g. [[1,0,0]]
    array = cnn.predict(image)
    print(array)
    # Tomamos la dimensión la primera dimensión. e.g. [1,0,0]
    result = array[0]
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
    if answer == 0:
        print("Predicción: es Ada")
    elif answer == 1:
        print("Predicción: es Kira")
    return answer

predict("./ada.jpeg")