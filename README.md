# Ciencia de datos aplicada al diagnóstico y seguimiento de la enfermedad de Parkinson

El presente repositorio es la base del proyecto de investigación Ciencia de datos aplicada al diagnóstico y seguimiento de la enfermedad de Parkinson, desarrollado en la UNLAM entre los años 2019-2020.

El objetivo del proyecto es lograr clasificar correctamente a aquellas personas que padezcan de Parkinson mediante el uso de espectrogramas y redes neuronales convolucionales.

El lenguaje principal utilizado en este repositorio es Python. Utilizamos una variedad de packages, pero los más notables son: Tensorflow, Keras, y Librosa.

## ¿Por qué espectrogramas y CNN?

En los últimos años se han utilizado redes neuronales convolucionales para la clasificación de audio con resultados bastante alentadores. 

Si bien las CNN se enfocan exclusivamente en imágenes, algunas características del audio pueden ser representadas como una imagen. Es ahí donde entran en juego
los espectrogramas: gráficos que visualizan el espectro de frecuencias de un audio. 

En este proyecto, los espectrogramas se realizan en base a grabaciones tanto de personas que padecen Parkinson como de voluntarios que no tienen dicha enfermedad. Se busca clasificar correctamente dichos espectrogramas, y evaluar si los resultados arrojados son prometedores para el diagnóstico de la enfermedad de Parkinson.

## Scripts 

En el repositorio hay una serie de scripts notables:

* Script para la generación y pre-procesamiento de espectrogramas.
* Script que utiliza una red neuronal pre-entrenada y transfer-learning para la clasificación de imágenes.
* Script que entrena una red neuronal desde cero.
* Script de predicción en base a un modelo ya generado.