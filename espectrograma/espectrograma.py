# Necesitamos tener instalados los packages: librosa, matplotlib

import sys
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

pathAudiosTrue = "./audios/true/"
pathAudiosFalse = "./audios/false/"

def main():

    print("1- Espectrograma audios UNLAM")
    print("2- Espectrograma audios MPOWER") 
    num = int(input("Selecciona: "))

    if (num == 1): 
        off = 0
        dur = 1.5
        for audio in os.listdir("./audios/audios-unlam/true/"):
            crearEspectrograma("./audios/audios-unlam/true/", audio, "./images-unlam/true", off, dur)
        for audio in os.listdir("./audios/audios-unlam/false/"):
            crearEspectrograma("./audios/audios-unlam/false/", audio, "./images-unlam/false", off, dur)

    if (num == 2):
        off = 2.5
        dur = 2.0
        listaFalse = os.listdir(pathAudiosFalse)
        listaTrue = os.listdir(pathAudiosTrue)

        for audio in listaFalse[1:porcentaje(len(listaFalse), 70)]:
            crearEspectrograma(pathAudiosFalse, audio, "../data/entrenamiento/false", off, dur)
            
        for audio in listaFalse[porcentaje(len(listaFalse), 70):len(listaFalse)]:
            crearEspectrograma(pathAudiosFalse, audio, "../data/validacion/false", off, dur)

        for audio in listaTrue[1:porcentaje(len(listaTrue), 70)]:
            crearEspectrograma(pathAudiosTrue, audio, "../data/entrenamiento/true", off, dur)
        
        for audio in listaTrue[porcentaje(len(listaTrue), 70):len(listaTrue)]:
            crearEspectrograma(pathAudiosTrue, audio, "../data/validacion/true", off, dur)

    print("Todos los espectrogramas fueron creados")

def porcentaje(num, porciento):   
    return int((num*porciento)/100) 

def crearEspectrograma(path, file, carpetaDestino, off, dur):
    # Obtenemos el nombre dl archivo sin extension
    filename = os.path.splitext(os.path.basename(file))[0]

    try: 
        # offset: cuantos segundos nos desplazamos desde el archivo original (float)
        # duration: cuantos segundos de audio leemos (float)
        #
        # Algunos audios tienen "problemas" al inicio y al final (los pacientes no dicen A instantáneamente, hay rudios, etc), por eso
        # es que aplicamos un offset y acortamos cuántos segundos de audio leemos, para reducir la mayor cantidad de ruidos y problemas

        # Además, se nos dijo que los pacientes recién comienzan a presentar "vibraciones" en la voz luego de unos segundos, así que este cambio
        # nos ayuda a centrarnos justamente donde la voz empieza a "vibrar"
        y, sr = librosa.load(path + file, offset=off, duration=dur)

        # Librosa utiliza una STFT (Transformación de Fourier de Tiempo Reducido) para realizar los espectrogramas.La STFT utiliza una "ventana de tiempo".
        # 
        # Las ventanas de tiempo son muy importantes e influyen en el resultado del espectrograma, debido a que uno de los problemas que tiene STFT es que, dependiendo del tamaño de la ventana,
        # puede perder resolución de frecuencia, o resolución de tiempo.
        #
        # La "resolución de frecuencia" es lo que nos permite identificar con claridad el valor de la frecuencia en el espectrograma. Si perdemos resolución de frecuencia en un espectrograma,
        # nos va a costar distinguir el valor de la frecuencia.
        #
        # La "resolución de tiempo" nos indica el tiempo en que las frecuencias cambian. Si perdemos resolución de frecuencia, en el espectrograma no vamos a notar claramente cuándo
        # cambian esas frecuencias.
        # 
        # La regla es la siguiente: si disminuimos el tamaño de la ventana, aumenta la resolución de tiempo. Si aumentamos el tamaño de la ventana, aumenta la resolución
        # de la frecuencia.
        #
        # Parámetros:
        # n_ftt se usa para establecer el tamaño de la ventana en la STFT
        # para análisis de voz, se recomienda utilizar n_ftt = 512 (lo que da una window lenght de 23ms, similar a la que usaron en el paper)
        # win_length es para dividir cada trama del audio en ventanas, si no se indica este parámetro por defecto es igual a n_ftt
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512)
        # Removemos los bordes del espectrograma y ajustamos su tamaño en pixeles (figsize)
        fig, ax = plt.subplots(1, figsize=(3,3))
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        # Removemos los ejes del espectrograma
        ax.axis('off')
        #Power_to_db convierte un espectrograma a unidades de decibeles
        #fmax es un parámetro para definir cuál es la frecuencia máxima
        librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), fmax=8000)
        # Guardamos la imagen en el directorio
        plt.savefig(carpetaDestino + '/' + filename + '.png')
        plt.close()
    except Exception as e: 
        print(e)
main()