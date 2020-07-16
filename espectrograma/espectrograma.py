# Necesitamos tener instalados los packages: librosa, matplotlib

import sys
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

pathAudiosTrue = "./audios/true/"
pathAudiosFalse = "./audios/false/"

#filename = "./audios/5388153.wav"
#y, sr = librosa.load(filename)
#spectrogram = librosa.feature.melspectrogram(y=y,sr=sr)#,win_length=500)

#fig, ax = plt.subplots(1, figsize=(1,1))
#fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
#ax.axis('off')
#librosa.display.specshow(librosa.power_to_db(spectrogram,
#                                             ref=np.max),
#                         y_axis='mel', fmax=8000,
#                         x_axis='time')

#librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), fmax=8000)

#plt.colorbar(format='%+2.0f dB')
#plt.title('Espectrograma')
#plt.tight_layout()
#plt.savefig('./images/spect.png')
#plt.show()


def main():
    listaFalse = os.listdir(pathAudiosFalse)
    listaTrue = os.listdir(pathAudiosTrue)

    for audio in listaFalse[1:porcentaje(len(listaFalse), 70)]:
        #crearEspectrograma(pathAudiosFalse, audio, "false")
        crearEspectrograma(pathAudiosFalse, audio, "../data/entrenamiento/false")
        
    for audio in listaFalse[porcentaje(len(listaFalse), 70):len(listaFalse)]:
        crearEspectrograma(pathAudiosFalse, audio, "../data/validacion/false")

    for audio in listaTrue[1:porcentaje(len(listaTrue), 70)]:
        crearEspectrograma(pathAudiosTrue, audio, "../data/entrenamiento/true")
    
    for audio in listaTrue[porcentaje(len(listaTrue), 70):len(listaTrue)]:
        crearEspectrograma(pathAudiosTrue, audio, "../data/validacion/true")

def porcentaje(num, porciento):   
    return int((num*porciento)/100) 


def crearEspectrograma(path, file, carpetaDestino):
    # Obtenemos el nombre dl archivo sin extension
    filename = os.path.splitext(os.path.basename(file))[0]

    try: 
        # Usamos magia del package librosa para crear el espectrograma
        # https://librosa.github.io/librosa/generated/librosa.core.load.html
        # offset: cuantos segundos nos desplazamos desde el archivo original (float)
        # duration: cuantos segundos de audio leemos (float)
        #
        # Algunos audios tienen "problemas" al inicio y al final (los pacientes no dicen A instantáneamente, hay rudios, etc), por eso
        # es que aplicamos un offset y acortamos cuántos segundos de audio leemos, para reducir la mayor cantidad de ruidos y problemas

        # Además, se nos dijo que los pacientes recién comienzan a presentar "vibraciones" en la voz luego de unos segundos, así que este cambio
        # nos ayuda a centrarnos justamente donde la voz empieza a "vibrar"
        y, sr = librosa.load(path + file, offset=2.0, duration=3.0)

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
        # win_length es para dividir cada trama del audio en ventanas, si no se indica este parámetro por defecto es igual a n_ftt
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=500)
        # Removemos los bordes del espectrograma y ajustamos su tamaño en pixeles (figsize)
        fig, ax = plt.subplots(1, figsize=(3,3))
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        # Removemos los ejes del espectrograma
        ax.axis('off')
        librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), fmax=8000)
        # Guardamos la imagen en el directorio
        #plt.savefig('./images/' + carpetaDestino + '/' + filename + '.png')
        plt.savefig(carpetaDestino + '/' + filename + '.png')
        plt.close()
    except Exception as e: 
        print(e)
main()