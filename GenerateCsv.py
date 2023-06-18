import numpy as np
import pandas as pd
import cv2 as cv
import shutil
import os

def calculate_hu_moments(image):
    
  momentos = cv.moments(image)
  momentos_hu = cv.HuMoments(momentos)

  momentos_hu = -np.sign(momentos_hu) * np.log10(np.abs(momentos_hu))
  momentos_hu = momentos_hu.flatten()

  return momentos_hu

def create_csv(source_path, shapes):

    df = pd.DataFrame(columns = ["Hu_Moment_1", "Hu_Moment_2", "Hu_Moment_3",
                             "Hu_Moment_4", "Hu_Moment_5", "Hu_Moment_6",
                             "Hu_Moment_7", "Labels"])

    # Etiquetas
    j = 0

    # Recorremos la lista de figuras
    for i in range(len(shapes)):

        actual_folder = os.path.join(source_path, shapes[i])

        for filename in os.listdir(actual_folder):
        
            # Obtener dirección de la imagen
            file_path = os.path.join(actual_folder, filename)

            # Abrimos imagen en escala de grises
            img = cv.imread(file_path, 0)

            # Obtenemos momentos de hu de la imagen actual
            moments = calculate_hu_moments(img)

            # Añadimos etiqueta
            moments = np.append(moments, shapes[j])
            
            # Agregamos los Momentos de Hu y su etiqueta
            df = df.append(pd.Series(moments, index=df.columns), ignore_index=True)

        j += 1

    # Guardar DataFrame en un archivo CSV
    df.to_csv('momentos_figuras.csv', index=False)

shapes = ["circle", "triangle", "square", "star"]
source_path = 'C:/Users/Yo/Documents/Adair/RN/Proyecto1/selectedShapes/'

create_csv(source_path, shapes)

# img = cv.imread(source_path, 0)

# calculate_hu_moments(img, "1")
