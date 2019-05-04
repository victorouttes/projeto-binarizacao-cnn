import os
import cv2
import numpy as np
from preprocessing.window import Window
from preprocessing.threshold import Threshold
from dataset.resize import Resize


# Classe para transformação dos dados
class Transformation:

    # Função que transforma todas as imagens de uma pasta "folder" numa base de dados
    # utilizando um tamanho de janela. Utiliza a pasta "folder_bw" para obter o valor
    # dos labels de cada imagem.
    @staticmethod
    def images_to_dataset(folder, window_size, folder_bw=None, resize=False):
        all_images = []
        all_labels = []
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        if folder_bw:
            files_bw = [f for f in os.listdir(folder_bw) if os.path.isfile(os.path.join(folder_bw, f))]
        for file in files:
            image = cv2.imread(folder + file)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if folder_bw:
                file_starts = file.split('.')[0]  # nome do arquivo sem extensao
                file_bw = [f for f in files_bw if f.startswith(file_starts)]
                binarized = cv2.imread(folder_bw + file_bw[0], cv2.IMREAD_UNCHANGED)
                binarized[binarized > 0] = 1
                binarized = np.asarray(binarized)
            else:
                binarized = Threshold.otsu(gray)

            if resize:
                binarized = Resize.resize(binarized, 30)
                gray = Resize.resize(gray, 30)

            labels = binarized.reshape(1, -1).T
            all_labels.extend(labels)

            images = Window.sliding_with_mirror(image=gray, step=1, size=window_size)
            all_images.extend(images)

        all_images = np.asarray(all_images)
        all_images = all_images.reshape(len(all_labels), -1)

        all_labels = np.asarray(all_labels)
        all_labels = all_labels.reshape(len(all_labels), -1)
        return all_images, all_labels

    # Converte uma imagem específica em uma amostra para aplicar o modelo CNN
    @staticmethod
    def image_to_data(image, window_size, resize=False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if resize:
            gray = Resize.resize(gray, 30)
        images = Window.sliding_with_mirror(image=gray, step=1, size=window_size)
        images = np.asarray(images, dtype=np.float32)
        images = images.reshape(len(images), -1)
        return images
