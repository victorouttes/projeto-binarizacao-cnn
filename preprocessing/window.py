import cv2
import math


# Classe de implementações de extração de janelas de imagens
class Window:

    # Dada uma imagem, retorna um array de subimagens de tamanho size, utilizando a técnica de
    # borda espelhada. 
    @staticmethod
    def sliding_with_mirror(image, step, size):
        result = []
        border = int(math.floor(size / 2))
        image = cv2.copyMakeBorder(image,
                                   top=border,
                                   bottom=border,
                                   left=border,
                                   right=border,
                                   borderType=cv2.BORDER_REFLECT
                                   )
        for x in range(0, image.shape[0] - size + 1, step):
            for y in range(0, image.shape[1] - size + 1, step):
                result.append(image[x:x + size, y:y + size].reshape(1, -1))
        return result
