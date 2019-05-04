import cv2


# Classe para redimensionamento de imagens
class Resize:
    # Redimensionamento mantendo a proporção. Se a escala não for passada,
    # utiliza 60%
    @staticmethod
    def resize(img, scale_percent=60):
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return resized
