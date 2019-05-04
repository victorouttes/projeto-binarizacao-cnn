# Proposta de algoritmo de binarização com janela 23x23

import tensorflow as tf
import numpy as np
from dataset.transformation import Transformation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dataset.resize import Resize
import cv2
import os


pasta_treinamento = 'D:/git/projeto-binarizacao-cnn/images-train/'
pasta_gt = 'D:/git/projeto-binarizacao-cnn/images-gt/'
pasta_testes = 'D:/git/projeto-binarizacao-cnn/images-tests/'
pasta_resultado = 'D:/git/projeto-binarizacao-cnn/results/janela_23/'
pasta_resultado_calibracao = 'D:/git/projeto-binarizacao-cnn/results/janela_23_calibracao/'
janela = 23

# cria a base de dados de treinamento
images, labels = Transformation.images_to_dataset(folder=pasta_treinamento,
                                                  window_size=janela,
                                                  folder_bw=pasta_gt,
                                                  resize=True
                                                  )

# iguala a quantidade de classes (papel e tinta)
labels_tinta = labels[np.where(labels == 0)[0]]
images_tinta = images[np.where(labels == 0)[0], :]
images_papel = images[np.where(labels == 1)[0], :]
images_papel = images_papel[:images_tinta.shape[0], :]
labels_papel = np.full_like(labels_tinta, 1)

images = np.concatenate((images_tinta, images_papel))
labels = np.concatenate((labels_tinta, labels_papel))
random_index = np.random.randint(0, len(labels), len(labels))
images = images[random_index, :]
labels = labels[random_index]

# transforma os labels no formato onehot
labels = tf.keras.utils.to_categorical(labels, num_classes=2)
# padroniza a escala da base de dados
scaler = StandardScaler()
images = scaler.fit_transform(images)

# divide randomicamente a base 2/3 para treinar e 1/3 para avaliar o modelo
x_tr, x_te, y_tr, y_te = train_test_split(images,
                                          labels,
                                          test_size=0.33,
                                          stratify=labels)

# rede neural convolucional
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Reshape(target_shape=(janela, janela, 1)))
model.add(tf.keras.layers.Conv2D(
            filters=48,  # 48 mapas
            kernel_size=[5, 5],  # kernel 4x4
            activation=tf.nn.relu,  # ReLU
            padding='same'
        ))
model.add(tf.keras.layers.MaxPool2D(
            pool_size=[2, 2],  # janela
            strides=2  # passos da janela
        ))
model.add(tf.keras.layers.Conv2D(
            filters=48,  # 48 mapas
            kernel_size=[5, 5],  # kernel 4x4
            activation=tf.nn.relu,  # ReLU
            padding='same'
        ))
model.add(tf.keras.layers.MaxPool2D(
            pool_size=[2, 2],  # janela
            strides=2  # passos da janela
        ))
model.add(tf.keras.layers.Conv2D(
            filters=48,  # 48 mapas
            kernel_size=[5, 5],  # kernel 4x4
            activation=tf.nn.relu,  # ReLU
            padding='same'
        ))
model.add(tf.keras.layers.MaxPool2D(
            pool_size=[2, 2],  # janela
            strides=2  # passos da janela
        ))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(200, activation=tf.nn.relu, input_dim=2*2*48))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
adam = tf.keras.optimizers.Adam(lr=0.01, decay=1e-6)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_tr, y_tr, batch_size=1000, epochs=5, verbose=1)

# avalia o modelo
score = model.evaluate(x_te, y_te, batch_size=1000)
print('<<<<<< SCORE NA AVALIACAO >>>>> : ', str(score))

# faz a binarização nas imagens de teste
files = [f for f in os.listdir(pasta_testes)]
for file in files:
    print('Previsao da foto: ', file)
    im = cv2.imread(pasta_testes + file)
    im = Resize.resize(im, 30)
    print('Leu foto...')
    dados = Transformation.image_to_data(im, window_size=janela)
    dados = scaler.transform(dados)
    print('Transformou os dados...')
    pixels = model.predict(dados)

    # pixels sem calibração
    pixels_n = np.argmax(pixels, axis=1)
    pixels_n = np.asarray([0 if p == 0 else 255 for p in pixels_n])
    imagem_bw_n = pixels_n.reshape(im.shape[0], im.shape[1])
    cv2.imwrite(pasta_resultado + file, imagem_bw_n)

    # pixels com calibração
    pixels_prob = np.asarray([0 if p_tinta >= 0.9 else 255 for p_tinta, p_papel in pixels])
    imagem_bw_prob = pixels_prob.reshape(im.shape[0], im.shape[1])
    cv2.imwrite(pasta_resultado_calibracao + file, imagem_bw_prob)

    print('Previsão completa!')
