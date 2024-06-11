from tensorflow import keras

# CIFAR-10をダウンロードして訓練、テストデータをまとめる
[x_train, y_train], [x_test, y_test] = keras.datasets.cifar10.load_data()

# データのピクセル値を255で割って0.0~1.0の範囲に正規化
x_train = x_train / 255
x_test = x_test / 255

import tensorflow as tf
import numpy as np

# エンコーダーのモデル
denoise_encoder = keras.models.Sequential([
    # 入力画像にガウスノイズを加える: 出力(バッチサイズ、32, 32, 3)
    keras.layers.GaussianNoise(
        0.1,
        input_shape=[32, 32, 3]),
    
    #畳み込み層: 出力（バッチサイズ、32, 32, 32)
    keras.layers.Conv2D(32,
                        kernel_size=3,
                        padding='same',
                        activation='relu'),
    # プーリング層: 出力(バッチサイズ, 16, 16, 32)
    keras.layers.MaxPool2D(),

    keras.layers.Flatten(),

    keras.layers.Dense(512, 
                       activation='relu'),
])

# デコーダーのモデル
denoise_decoder = keras.models.Sequential([
    # 全結合層: 出力(バッチサイズ, 8192)
    keras.layers.Dense(16 * 16 * 32,
                       activation='relu',
                       input_shape=[512]),
    
    # テンソルの形状を(16, 16, 32)にする
    keras.layers.Reshape([16, 16, 32]),

    # 転置畳み込み層: 出力(バッチサイズ、32, 32, 3)
    keras.layers.Conv2DTranspose(
        filters=3,
        kernel_size=3,
        strides=2,
        padding='same',

        activation='sigmoid')
])

# サマリを出力
denoise_encoder.summary()
denoise_decoder.summary()

denoise_autoencoder = keras.models.Sequential(
    [denoise_encoder, denoise_decoder]
)

# モデルのコンパイル
denoise_autoencoder.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.Nadam(),
    metrics=['mse'])

history = denoise_autoencoder.fit(
    x_train,
    x_train,
    epochs=10,
    validation_data=(x_test, x_test)
)

import matplotlib.pyplot as plt

# 使用する画像の枚数
n_images = 10
# データセットから画像を抽出
new_images = x_test[:n_images]
# ランダムにノイズを加える
new_images_noisy = new_images + np.random.randn(n_images, 32, 32, 3) * 0.1
# オートエンコーダーに入力して復元画像を取得
new_images_denoised = denoise_autoencoder.predict(new_images_noisy)

# 描画範囲は幅600ピクセル、高さは画像の枚数10×２ピクセル
plt.figure(figsize=(6, n_images * 2))
# オリジナルの画像、ノイズを加えた画像、復元画像を出力
for index in range(n_images):
    # オリジナルの画像を出力
    plt.subplot(n_images, 3, index * 3 + 1)
    plt.imshow(new_images[index])
    plt.axis('off')
    if index == 0:
        plt.title('Original')
    # ノイズを加えた画像を出力
    plt.subplot(n_images, 3, index, * 3 + 2)
    plt.imshow(np.clip(new_images_noisy[index], 0., 1.))
    plt.axis('off')
    if index == 0:
        plt.title('Noisy')
    # 復元された画像を出力
    plt.subplot(n_images, 3, index * 3 + 3)
    plt.imshow(new_images_denoised[index])
    plt.axis('off')
    if index == 0:
        plt.title('Denoised')
plt.show()


