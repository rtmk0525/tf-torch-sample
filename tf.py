import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential

# GPUが利用可能かどうかをチェック
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(physical_devices[0], "GPU")
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(physical_devices), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        print(f"{physical_devices=}")
        print(f"{logical_gpus=}")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

# データの生成
np.random.seed(0)
X = np.random.rand(1000, 5)
y = np.random.randint(2, size=(1000, 1))

# モデルの構築
model = Sequential()
model.add(Dense(10, input_dim=5, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# モデルのコンパイル
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# モデルのトレーニング
model.fit(X, y, epochs=10, batch_size=32)
