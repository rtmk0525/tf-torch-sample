import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# GPUが利用可能かどうかをチェック
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(physical_devices[0], "GPU")
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(physical_devices), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(f"Available GPUs: {physical_devices}")
print(f'Current GPU: {physical_devices[0] if len(physical_devices) > 0 else "None"}')

# MNISTデータセットの読み込み
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# データの前処理
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# モデルの構築
model = tf.keras.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

# モデルのコンパイル
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# モデルのトレーニング
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))
