import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("✅ GPU使用が有効になっています。")
    except:
        print("⚠️ GPUの初期化に失敗しました。")
else:
    print("⚠️ GPUが見つかりません。CPUで実行します。")

import shogi_game  

class Bias(layers.Layer):
    def __init__(self, input_shape):
        super(Bias, self).__init__()
        self.W = tf.Variable(
            initial_value=tf.zeros(shape=input_shape, dtype=tf.float32),
            trainable=True,
            name='bias'
        )

    def call(self, inputs):
        return inputs + self.W

# モデル構築
input_layer = keras.Input(shape=(5, 5, 23))
x = layers.Conv2D(256, 3, padding="same", activation="relu")(input_layer)
for _ in range(10):
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)

p = layers.Conv2D(2, 1, activation="relu")(x)
p = layers.Flatten()(p)
p = layers.Dense(4672, activation="softmax", name="policy_head")(p)

v = layers.Conv2D(1, 1, activation="relu")(x)
v = layers.Flatten()(v)
v = layers.Dense(256, activation="relu")(v)
v = layers.Dense(1, activation="tanh", name="value_head")(v)

model = keras.Model(inputs=input_layer, outputs=[p, v])

# ハイパーパラメータ（必要に応じて変更）
num_iterations = 10
num_self_play_games = 5

# 自己対局＋訓練ループ
for iteration in range(num_iterations):
    games = []
    for _ in range(num_self_play_games):
        game_data = shogi_game.self_play_game(model)
        games.extend(game_data)

    shogi_game.train_model(model, games)


class Bias(layers.Layer):
    def __init__(self, input_shape):
        super(Bias, self).__init__()
        self.W = tf.Variable(
            initial_value=tf.zeros(shape=input_shape, dtype=tf.float32),
            trainable=True,
            name='bias'
        )

    def call(self, inputs):
        return inputs + self.W

input_layer = keras.Input(shape=(5, 5, 23))
x = layers.Conv2D(256, 3, padding="same", activation="relu")(input_layer)
for _ in range(10):
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)

p = layers.Conv2D(2, 1, activation="relu")(x)
p = layers.Flatten()(p)
p = layers.Dense(4672, activation="softmax", name="policy_head")(
    p
)

v = layers.Conv2D(1, 1, activation="relu")(x)
v = layers.Flatten()(v)
v = layers.Dense(256, activation="relu")(v)
v = layers.Dense(1, activation="tanh", name="value_head")(v)

model = keras.Model(inputs=input_layer, outputs=[p, v])

for iteration in range(num_iterations):
    games = []
    for _ in range(num_self_play_games):
        game_data = shogi_game.self_play_game(model)
        games.extend(game_data)

    shogi_game.train_model(model, games)

model.save("model_best.h5")

