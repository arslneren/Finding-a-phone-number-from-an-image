"""Keras YSA (yapay sinir ağı) modelleri.

cnn : evrişimli YSA - asıl model (rakam şekillerini konumdan bağımsız öğrenir)
mlp : klasik tam bağlantılı YSA - karşılaştırma için temel çizgi
"""
from tensorflow import keras
from tensorflow.keras import layers

IMG = 128


def build_cnn():
    def conv(f):
        return [layers.Conv2D(f, 3, padding="same", use_bias=False),
                layers.BatchNormalization(), layers.ReLU()]

    return keras.Sequential(
        [layers.Input((IMG, IMG, 1)), layers.Rescaling(1 / 127.5, offset=-1)]
        + conv(16) + [layers.MaxPooling2D()]
        + conv(32) + conv(32) + [layers.MaxPooling2D()]
        + conv(64) + conv(64) + [layers.MaxPooling2D()]
        + conv(128) + conv(128) + [layers.MaxPooling2D()]
        + conv(256)
        # Global max-pool: "patch'in HERHANGİ bir yerinde rakam var mı?"
        + [layers.GlobalMaxPooling2D(), layers.Dropout(0.3),
           layers.Dense(1, activation="sigmoid", name="has_number")],
        name="ysa_cnn")


def build_mlp():
    return keras.Sequential([
        layers.Input((IMG, IMG, 1)),
        layers.Rescaling(1 / 255.0),
        layers.Flatten(),
        layers.Dense(512, activation="relu"), layers.Dropout(0.3),
        layers.Dense(256, activation="relu"), layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid", name="has_number"),
    ], name="ysa_mlp")


BUILDERS = {"cnn": build_cnn, "mlp": build_mlp}
