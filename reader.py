"""Numara OKUYUCU: CRNN (CNN + çift yönlü LSTM) + CTC.

64x256'lık bir satır görselinden rakam dizisini okur. Rakam hangi Unicode
setinde yazılmış olursa olsun (𝟎 ٠ ⓪ ０ ...) çıktı normal ASCII rakamdır.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

LINE_H, LINE_W, MAX_LABEL = 64, 256, 20
BLANK = 10                 # CTC boş sınıfı (0-9 rakamlar)
T = LINE_W // 4            # zaman adımı sayısı (64)


def build_reader(width=LINE_W):
    """width=None -> değişken genişlik: eğitimde 256 px ile öğrenilen ağırlıklar
    (konvolüsyon + zaman adımı başına Dense + LSTM) her genişlikte çalışır; böylece
    uzun numaralar tam genişlikte bir şeritte tek parça okunabilir."""
    def cbr(x, f):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        return layers.ReLU()(layers.BatchNormalization()(x))

    inp = layers.Input((LINE_H, width, 1), name="image")
    x = layers.Rescaling(1 / 127.5, offset=-1)(inp)
    x = layers.MaxPooling2D()(cbr(x, 32))                    # 32 x 128
    x = layers.MaxPooling2D()(cbr(x, 64))                    # 16 x 64
    x = layers.MaxPooling2D((2, 1))(cbr(cbr(x, 128), 128))   # 8 x 64
    x = layers.MaxPooling2D((2, 1))(cbr(cbr(x, 256), 256))   # 4 x 64
    x = layers.Permute((2, 1, 3))(x)                         # genişlik = zaman
    x = layers.Reshape((-1, 4 * 256))(x)
    x = layers.Dropout(0.25)(layers.Dense(256, activation="relu")(x))
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    out = layers.Dense(BLANK + 1, activation="softmax", name="chars")(x)
    return keras.Model(inp, out, name="ysa_okuyucu_crnn")


def ctc_loss(y_true, y_pred):
    """y_true = [etiket(20, -1 dolgulu) | uzunluk]"""
    labels = tf.cast(y_true[:, :MAX_LABEL], tf.int32)
    label_len = tf.cast(y_true[:, MAX_LABEL:], tf.int32)
    input_len = tf.fill([tf.shape(y_pred)[0], 1], tf.shape(y_pred)[1])
    # CTC op boş etiketi kabul etmez: onları geçici 1 uzunlukla hesaplayıp maskeliyoruz,
    # boş etiketin gerçek kaybı = her adımda 'boş' sınıfının -log olasılığı.
    empty = tf.equal(label_len, 0)
    ctc = keras.backend.ctc_batch_cost(labels, y_pred, input_len, tf.maximum(label_len, 1))
    blank = -tf.reduce_sum(tf.math.log(y_pred[:, :, BLANK] + 1e-7), axis=1, keepdims=True)
    return tf.where(empty, blank, ctc)


def decode(probs):
    """Greedy CTC çözümü -> [(metin, güven, [(adım_baş, adım_son)...])]."""
    out = []
    best = probs.argmax(-1)
    conf = probs.max(-1)
    for b, c in zip(best, conf):
        text, confs, prev, spans = [], [], BLANK, []
        for t, k in enumerate(b):
            if k != BLANK and k != prev:
                text.append(str(k))
                confs.append(c[t])
                spans.append(t)
            prev = k
        # güven: yazılan karakterlerin ve boş adımların ortalaması (hepsi eminse yüksek)
        out.append(("".join(text), float(np.mean(confs)) if confs else float(c.mean()), spans))
    return out
