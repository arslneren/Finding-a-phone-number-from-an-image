"""Okuyucu (CRNN+CTC) eğitimi — %80 eğitim / %20 test.

    python train_reader.py --epochs 15
"""
import argparse
import json
import os

import numpy as np

# eğitimde cuDNN autotune AÇIK (kapalıyken epoch ~2.4x yavaş); gpu_setup yalnızca tahmin için kapatır
os.environ.setdefault("TF_CUDNN_USE_AUTOTUNE", "1")
import gpu_setup  # noqa: F401,E402
import tensorflow as tf
from sklearn.model_selection import train_test_split

from reader import build_reader, ctc_loss, decode, MAX_LABEL

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODELS = os.path.join(HERE, "models")

for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)


class LineSeq(tf.keras.utils.Sequence):
    def __init__(self, x, lab, idx, batch, train):
        self.x, self.lab, self.idx, self.batch, self.train = x, lab, np.array(idx), batch, train
        self.rng = np.random.default_rng(0)
        if train:
            self.rng.shuffle(self.idx)

    def __len__(self):
        return int(np.ceil(len(self.idx) / self.batch))

    def __getitem__(self, i):
        b = np.sort(self.idx[i * self.batch:(i + 1) * self.batch])
        x = np.asarray(self.x[b], dtype=np.float32)
        if self.train:
            inv = self.rng.random(len(b)) < 0.2
            x[inv] = 255.0 - x[inv]
        lab = self.lab[b].astype(np.float32)
        length = (self.lab[b] >= 0).sum(1, keepdims=True).astype(np.float32)
        return x, np.concatenate([np.maximum(lab, 0), length], 1)

    def on_epoch_end(self):
        if self.train:
            self.rng.shuffle(self.idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--init", help="ince ayar için başlangıç ağırlıkları (.h5)")
    ap.add_argument("--out", default="digit_reader.h5", help="models/ altındaki çıktı dosyası")
    a = ap.parse_args()
    tf.keras.utils.set_random_seed(42)

    x = np.load(os.path.join(DATA, "lines_x.npy"), mmap_mode="r")
    lab = np.load(os.path.join(DATA, "lines_y.npy"))
    idx = np.arange(len(lab))
    i_tr, i_te = train_test_split(idx, test_size=0.2, random_state=42)
    i_tr, i_va = train_test_split(i_tr, test_size=0.1, random_state=42)
    i_te = np.sort(i_te)
    print(f"toplam={len(lab)}  eğitim={len(i_tr)}  doğrulama={len(i_va)}  test={len(i_te)} (%20)")

    model = build_reader()
    if a.init:
        model.load_weights(a.init)
    steps = a.epochs * int(np.ceil(len(i_tr) / a.batch))
    model.compile(tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(a.lr, steps), clipnorm=5.0),
                  loss=ctc_loss)
    path = os.path.join(MODELS, a.out)
    model.fit(LineSeq(x, lab, i_tr, a.batch, True), validation_data=LineSeq(x, lab, i_va, 64, False),
              epochs=a.epochs, verbose=2, workers=4, max_queue_size=32, callbacks=[
                  tf.keras.callbacks.ModelCheckpoint(path, save_best_only=True),
                  tf.keras.callbacks.CSVLogger(os.path.join(MODELS, "history_reader.csv"))])

    model = tf.keras.models.load_model(path, compile=False)
    probs = model.predict(LineSeq(x, lab, i_te, 64, False), verbose=0)
    pred = [t for t, _, _ in decode(probs)]
    true = ["".join(str(v) for v in row if v >= 0) for row in lab[i_te]]
    exact = np.mean([p == t for p, t in zip(pred, true)])
    has = np.array([len(t) > 0 for t in true])
    exact_num = np.mean([p == t for p, t, h in zip(pred, true, has) if h])
    # karakter doğruluğu (1 - normalize edit mesafesi)
    def ed(s, t):
        d = list(range(len(t) + 1))
        for i, cs in enumerate(s, 1):
            prev, d[0] = d[0], i
            for j, ct in enumerate(t, 1):
                prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (cs != ct))
        return d[-1]
    cer = sum(ed(p, t) for p, t in zip(pred, true)) / max(1, sum(len(t) for t in true))
    print(f"\n=== OKUYUCU TEST (%20, n={len(i_te)}) ===")
    print(f"Tam doğru okunan satır:         {exact:.4f}")
    print(f"Numaralı satırlarda tam doğru:  {exact_num:.4f}")
    print(f"Karakter hata oranı (CER):      {cer:.4f}")
    for p, t in list(zip(pred, true))[:15]:
        print(f"  gerçek={t:14s} okunan={p:14s} {'OK' if p == t else 'X'}")
    with open(os.path.join(MODELS, "metrics_reader.json"), "w") as f:
        json.dump({"exact": float(exact), "exact_numbers": float(exact_num), "cer": float(cer)}, f, indent=1)


if __name__ == "__main__":
    main()
