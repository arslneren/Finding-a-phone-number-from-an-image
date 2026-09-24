"""Keras YSA eğitimi: tüm veri birleştirilir, %80 eğitim / %20 test ayrılır.

    python train.py --arch cnn --epochs 10
    python train.py --arch mlp --epochs 15

Test seti eğitim boyunca hiç görülmez; model seçimi (checkpoint) eğitim
kısmından ayrılan %10'luk doğrulama dilimiyle yapılır.
"""
import argparse
import json
import os

import numpy as np

# eğitimde cuDNN autotune AÇIK (kapalıyken çok daha yavaş); gpu_setup yalnızca tahmin için kapatır
os.environ.setdefault("TF_CUDNN_USE_AUTOTUNE", "1")
import gpu_setup  # noqa: F401,E402  (Windows GPU DLL'leri; TF'den önce)
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from model import BUILDERS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODELS = os.path.join(HERE, "models")


for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)


class IndexSeq(tf.keras.utils.Sequence):
    """x[idx] üzerinden batch üretir; x RAM'de bir dizi ya da diskte memmap olabilir,
    böylece %80/%20 ayrımı veriyi kopyalamadan indekslerle yapılır."""

    def __init__(self, x, y, idx, batch, train):
        self.x, self.y, self.idx, self.batch, self.train = x, y, np.array(idx), batch, train
        self.rng = np.random.default_rng(0)
        if train:
            self.rng.shuffle(self.idx)

    def __len__(self):
        return int(np.ceil(len(self.idx) / self.batch))

    def __getitem__(self, i):
        b = np.sort(self.idx[i * self.batch:(i + 1) * self.batch])  # sıralı okuma memmap'te hızlı
        x = np.asarray(self.x[b], dtype=np.float32)
        if self.train:  # açık zemin/koyu yazı ile koyu zemin/açık yazı fark etmemeli
            inv = self.rng.random(len(b)) < 0.2
            x[inv] = 255.0 - x[inv]
        return x, self.y[b].astype(np.float32)

    def on_epoch_end(self):
        if self.train:
            self.rng.shuffle(self.idx)


class Concat:
    """İki diziyi (ör. 5 GB memmap + RAM'deki zor negatifler) kopyalamadan tek dizi gibi indeksler."""

    def __init__(self, a, b):
        self.a, self.b, self.n = a, b, len(a)

    def __len__(self):
        return self.n + len(self.b)

    def __getitem__(self, idx):
        idx = np.asarray(idx)
        m = idx < self.n
        out = np.empty((len(idx),) + self.a.shape[1:], self.a.dtype)
        out[m] = self.a[idx[m]]
        out[~m] = self.b[idx[~m] - self.n]
        return out


def load_all(hard_neg=None):
    """Büyük set (big_*.npy, memmap) varsa onu, yoksa train+val.npz birleşimini kullanır.
    hard_neg: mine_negatives.py çıktısı; 'numara yok' etiketiyle havuza eklenir
    (böylece zor negatifler de %80/%20 ayrımına girer)."""
    if os.path.exists(os.path.join(DATA, "big_y.npy")):
        x, y, s = (np.load(os.path.join(DATA, "big_x.npy"), mmap_mode="r"),
                   np.load(os.path.join(DATA, "big_y.npy")), np.load(os.path.join(DATA, "big_s.npy")))
    else:
        xs, ys, ss = [], [], []
        for split in ("train", "val"):
            d = np.load(os.path.join(DATA, f"{split}.npz"))
            xs.append(d["x"]); ys.append(d["y"]); ss.append(d["s"])
        x, y, s = np.concatenate(xs), np.concatenate(ys), np.concatenate(ss)
    if hard_neg:
        h = np.load(hard_neg)
        h = h[..., None] if h.ndim == 3 else h
        x = Concat(x, h)
        y = np.concatenate([y, np.zeros(len(h), np.uint8)])
        s = np.concatenate([s, np.array(["zor_negatif"] * len(h), dtype=s.dtype)])
    return x, y, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=sorted(BUILDERS), default="cnn")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--hard_neg", help="zor negatifler (.npy), ör. data/hard_neg.npy")
    ap.add_argument("--init", help="ince ayar için başlangıç modeli (.h5)")
    a = ap.parse_args()
    os.makedirs(MODELS, exist_ok=True)
    tf.keras.utils.set_random_seed(42)

    x, y, s = load_all(a.hard_neg)
    idx = np.arange(len(y))
    i_tr, i_te = train_test_split(idx, test_size=a.test_size, stratify=y, random_state=42)
    i_tr, i_va = train_test_split(i_tr, test_size=0.1, stratify=y[i_tr], random_state=42)
    i_te = np.sort(i_te)
    y_te, s_te = y[i_te], s[i_te]
    print(f"toplam={len(y)}  eğitim={len(i_tr)}  doğrulama={len(i_va)}  test={len(i_te)} (%{a.test_size * 100:.0f})")

    model = tf.keras.models.load_model(a.init, compile=False) if a.init else BUILDERS[a.arch]()
    steps = a.epochs * int(np.ceil(len(i_tr) / a.batch))
    model.compile(tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(a.lr, steps)),
                  "binary_crossentropy", metrics=["accuracy"])
    model.summary()

    print("GPU:", tf.config.list_physical_devices("GPU") or "yok (CPU)")
    path = os.path.join(MODELS, f"digit_detector_{a.arch}.h5")  # .h5: TF 2.10 (Windows GPU) ile de uyumlu
    model.fit(IndexSeq(x, y, i_tr, a.batch, True), validation_data=IndexSeq(x, y, i_va, 256, False),
              epochs=a.epochs, verbose=2, workers=4, max_queue_size=32, callbacks=[
                  tf.keras.callbacks.ModelCheckpoint(path, monitor="val_accuracy", save_best_only=True),
                  tf.keras.callbacks.CSVLogger(os.path.join(MODELS, f"history_{a.arch}.csv"))])

    # ---- %20 test seti -----------------------------------------------------
    model = tf.keras.models.load_model(path)
    p = model.predict(IndexSeq(x, y, i_te, 256, False), verbose=0)[:, 0]
    pred = (p >= 0.5).astype(np.uint8)
    acc = float((pred == y_te).mean())
    cm = confusion_matrix(y_te, pred)
    print(f"\n=== TEST (%{a.test_size * 100:.0f}, n={len(y_te)}) — {a.arch.upper()} ===")
    print(f"Doğruluk (accuracy): {acc:.4f}")
    print(classification_report(y_te, pred, target_names=["numara yok", "numara var"], digits=4))
    print("Karışıklık matrisi [gerçek x tahmin]:")
    print(f"               tahmin:yok  tahmin:var\n  gerçek:yok   {cm[0, 0]:10d}  {cm[0, 1]:10d}"
          f"\n  gerçek:var   {cm[1, 0]:10d}  {cm[1, 1]:10d}")

    # test setindeki pozitiflerin rakam setine göre yakalanma oranı
    per_set = {}
    for name in sorted(set(s_te[y_te == 1])):
        m = (s_te == name) & (y_te == 1)
        per_set[name] = {"n": int(m.sum()), "recall": float(pred[m].mean())}
    print("\nRakam setine göre yakalama oranı (test):")
    for k, v in sorted(per_set.items(), key=lambda kv: kv[1]["recall"]):
        print(f"  {k:22s} {v['recall']:.3f}  (n={v['n']})")

    with open(os.path.join(MODELS, f"metrics_{a.arch}.json"), "w", encoding="utf-8") as f:
        json.dump({"test_accuracy": acc, "confusion_matrix": cm.tolist(),
                   "report": classification_report(y_te, pred, output_dict=True),
                   "per_set_recall": per_set}, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
