"""Modelin yanıldığı görselleri bir ızgarada gösterir (hata analizi).

    python errors.py --model models/digit_detector_cnn.h5
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

import gpu_setup  # noqa: F401
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "models", "digit_detector_cnn.h5"))
    ap.add_argument("--data", default=os.path.join(HERE, "data", "test_sets.npz"))
    ap.add_argument("--n", type=int, default=48)
    a = ap.parse_args()
    d = np.load(a.data)
    x, y, s = d["x"], d["y"], d["s"]
    p = tf.keras.models.load_model(a.model).predict(x.astype(np.float32), batch_size=256, verbose=0)[:, 0]
    wrong = np.where((p >= 0.5) != (y == 1))[0]
    print(f"hata: {len(wrong)}/{len(y)}  (doğruluk {1 - len(wrong) / len(y):.4f})")
    for lab, name in ((1, "kaçırılan numara (FN)"), (0, "yanlış alarm (FP)")):
        print(f"  {name}: {(y[wrong] == lab).sum()}")
    rng = np.random.default_rng(0)
    pick = rng.choice(wrong, min(a.n, len(wrong)), replace=False)
    cols, W = 8, 128
    grid = Image.new("L", (cols * (W + 4), int(np.ceil(len(pick) / cols)) * (W + 18)), 255)
    dr = ImageDraw.Draw(grid)
    for k, i in enumerate(pick):
        cx, cy = (k % cols) * (W + 4), (k // cols) * (W + 18)
        grid.paste(Image.fromarray(x[i, ..., 0]), (cx, cy))
        dr.text((cx + 2, cy + W + 2), f"{'FN' if y[i] else 'FP'} {p[i]:.2f} {s[i][:12]}", fill=0)
    out = os.path.join(HERE, "data", "errors.png")
    grid.save(out)
    print("->", out)


if __name__ == "__main__":
    main()
