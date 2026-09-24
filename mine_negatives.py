"""Zor negatif madenciliği: numarasız tam boyutlu görselleri (temiz fotoğraf ya da
üstünde sadece KELİME/sembol olan fotoğraf) inference'taki gibi karolara bölüp
modelin yanlış alarm verdiği karoları toplar -> data/hard_neg.npy

    python mine_negatives.py --max 40000
"""
import argparse
import glob
import os
import random

import numpy as np
from PIL import Image, ImageDraw

from fonts import build_coverage, load_font
from predict import tiles, Detector
from prepare_data import list_images
from synth import Synth, charset_table

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CUSTOM_NEG = os.path.join(DATA, "custom_negatives")   # kendi numarasız fotoğraflarınız


def words_overlay(im, S, r):
    """Fotoğrafa rakamsız yazılar (Türkçe ilan kelimeleri, süslü Unicode harfler, semboller)."""
    d = ImageDraw.Draw(im)
    for _ in range(r.randint(1, 4)):
        pieces = S.word_pieces()
        size = r.randint(16, 90)
        x, y = r.randint(-50, im.width - 50), r.randint(0, im.height - 20)
        color = r.choice([0, 255, r.randint(0, 255)])
        stroke = r.choice([0, 0, 2])
        for ch, face in pieces:
            f = load_font(face[0], face[1], size)
            d.text((x, y), ch, font=f, fill=color, stroke_width=stroke, stroke_fill=255 - color)
            x += f.getlength(ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=40000)
    ap.add_argument("--min_score", type=float, default=0.3)
    a = ap.parse_args()
    r = random.Random(123)
    S = Synth(build_coverage(charset_table()), seed=123)
    det = Detector()
    # her 5. fotoğraf görsel düzeyindeki test (eval_images.py) için ayrılır, burada KULLANILMAZ
    photos = [p for i, p in enumerate(sorted(glob.glob(os.path.join(DATA, "flower_photos", "*", "*.jpg"))))
              if i % 5 != 0]
    # Kendi numarasız fotoğraflarınız (ör. gerçek ilan fotoğrafları): üzerlerinde HİÇ rakam olmamalı,
    # çünkü buradan çıkan her karo "numara yok" olarak eğitime girer.
    custom = list_images(CUSTOM_NEG)
    print(f"{len(photos)} flower_photos + {len(custom)} kendi fotoğrafınız ({CUSTOM_NEG})")
    photos += custom
    r.shuffle(photos)
    out, n_img, n_tiles = [], 0, 0
    for rep in range(3):                     # her foto: temiz + kelimeli varyantlar
        for p in photos:
            im = Image.open(p).convert("L")
            s = r.uniform(1.0, 3.0)          # ilan fotoğrafı boyutlarına yaklaş
            im = im.resize((int(im.width * s), int(im.height * s)), Image.BILINEAR)
            if rep > 0 or r.random() < 0.5:
                words_overlay(im, S, r)
            t = [a_ for a_, _ in tiles(im)]
            x = np.stack(t)[..., None]
            prob = det.model.predict(x.astype(np.float32), batch_size=256, verbose=0)[:, 0]
            keep = x[prob >= a.min_score]
            out.extend(keep)
            n_img += 1
            n_tiles += len(t)
            if n_img % 200 == 0:
                print(f"{n_img} görsel, {n_tiles} karo, {len(out)} zor negatif", flush=True)
            if len(out) >= a.max:
                break
        if len(out) >= a.max:
            break
    arr = np.stack(out[:a.max]).astype(np.uint8)
    np.save(os.path.join(DATA, "hard_neg.npy"), arr)
    print(f"bitti: {n_img} görsel, {n_tiles} karo, yanlış alarm oranı %{100 * len(out) / n_tiles:.1f}, "
          f"{len(arr)} zor negatif kaydedildi")


if __name__ == "__main__":
    main()
